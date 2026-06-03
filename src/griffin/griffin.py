# from jaxtyping import Array, Float32
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Literal
from .rmsnorm import RMSNorm


# Based on Griffin implementation from https://github.com/knotgrass/Griffin/tree/main/griffin
class Gated_MLP_block(nn.Module):
    def __init__(
        self,
        D: int,
        expansion_factor: int = 3,
        approximate: Literal["none", "tanh"] = "none",
    ) -> None:
        super().__init__()
        self.D = D
        self.M = expansion_factor
        # Use approximate='tanh' for GELU approximation if needed, else 'none'
        self.gelu = nn.GELU(approximate=approximate)
        self.p1 = nn.Linear(in_features=D, out_features=D*self.M)
        self.p2 = nn.Linear(in_features=D, out_features=D*self.M)
        self.p3 = nn.Linear(in_features=D*self.M, out_features=D)

    def forward(self, x:Tensor) -> Tensor:
        # left branch
        x1 = self.p1(x)
        x1 = self.gelu(x1)

        # right branch
        x2 = self.p2(x)

        y = x1 * x2  # element-wise multiplication
        y = self.p3(y)

        return y


# Based on Griffin implementation from https://github.com/knotgrass/Griffin/tree/main/griffin
class Temporal_Conv1D(nn.Module):
    def __init__(self, D: int, kernel_size: int=4):
        super().__init__()
        # A separable 1D convolution:
        # - Input channels = output channels = D
        # - groups = D makes it depthwise (channel-wise) convolution.
        self.conv = nn.Conv1d(
            in_channels=D,
            out_channels=D,
            kernel_size=kernel_size,
            groups=D,
            bias=False,
            padding=kernel_size // 2  # Changed padding back to kernel_size // 2
        )

    def forward(self, x: Tensor) -> Tensor:
        # https://chatgpt.com/share/67692a55-5224-8005-a271-80067aa3bcbb
        # x: (B, T, D)
        # B = Batch size
        # T = Sequence length
        # D = Feature dimension
        # x: (B, T, D)
        # Transpose to (B, D, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)  # (B, D, T)
        # Transpose back to (B, T, D)
        x = x.transpose(1, 2)
        return x


# Based on Griffin implementation from https://github.com/knotgrass/Griffin/tree/main/griffin
class Real_Gated_Linear_Recurrent_Unit(nn.Module):
    c = 8.0

    def __init__(
        self, D: int, expansion_factor: int | float = 3, device=None, dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.dtype = dtype
        self.device = device
        self.D = D
        self.hidden_dim = int(round(D * expansion_factor))

        self.Wa = nn.Parameter(torch.empty(self.hidden_dim, D, **factory_kwargs)) # Gate r parameters
        self.Wx = nn.Parameter(torch.empty(self.hidden_dim, D, **factory_kwargs)) # Gate i parameters
        self.proj_x = nn.Parameter(torch.empty(self.hidden_dim, D, **factory_kwargs)) # Input projection parameters
        self.ba = nn.Parameter(torch.empty(self.hidden_dim, **factory_kwargs))
        self.bx = nn.Parameter(torch.empty(self.hidden_dim, **factory_kwargs))
        # self.b_proj_x = nn.Parameter(torch.empty(self.hidden_dim, **factory_kwargs)) # Optional bias for proj_x
        self.Lambda = nn.Parameter(torch.empty(self.hidden_dim, **factory_kwargs))  # Λ
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # https://tinyurl.com/lecuninit
        # Initialize weights similar to other linear layers
        nn.init.normal_(self.Wa, mean=0, std=1 / (self.D ** 0.5))
        nn.init.normal_(self.Wx, mean=0, std=1 / (self.D ** 0.5))
        nn.init.normal_(self.proj_x, mean=0, std=1 / (self.D ** 0.5))

        # init bias (ba, bx are initialized to zeros by default if not specified otherwise)
        nn.init.zeros_(self.ba) # Explicitly initialize biases to zero
        nn.init.zeros_(self.bx)
        # if hasattr(self, 'b_proj_x'): nn.init.zeros_(self.b_proj_x)
        # init Λ
        nn.init.uniform_(self.Lambda.data, a=0.9, b=0.999) # Initialize the data of the parameter
        # Modify the parameter's data in-place instead of reassigning
        with torch.no_grad():
            new_lambda_data = -torch.log((self.Lambda.data ** (-1.0 / self.c)) - 1.0)
            self.Lambda.copy_(new_lambda_data)

    # Note: foresee method seems identical to forward in the provided code.
    # Keeping both for now, but they might be intended to be different or one might be redundant.
    def foresee(self, x: torch.Tensor  # (batch_size, sequence_length, dim)
                ) -> torch.Tensor:     # (batch_size, sequence_length, dim)

        batch_size, sequence_length = x.shape[:2]
        ht = torch.zeros(batch_size, self.hidden_dim,
                         dtype=self.dtype, device=self.device)
        y = torch.empty(batch_size, sequence_length, self.hidden_dim,
                        dtype=self.dtype, device=self.device)
        for t in range(sequence_length):
            xt = x[:, t, :]
            rt = torch.sigmoid(F.linear(xt, self.Wa, self.ba))  # (1)
            it = torch.sigmoid(F.linear(xt, self.Wx, self.bx))  # (2)

            # Appendix A - https://github.com/kyegomez/Griffin/issues/6
            log_at = - self.c * F.softplus(-self.Lambda, beta=1, threshold=20) * rt # (6)
            at = torch.exp(log_at)

            # Project xt before multiplying with it
            xt_proj = F.linear(xt, self.proj_x) # Removed bias for simplicity, add self.b_proj_x if needed
            # ht = at * ht + torch.sqrt(1 - at**2) * (it * xt_proj) # (4) - Use projected xt
            
            # Clamp for numerical stability before taking the square root
            sqrt_input = 1 - at**2
            sqrt_val = torch.sqrt(torch.clamp(sqrt_input, min=1e-8))
            ht = at * ht + sqrt_val * (it * xt_proj)

            y[:, t, :] = ht

        return y

    def forward(self, x: torch.Tensor  # (batch_size, sequence_length, dim)
                ) -> torch.Tensor:     # (batch_size, sequence_length, dim)

        batch_size, sequence_length = x.shape[:2]
        ht = torch.zeros(batch_size, self.hidden_dim, dtype=self.dtype, device=self.device)
        y = []
        for t in range(sequence_length):
            xt = x[:, t, :]
            rt = torch.sigmoid(F.linear(xt, self.Wa, self.ba))  # (1)
            it = torch.sigmoid(F.linear(xt, self.Wx, self.bx))  # (2)

            # Appendix A - https://github.com/kyegomez/Griffin/issues/6
            log_at = - self.c * F.softplus(-self.Lambda, beta=1, threshold=20) * rt # (6)
            at = torch.exp(log_at)

            # Project xt before multiplying with it
            xt_proj = F.linear(xt, self.proj_x) # Removed bias for simplicity, add self.b_proj_x if needed
            # ht = at * ht + torch.sqrt(1 - at**2) * (it * xt_proj) # (4) - Use projected xt

            # Clamp for numerical stability before taking the square root
            # Ensure 1 - at**2 does not become negative due to floating point precision
            sqrt_input = 1 - at**2
            # Use a small epsilon so the input does not become zero when at is exactly 1
            sqrt_val = torch.sqrt(torch.clamp(sqrt_input, min=1e-8))
            ht = at * ht + sqrt_val * (it * xt_proj)

            y.append(ht.unsqueeze(1))

        y = torch.cat(y, dim=1)

        return y


# Alias for potential backward compatibility or alternative naming
RGLRU = Real_Gated_Linear_Recurrent_Unit


# Based on Griffin implementation from https://github.com/knotgrass/Griffin/tree/main/griffin
class Recurrent_block(nn.Module):
    def __init__(self, D:int, D_rnn:int, device=None, # Add device argument
                 approximate:Literal['none', 'tanh']='none'):
        super().__init__()
        self.D = D
        self.D_rnn = D_rnn
        # Use approximate='tanh' for GELU approximation if needed, else 'none'
        self.gelu = nn.GELU(approximate=approximate)
        # Linear layers will be moved to device by .to(device) on the parent module
        self.p1 = nn.Linear(in_features=D, out_features=D_rnn)
        self.p2 = nn.Linear(in_features=D, out_features=D_rnn)
        self.p3 = nn.Linear(in_features=D_rnn, out_features=D)
        # Initialize Conv1D and RGLRU with D_rnn as they operate on the expanded dimension
        # Conv1D parameters will be moved by .to(device)
        self.separableConv1D = Temporal_Conv1D(D_rnn, kernel_size=4)
        # rnn_rglru_output_dim = int(round(D_rnn * 3)) # RGLRU的默认expansion factor是3
        # Pass device explicitly to RGLRU for ht initialization
        self.rglru = RGLRU(D_rnn, 1, device=device) # Pass device here

    def forward(self, x:Tensor) -> Tensor:
        # left branch
        x1 = self.p1(x)
        x1 = self.gelu(x1)
        seq_len = x1.size(1) # Get sequence length from x1

        # right branch
        x2 = self.p2(x)
        x2 = self.separableConv1D(x2)
        # Crop x2 to match x1's sequence length
        if x2.size(1) > seq_len:
            x2 = x2[:, :seq_len, :]
        elif x2.size(1) < seq_len:
            # This case is less likely with padding=kernel_size//2 but added for robustness
            # If needed, more sophisticated padding can be implemented
             raise RuntimeError(f"Temporal_Conv1D output sequence length ({x2.size(1)}) is shorter than input ({seq_len})") # Or pad x2

        x2 = self.rglru(x2)

        y = x1 * x2  # element-wise multiplication
        y = self.p3(y)
        return y


# Based on Griffin implementation from https://github.com/knotgrass/Griffin/tree/main/griffin
class Residual_block(nn.Module):
    def __init__(self,
                 D:int,
                 mlp_expansion_factor: int = 3,
                 rnn_expansion_factor: float = 4/3,
                 approximate_gelu: Literal['none', 'tanh'] = 'none',
                 device=None, # Add device argument
                 dropout: float = 0.0): # Add dropout argument
        super().__init__()
        self.mlp = Gated_MLP_block(D, expansion_factor=mlp_expansion_factor, approximate=approximate_gelu) # Gated_MLP_block params moved by .to(device)
        # Pass device to Recurrent_block
        self.tmb = Recurrent_block(D, D_rnn=int(rnn_expansion_factor * D), approximate=approximate_gelu, device=device) # Pass device here
        self.rmsnorm1 = RMSNorm(d=D) # RMSNorm doesn't seem to need device explicitly
        self.rmsnorm2 = RMSNorm(d=D)
        self.dropout_rec = nn.Dropout(dropout)
        self.dropout_mlp = nn.Dropout(dropout)

    def forward(self, x:Tensor) -> Tensor:
        # Recurrent part with pre-norm and residual connection
        residual = x
        x_norm = self.rmsnorm1(x)
        x_rec = self.tmb(x_norm)
        x_rec = self.dropout_rec(x_rec) # Apply dropout
        x = residual + x_rec

        # MLP part with pre-norm and residual connection
        residual = x
        x_norm = self.rmsnorm2(x)
        x_mlp = self.mlp(x_norm)
        x_mlp = self.dropout_mlp(x_mlp) # Apply dropout
        x = residual + x_mlp

        return x

# Main Griffin Model Class
class Griffin(nn.Module):
    def __init__(
        self,
        D: int,                     # Model dimension
        depth: int,                 # Number of residual blocks (layers)
        mlp_expansion_factor: int = 3, # Expansion factor for Gated MLP
        rnn_expansion_factor: float = 4/3, # Expansion factor for Recurrent block's hidden dim
        approximate_gelu: Literal['none', 'tanh'] = 'none', # GELU approximation
        device = None,               # Device for tensor allocation
        dropout: float = 0.0         # Dropout rate
    ):
        super().__init__()
        self.D = D
        self.depth = depth
        self.device = device
        # self.dropout_val = dropout # Store dropout rate if needed elsewhere, or pass directly

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Residual_block(
                D=D,
                mlp_expansion_factor=mlp_expansion_factor,
                rnn_expansion_factor=rnn_expansion_factor,
                approximate_gelu=approximate_gelu,
                device=device,
                dropout=dropout # Pass dropout to Residual_block
            ))

        self.norm = RMSNorm(d=D) # Final normalization layer

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Griffin model.
        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, D)
        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, D)
        """
        # Ensure input tensor is on the correct device
        if self.device:
            x = x.to(self.device)

        # Pass input through all residual blocks
        for layer in self.layers:
            x = layer(x)

        # Apply final normalization
        x = self.norm(x)

        return x