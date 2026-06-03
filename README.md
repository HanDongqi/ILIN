# ILIN: Iteration-Guided Logical Injection Network for Complex Query Answering on Knowledge Graphs

This repository contains the official PyTorch implementation of the paper **"ILIN: Iteration-Guided Logical Injection Network for Complex Query Answering on Knowledge Graphs"** (Accepted by the *Journal of Intelligent Information Systems (JIIS)*).

In this work, we propose **ILIN**, a novel framework that reshapes the complex query answering (CQA) process from a static, single-pass aggregation into a dynamic, multi-round negotiation. To effectively resolve intrinsic logical conflicts (especially negation) and capture higher-order dependencies, ILIN introduces three core innovations: an **Iterative Message Refinement** mechanism powered by Self-Attention and GRUs, a **Relation-Conditioned Logical Injection** module for context-aware polarity processing, and a **Dynamic Task Weighting** strategy to stabilize multi-task optimization.

## 📌 Citation
If you find our code or paper useful for your research, please kindly cite our work:

```bibtex
@article{han2026ilin,
  title={{ILIN}: Iteration-Guided Logical Injection Network for Complex Query Answering on Knowledge Graphs},
  author={Han, Dongqi and Yu, Jinghua and Lu, Hu},
  journal={Journal of Intelligent Information Systems},
  year={2026},
  publisher={Springer}
}
```
*(Note: The BibTeX will be updated with volume/page numbers and DOI once officially published online.)*

## 🙏 Acknowledgements

This codebase is built upon the excellent work of previous researchers in the CQA domain. We would like to express our sincere gratitude to:
* The [**CLMPT**](https://github.com/qianlima-lab/CLMPT/tree/master) repository, which provided the foundational message-passing framework for our implementation.
* The [**CQD**](https://github.com/uclnlp/cqd) repository for providing the pre-processed datasets and the pre-trained neural link predictor (ComplEx) checkpoints used in our experiments.

## 🚀 Usage Instructions

### 1. Prepare Datasets and Checkpoints

We provide a script to automatically prepare all required data. Specifically, you can run:

```bash
sh script_prepare.sh
```

By running this script:
1. The benchmark datasets (FB15k, FB15k-237, NELL995) will be automatically downloaded into the `./data` folder and converted to the specific format required by the model.
2. The checkpoints for the pre-trained neural link predictor, originally released by [CQD](https://github.com/uclnlp/cqd), will be properly downloaded and loaded into the `./pretrain` folder.

### 2. Train ILIN

To train the ILIN model on different benchmark datasets, we provide separate shell scripts with default hyperparameters (including the configured refinement steps $K=2$). You can easily start the training process by running the corresponding script:

```bash
python3 train_gnn.py \
  --task_folder data/FB15k-237-betae \
  --output_dir log/fb15k-237/ilin \
  --checkpoint_path pretrain/cqd/FB15k-237-model-rank-1000-epoch-100-1602508358.pt \
  --agg_func mean \
  --epoch 100 \
  --reasoner clmpt \
  --embedding_dim 1000 \
  --hidden_dim 8192 \
  --device cuda:0 \
  --batch_size 512 \
  --learning_rate 1e-4 \
  --num_layers 1
```

### 3. Evaluate and Summarize Results

Once the training is complete, the evaluation logs will be saved. We provide a Python script `./read_eval_from_log.py` to automatically parse the log files and summarize the model's performance metrics (e.g., MRR on different query structures).

For example, to evaluate the trained model on FB15k-237, run:

```bash
python3 read_eval_from_log.py --log_file log/FB15k-237/ilin/output.log
```
*(Please adjust the `--log_file` path according to your actual log directory structure. We have updated the example path to reflect the `ilin` directory).*
