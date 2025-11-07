# In-Context Learning Replication

Replication of "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes" (Garg et al., NeurIPS 2022)

## Project Structure

```
.
├── src/
│   ├── data/           # Data generation modules
│   ├── models/         # Transformer model architecture
│   ├── training/       # Training loop and utilities
│   └── evaluation/     # Evaluation and baseline implementations
├── configs/            # Configuration files
├── checkpoints/        # Model checkpoints
├── results/            # Evaluation results and plots
└── scripts/            # Training and evaluation scripts
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run small test locally
python scripts/train.py --config configs/test_local.yaml

# Run full training on AWS
python scripts/train.py --config configs/train_linear_20d.yaml
```

## Paper Reference

- Paper: https://arxiv.org/abs/2208.01066
- Original repo: https://github.com/dtsip/in-context-learning

## Training Progress

- [ ] Linear functions (20D) - 500k steps
- [ ] Distribution shift experiments
- [ ] Sparse linear functions
- [ ] Decision trees
- [ ] 2-layer neural networks
