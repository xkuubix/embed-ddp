# Project overview

This repository implements Distributed Data Parallel (DDP) training for the EMBED (Emory Breast Imaging Dataset) classification pipeline. It provides utilities, configuration (#TODO), and example launch scripts to train models across multiple GPUs/nodes using PyTorch DDP.

Dataset: https://registry.opendata.aws/emory-breast-imaging-dataset-embed/

## Using pytorch-ddp.sh with Slurm (sbatch)

Quick instructions to queue a DDP job that runs the repo's pytorch-ddp.sh.
Submit:
```
sbatch pytorch-ddp.sh
```
## TODO List
- Add config file: configs/config.yaml.
- Implement data split: use cohort 1 for training+validation, cohort 2 as test.
- Add modulo-batch validation logic to training loop.


