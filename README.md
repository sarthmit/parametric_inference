## Amortized In-Context Parametric Inference

This repository contains the official implementation for the papers:

- [**Amortized In-Context Bayesian Posterior Estimation**](https://arxiv.org/abs/2502.06601)
- **In-Context Parametric Inference: Point or Distribution Estimators?**


## Overview
This repository provides code for both **in-distribution evaluation** and **model misspecification experiments**:
- **In-distribution evaluation:** Includes training and evaluation of both **fixed-dimensional** and **variable-dimensional** estimators.
- **Model misspecification experiments:** Covers synthetic model misspecification and applications to real-world tabular tasks.

## Running Experiments
The experiments were run on python 3.9 with PyTorch version 2.2 and were mostly run on RTX8000 GPU.

### Fixed-Dimensional & Variable-Dimensional Experiments
The following experiments, with different configurations of probabilistic models, training objectives and modeling setup, can be run through the following script

```python train.py```

where the key arguments can be set as described below.

#### Key Arguments:
- `--setup fixed/variable`: Choose between **fixed-dimensional** and **variable-dimensional** estimators.
- `--model Flow/Vanilla`: 
  - `Flow`: Uses a discrete-time normalizing flow model.
  - `Vanilla`: Used in all other cases.
- `--encoder DeepSets/Transformer/GRU`: Select backbone architecture for dataset conditioning.
  - **Recommended:** `Transformer`.
- `--mode train/eval/baseline`: Specify whether to train, evaluate a trained model, or obtain baseline performance.
- `--objective forward/mle/diffusion/...`: Choice of modeling objective (e.g., minimizing forward KL divergence, MLE, or score-based diffusion modeling).
- `--experiment gaussian/linear_regression/...`: Define the likelihood model whose parameters need to be inferred.
- `--dim 2/100/...`: Set the observation dimensionality, which defines the parameter space.

In addition, similar arguments are provided for number of classes (`--num_classes`), number of mixtures (`--n_mixtures`), the range for the number of observations provided in-context (`--min_len/max_len`) and the dimensionalities of tasks to train on (`--min_param_len/max_param_len`) for variable-dimensional case. Finally, the `--iters` argument describes the number of training iterations for the in-context learner.

### Model Misspecification Experiments
To launch model misspecification experiments, run:
```
python train_misspec.py
```
with an additional argument in conjunction to those defined above
- `--train_data linear/gp/...`: Specifies the training data used for the amortized estimator.

### Inference on Tabular Tasks
Inference on real-world tabular tasks requires a pre-trained amortized in-context estimator, which is assumed to have been trained in the variable-dimensional setup and saved in appropriate directories:
```
python train_tabular.py 
```
with an additional argument describing which real-world dataset to use
- `--dataset_idx`: Index of the tabular dataset for evaluation and fine-tuning.

The datasets are pre-processed and provided in the `data/` directory.

### Configuration & Arguments
All relevant arguments, including choices for probabilistic models, dimensionality settings, and other hyperparameters, are defined in [`args.py`](args.py).

## Acknowledgement and Contact
If you find this repository useful, please cite our work:
```bibtex
@article{mittal2025amortized,
  title={Amortized In-Context Bayesian Posterior Estimation},
  author={Mittal, Sarthak and Bracher, Niels Leif and Lajoie, Guillaume and Jaini, Priyank and Brubaker, Marcus},
  journal={arXiv preprint arXiv:2502.06601},
  year={2025}
}
```

For questions or collaborations, please open an issue or reach out to us.

⭐ If you find this repository helpful, consider giving it a star!