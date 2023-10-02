Repository containing the code base for training the multi-agent coordination policy used in the paper "[A Framework for Real-World Multi-Robot Systems Running Decentralized GNN-Based Policies](https://arxiv.org/abs/2111.01777)".

Supplementary video material:

[![Video preview](https://img.youtube.com/vi/kcmr6RUgucw/0.jpg)](https://www.youtube.com/watch?v=kcmr6RUgucw)

## Citation
If you use any part of this code in your research, please cite our paper:

```
@inproceedings{blumenkamp2023visualnv,
  title={See What the Robot Can't See: Learning Cooperative Perception for Visual Navigation},
  author={Blumenkamp, Jan and Li, Qingbiao and Wang, Binyu and Liu, Zhe and Prorok, Amanda},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year={2023},
  organization={IEEE}
}
```

## Setup
Install conda and configure to use with (Libmamba)[https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community]. Then run `conda env create -f environment.yml` to install the dependencies. Activate the environment as instructed.

## Simulation Dataset
### Pre-generated dataset
WIP. The link to the dataset will be provided.

### Generate dataset using Webots and Docker
WIP.

## Train with simulation data
Start training by running from the root of this repository:
```
python3 -m train_sim fit --config configs/sim.yaml --data.data_path <path/to/extracted/sim/dataset/folder> --config configs/logging_sim.yaml
```
Optionally, if you wish to run without logging, remove the second `config` argument. When using logging, the model artifact will be uploaded to wandb.

## Obtaining models
### Pre-trained models.
WIP. The link to the pre-trained models will be provided.

### Download trained models
Find the run id from Wandb (this is a cryptic string and not the run name displayed). Run `python3 util/wandb_download_best_model.py <wandb group name> <run id>`.

## Evaluate in Simulation
WIP.

## Sim-to-Real Dataset
WIP. The link to the dataset will be provided.

## Train Sim-to-Real
Start training by running from the root of this repository:
```
python3 -m train_sim2real fit --config configs/sim2real.yaml --model.model_checkpoint <model checkpoint> --data.data_path <path/to/extracted/sim2real/dataset/folder> --config configs/logging_sim2real.yaml
```
Optionally, if you wish to run without logging, remove the second `config` argument. When using logging, the model artifact will be uploaded to wandb.
