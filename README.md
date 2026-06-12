# CHOPPERS

Code and resources for the paper:
**"Chopping and distilling variational autoencoders for real-time anomaly detection in high energy physics"**

## Overview

This repository contains code for two compression strategies applied to VAE-based anomaly detection triggers:
1. **Chopping** — using only the VAE encoder (latent space) for inference, removing the decoder
2. **Knowledge Distillation** — training smaller student models (NN and BDT) to regress the teacher anomaly score

Both strategies are studied on two datasets:

| Folder | Dataset | VAE type |
|---|---|---|
| `JetImages/` | Jet Images (CNN VAE) | Image-based, 24×24 pixels |
| `MHz_40/` | 40 MHz trigger dataset (DNN VAE) | Kinematic features |

## Requirements

- Python 3.8+, TensorFlow/Keras, ROOT with TMVA
- [`fwXmachina`](https://github.com/tae-min-hong/fwX) for BDT hardware conversion
- `hls4ml` for NN hardware conversion

## Citation

If you use this code, please cite our paper (link to be added upon publication).
