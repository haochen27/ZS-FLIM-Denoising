# Zero-Shot Denoising for Fluorescence Lifetime Imaging Microscopy with Intensity-Guided Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/pytorch-1.8+-red.svg)](https://pytorch.org/)

This repository implements a novel zero-shot learning approach for denoising in Fluorescence Lifetime Imaging Microscopy (FLIM) data using intensity-guided learning.

## Overview

FLIM captures multi-channel data that simultaneously records both lifetime and intensity distributions, providing critical insights into molecular microenvironments. However, FLIM imaging quality is fundamentally limited by low photon counts from biological samples, introducing shot noise that distorts accurate lifetime measurements.

Our method leverages pre-trained intensity denoising models to guide the refinement of lifetime components across multiple channels. Unlike conventional approaches, our zero-shot framework doesn't require paired noisy-clean training data for FLIM channels, while preserving inherent correlations to maintain biologically meaningful data.

## Key Contributions

- A zero-shot framework for denoising without paired ground truth data using intensity-lifetime correlations to preserve their physical relationships by applying an intensity denoising model as a structural prior
- A comprehensive loss function design with intensity guidance and channel-wise constraints to ensure effective denoising while preserving lifetime information
- Implementation of a dual-encoder network architecture with separate processing paths for each channel

## Requirements

```
python 3.8+
pytorch 1.8+
numpy
matplotlib
scikit-image
tifffile
tqdm
argparse
```

## Installation

```bash
# Clone the repository
git clone https://github.com/haochen27/ZS-FLIM-Denoising.git
cd ZS-FLIM-Denoising

# Install dependencies
pip install -r requirements.txt
```

## Dataset

### FLIM Dataset
The FLIM dataset used in this project is available at:
[FLIM Dataset Google Drive](https://drive.google.com/drive/folders/1VX73DctMlUL7XtVyA3zU1ld4aFFGoPzk?usp=sharing)

### Pre-training Dataset
For pre-training the intensity denoising model, we use the Poisson-Gaussian denoising dataset with real fluorescence microscopy images:

```
@inproceedings{zhang2019poisson,
  title={A poisson-gaussian denoising dataset with real fluorescence microscopy images},
  author={Zhang, Yide and Zhu, Yinhao and Nichols, Evan and Wang, Qingfei and Zhang, Siyuan and Smith, Cody and Howard, Scott},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11710--11718},
  year={2019}
}
```

### Dataset Structure

Organize your FLIM dataset as follows:

```
FLIM_dataset/
├── BPAE/
│   ├── BPAE1/
│   │   ├── AG1/
│   │   │   ├── imageI.tif  # I image
│   │   │   ├── imageG.tif  # g image
│   │   │   ├── imageS.tif  # s image  
│   │   └── ...
│   └── ...
└── 3T3/
    └── ...
```

## Usage

### Basic Training

#### FLIM Zero-Shot Denoising
```bash
python main.py --FLIM --root './FLIM_dataset' --type './BPAE/BPAE1/AG1' --epochs 800 --lr 1e-3 --alpha 3e-3
```

#### Intensity Model Pre-training
```bash
python main.py --train --root './dataset' --noise_levels 1 2 4 8 16 --lr 1e-3 --epochs 100 --batch_size 16
```

### Full Options

```bash
python main.py --FLIM \
               --root './FLIM_dataset' \
               --types './BPAE/BPAE1/AG1' './BPAE/BPAE2/AG1' \
               --epochs 800 \
               --batch_size 1 \
               --lr 1e-3 \
               --alpha 3e-3 \
               --pretrained_model './model2_mix/best_dnflim.pth'
```

### Testing a Trained Model

```bash
python main.py --test \
               --root './FLIM_dataset' \
               --types './BPAE/BPAE1/AG1' \
               --batch_size 1 \
               --pretrained_model './checkpoints/best_dnflim.pth'
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lr` | float | 1e-3 | Learning rate |
| `--batch_size` | int | 64 | Batch size (use 1 for FLIM) |
| `--epochs` | int | 800 | Number of training epochs |
| `--root` | str | Required | Root directory for the dataset |
| `--noise_levels` | int | 1 | Noise levels in the dataset |
| `--types` | str | None | Specific types or categories of data to use |
| `--train` | flag | False | Trigger training mode |
| `--test` | flag | False | Trigger testing mode |
| `--pretrained_model` | str | './model2_mix/best_dnflim.pth' | Path to pretrained model |
| `--FLIM` | flag | False | Enable zero-shot training for FLIM |
| `--alpha` | float | 3e-3 | Weight parameter for loss function |

## Model Architecture

Our model is based on a dual-encoder U-Net architecture:

1. **Pre-trained Intensity Model**: A U-Net trained on general fluorescence microscopy data to denoise intensity images
2. **Dual-Encoder FLIM Network**: 
   - Separate encoder paths for processing g and s channels
   - Three specialized decoders:
     - Two channel-specific decoders for reconstructing g·I and s·I
     - A fusion decoder that produces I by leveraging features from both encoder pathways

## Loss Function

Our comprehensive joint loss function addresses multiple aspects of FLIM signal fidelity:

```
L = Lintensity + λ1*Lfidelity + λ2*Lstructure + λ3*LTV
```

Where:
- **Lintensity**: Aligns predicted intensity with pre-trained denoising output
- **Lfidelity**: Ensures consistency between inputs and reconstructions 
- **Lstructure**: Maintains structural coherence
- **LTV**: Reduces noise in uniform areas while preserving sharp edges

## Citation

If you use this code for your research, please cite the original paper:

```
@article{chen2023zero,
  title={Zero-Shot Denoising for Fluorescence Lifetime Imaging Microscopy with Intensity-Guided Learning},
  author={Chen, Hao and Najera, Julian and Geresu, Dagmawit and Datta, Meenal and Smith, Cody and Howard, Scott},
  journal={},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.