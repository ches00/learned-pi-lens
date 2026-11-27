# Learned 𝜋 Metalens for Wide-Field-of-View Near-Infrared Imaging

### [Project Page]((https://eschoi.com/learned-pi-lens/) | [Paper](https://dl.acm.org/doi/abs/10.1145/3842569)
[Eunsue Choi*](https://eschoi.com), [Gyeongtae Kim*](https://scholar.google.co.kr/citations?user=0rZekfsAAAAJ), [Eunji Lee*](https://scholar.google.com/citations?user=yvNUK8AAAAAJ), [Seungwoo Yoon](https://sites.google.com/view/seungwoo-yoon/), [Youngmo Jeong](https://scholar.google.com/citations?user=LBxbo9kAAAAJ), [Woohyeok Kim](https://woo525.github.io/), [Minseok Choi](https://scholar.google.com/citations?user=77qfYpwAAAAJ), [Sunghyun Cho](https://scho.postech.ac.kr/), [Junsuk Rho+](https://sites.google.com/site/junsukrho/), [Seung-Hwan Baek+](https://www.shbaek.com/)

This repository contains the implementation and supplementary files for the paper:

**Learned 𝜋 Metalens for Wide-Field-of-View Near-Infrared Imaging**

Eunsue Choi\*, Gyeongtae Kim\*, Eunji Lee\*, Seungwoo Yoon, Youngmo Jeong, Woohyeok Kim, Minseok Choi, Sunghyun Cho, Junsuk Rho+, Seung-Hwan Baek+

\* Equal contribution, + Corresponding authors

***ACM Transactions on Graphics (TOG), SIGGRAPH Asia 2026***

This code implements, in PyTorch, an angle-aware point spread function (PSF) simulation for wide-FoV lens optimization, an angle-aware metalens representation, and image simulation. These components are jointly optimized end-to-end using standard deep learning optimizers.


## Requirements

This code was trained and tested on Linux with an NVIDIA A6000 GPU (48 GB VRAM).

We recommend using a Conda environment. Create it from the `environment.yaml` file provided in this repository:

```bash
conda env create --file environment.yaml
```


## Dataset Setup
* For training, place ground-truth (GT) images in the `data/train` and `data/valid` folders.
* For testing, we provide our optimized metalens phase map in the `data` folder. To test your own learned phase map, simply replace this file.
* We used the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset for training and for the evaluations reported in the paper.

## Train
```bash
python train.py
```
* Training configurations can be customized in `Argparse.py`.

## Test
```bash
python test.py
```


## Hardware Requirements
* Training requires an NVIDIA GPU with at least 48 GB of memory.
* We used an NVIDIA A6000 for training.


## Acknowledgement

Parts of our code are based on [pado](https://github.com/shwbaek/pado).
We also thank the authors of the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset, which we used for training and evaluation.

## Citation

If you find our work useful in your research, please cite:

```bibtex
@article{choi2026learned,
  author    = {Choi, Eunsue and Kim, Gyeongtae and Lee, Eunji
               and Yoon, Seungwoo and Jeong, Youngmo
               and Kim, Woohyeok and Choi, Minseok
               and Cho, Sunghyun and Rho, Junsuk
               and Baek, Seung-Hwan},
  title     = {Learned pi Metalens for Wide-Field-of-View
               Near-Infrared Imaging},
  journal   = {ACM Transactions on Graphics (TOG)},
  year      = {2026},
  volume    = {45},
  number    = {6},
  articleno = {230},
  numpages  = {15},
  doi       = {10.1145/3842569},
  url       = {https://doi.org/10.1145/3842569}
}
```
