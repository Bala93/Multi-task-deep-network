# Multi-task-deep-network
### *Multi-task deep learning based approaches for semantic segmentation in medical images* 
> [Psi-Net:  Shape  and  boundary  aware  joint  multi-task  deep  network  for medical  image  segmentation](https://arxiv.org/abs/1902.04099) (EMBC 2019) 
![Psi-Net Architecture](appendix/PSI_Net.jpg)
> [Conv-MCD: A Plug-and-Play Multi-task Module for Medical Image Segmentation](https://arxiv.org/abs/1908.05311) (MICCAIW - MLMI 2019)
![Conv-MCD Architecture](appendix/Conv-MCD.png)

## Dependencies
#### Packages
* *PyTorch*
* *TensorboardX*
* *OpenCV*
* *numpy*
* *tqdm*
 
An exhaustive list of packages used could be found in the *requirements.txt* file. Install the same using the following command:

```bash
 conda create --name <env> --file requirements.txt
```

#### Preprocessing
Contour and Distance Maps are pre-computed. (Code to be added) 

#### Directory Structure
Train and Test folders should contain the following structure:

```
├── contour
    |-- 1.png
    |-- 2.png
    ...
├── dist_contour
    |--1.mat 
    |--2.mat
    ...
├── dist_mask
    |-- 1.mat
    |-- 2.mat
    ...
├── dist_signed
    |-- 1.mat
    |-- 2.mat
    ...
├── image
    |-- 1.jpg
    |-- 2.jpg
    ...
└── mask
    |-- 1.png
    |-- 2.png
    ...
```
[//]: # (## Sample Results)

## Citations
If you use the Conv-MCD or Psi-Net code in your research, please consider citing the respective paper:
```
@article{Murugesan2019PsiNetSA,
  title={Psi-Net: Shape and boundary aware joint multi-task deep network for medical image segmentation},
  author={Balamurali Murugesan and Kaushik Sarveswaran and Sharath M. Shankaranarayana and Keerthi Ram and Mohanasankar Sivaprakasam},
  journal={ArXiv},
  year={2019},
  volume={abs/1902.04099}
}
```
```
@misc{murugesan2019convmcd,
    title={Conv-MCD: A Plug-and-Play Multi-task Module for Medical Image Segmentation},
    author={Balamurali Murugesan and Kaushik Sarveswaran and Sharath M Shankaranarayana and Keerthi Ram and Jayaraj Joseph and Mohanasankar Sivaprakasam},
    year={2019},
    eprint={1908.05311},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
