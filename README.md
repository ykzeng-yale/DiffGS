<p align="center">
<h1 align="center">DiffGS: Functional Gaussian Splatting Diffusion <br>
(NeurIPS 2024)</h1>
<p align="center">
    <a href="https://junshengzhou.github.io/"><strong>Junsheng Zhou*</strong></a>
    ·
    <a href="https://weiqi-zhang.github.io/"><strong>Weiqi Zhang*</strong></a>
    ·
    <a href="https://yushen-liu.github.io/"><strong>Yu-Shen Liu</strong></a>
</p>
<p align="center"><strong>(* Equal Contribution)</strong></p>
<h3 align="center"><a href="https://arxiv.org/abs/2410.19657">Paper</a> | <a href="https://junshengzhou.github.io/DiffGS">Project Page</a></h3>
<div align="center"></div>
</p>
<p align="center">
    <img src="figs/mainfig.png" width="780" />
</p>


We release the code of the paper <a href="https://arxiv.org/abs/2410.19657">DiffGS: Functional Gaussian Splatting Diffusion</a> in this repository.

## Abstract

<p>
In this work, we propose DiffGS, a general Gaussian generator based on latent diffusion models. DiffGS is a powerful and efficient 3D generative model which is capable of generating Gaussian primitives at arbitrary numbers for high-fidelity rendering with rasterization. We explore DiffGS for various tasks, including unconditional generation, conditional generation from text, image, and partial 3DGS, as well as Point-to-Gaussian generation. We believe that DiffGS provides a new direction for flexibly modeling and generating Gaussian Splatting. 
          </p>

## Method

<p align="center">
  <img src="figs/method.png" width="780" />
</p>
<p style="margin-top: 30px">
<b>Overview of DiffGS.</b> <b>(a)</b> We disentangle the fitted 3DGS into three Gaussian Splatting Functions to model the Gaussian probability, colors and transforms. We then train a Gaussian VAE with a conditional latent diffusion model for generating these functions. <b>(b)</b> During generation, we first extract Gaussian geometry from the generated <b>GauPF</b>, followed by the <b>GauCF</b> and <b>GauTF</b> to obtain the Gaussian attributes.
</p>


## Generation Results

### Unconditional Generations

<img src="./figs/unconditional.gif" class="center">

### Visual Comparisons

<img src="./figs/shapenetchair.png" class="center">

## Visualization Results

### Text/Image-to-3D Generation

<img src="./figs/conditional.gif" class="center">

### Gaussian Completion

<img src="./figs/completion.gif" class="center">

### Point-to-Gaussian Generation

<img src="./figs/point2gaussian.gif" class="center">

## Install
We recommend creating an [anaconda](https://www.anaconda.com/) environment using our provided `environment.yml`:
```
conda env create -f environment.yml
conda activate diffgs
```
**Notice**：Since the code uses the original repository of Gaussian Splatting, please follow the environment setup instructions provided in the [official repository](https://github.com/graphdeco-inria/gaussian-splatting) to install the required dependencies.

If you would like to experiment with the PTv3 encoder from [Pointcept](https://github.com/Pointcept/Pointcept), install the additional dependency via

```
pip install pointcept
```
and set `"encoder": "ptv3"` in `GSModelSpecs` of your configuration file.

## Pretrained model
We first provide the pretrained models: `Gaussian VAE` and `Gaussian LDM` of the chair unconditional model. Please download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/13JyZtXV6ep26HnVIiFza0jn9F8VL5I1_?usp=sharing).


## Inference

To inference pretrained model of ShapeNet Chair, save the downloaded model checkpoint  to `config/stage1` and `config/stage2`. Additionally, you also need to adjust the checkpoint path in `config/genetate/specs.json`, then run the following command:

```
python test.py -e config/generate/
```
## Data preparation
1. We would like to thank [Stanford ShapeNet Renderer repository](https://github.com/panmari/stanford-shapenet-renderer) for their contribution,  we have made modifications to the code based on their open-source work. Please install `Blender` and run the following command: 

```bash
cd proecess_data
blender --background --python render_blender.py -- --output_folder {images_path} {mesh_path}
```

2. Next, perform point sampling on the mesh and modify the `shapene_folder` path in `sample_points.py`. The sampled points will be used as the initial positions for the Gaussians.
```
python sample_points.py
```
3. Run the Gaussian fitting script provided by us.

```
python train_gaussian.py -s <path to COLMAP or NeRF Synthetic dataset>
```
4. Run the conversion script `convert.py` provided by us to transform the Gaussians into data suitable for training, and perform sampling of the Gaussian probability field.

```
python convert_data.py
```

## Training

### 1. Train Gaussian modulations
```
python train.py -e config/stage1/ -b 4 -w 8    # -b for batch size, -w for workers, -r to resume training
```

### 2. Train the diffusion model using the modulations extracted from the first stage
```
# extract the modulations / latent vectors, which will be saved in a "modulations" folder in the config directory
# the folder needs to correspond to "Data_path" in the diffusion config files

python test.py -e config/stage1/ -r {num epoch}

python train.py -e config/stage2 -b 32 -w 8 
```

### Application: 

### 1. Trian Point to Gaussian

If you want to train point2gaussian, simply add `--point2gs` after the "Train Gaussian modulations" command.

```
python train.py -e config/stage1/ -b 4 -w 8 --point2gs
```

### 2. Train Conditional Generation

If you want to train a conditional generative model, please first prepare the condition for each Gaussian, set the `context_path` in `specs.json` to the correct path, and then run the following command.

```
python train.py -e config/stage2_conditional -b 32 -w 8 
```



## Citation

If you find our code or paper useful, please consider citing

    @inproceedings{diffgs,
        title={DiffGS: Functional Gaussian Splatting Diffusion},
        author={Zhou, Junsheng and Zhang, Weiqi and Liu, Yu-Shen},
        booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
        year={2024}
    }
