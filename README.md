# E-commercial-dataset
Dataset of electronic commercial image used for saliency.

The dataset can be downloaded in https://www.dropbox.com/s/xsui782oy3kvjsm/E-commercial%20dataset.zip?dl=0. 

## IMAGE

Original images are saved in this path as *.jpg

## FIXATION

Fixation maps are saved as *\_fixPts.jpg, while saliency maps are saved as *\_.fixMap.jpg.

## TEXT REGION

The text detection results are stored in csv file, with the affinity score and region score.

# SSwin-transformer Model added in Repo
[[![](https://img.shields.io/badge/blog-@champyin-red.svg)]](https://img.shields.io/badge/pytorch-1.8.0-brightgreen)
## To-do list
1. -[ ] Adding environment setting (you can use environment same as swin-transformer as temporary alternatives)
2. -[ ] Refine the code into efficient way

- Clone this repo:

```bash
git clone https://github.com/leafy-lee/E-commercial-dataset.git
cd e-commercial
```

- Create a conda virtual environment and activate it:

```bash
conda create -n ecom python=3.7 -y
conda activate ecom
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.8.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
```

- Install `timm==0.4.12`:

```bash
pip install timm==0.4.12
```

- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
```
### Evaluation

To evaluate a pre-trained `Swin Transformer` on ImageNet val, run:

```bash
python main.py --eval --cfg <config-file> --resume <checkpoint> --data-path <imagenet-path> 
