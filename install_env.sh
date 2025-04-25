#!/bin/bash

conda create -n taylorgaussian python=3.10
conda activate taylorgaussian

pip install -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install setuptools==69.5.1  
pip install numpy==1.23.5
pip install Cmake

pip install git+https://github.com/Po-Hsun-Su/pytorch-ssim.git
pip install git+https://github.com/facebookresearch/pytorch3d.git


cd submodules
git clone https://github.com/ashawkey/diff-gaussian-rasterization.git
cd ./diff-gaussian-rasterization/
pip install -e .
cd ..
git clone https://github.com/camenduru/simple-knn.git
cd ./simple-knn/
pip install -e .