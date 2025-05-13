conda deactivate
conda create -n colmapenv python=3.8
conda activate colmapenv
pip install opencv-python-headless
pip install tqdm
pip install natsort
pip install Pillow
conda install colmap -c conda-forge
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia