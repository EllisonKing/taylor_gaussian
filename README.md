[comment]: <> (# Learnable Infinite Taylor Gaussian for Dynamic View Rendering)

<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"> Learnable Infinite Taylor Gaussian for Dynamic View Rendering
  </h1>
  <p align="center">
    <p align="center">
    <strong>Bingbing Hu </strong> 路
    <strong>Yanyan Li</strong> 路
    <strong>Rui Xie </strong> 路
    <strong>Bo Xu</strong> 路
    <strong>Haoye Dong</strong> 路
    <strong>Junfeng Yao</strong>  路
    <strong>Gim Hee Lee </strong>
    </p>
  </p>




[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://arxiv.org/pdf/2412.04282">Paper</a> | <a href="https://ellisonking.github.io/TaylorGaussian/">Project Page</a></h3>
  <div align="center"></div>



<p align="center">
  <a href="">
    <img src="./imgs/Framework.jpg" alt="teaser" width="100%">
  </a>
</p>


## 馃摉 Motivation and Abstract


Instead of relying on **time-conditioned polynomial functions** to approximate Gaussian trajectories and directions, this solution investigates a more **accurate Gaussian evolution** model for dynamic scenarios.


<p >
Capturing the temporal evolution of Gaussian properties such as position, rotation, and scale is a challenging task due to the vast number of time-varying parameters and the limited photometric data available, which generally results in convergence issues, making it difficult to find an optimal solution. While feeding all inputs into an end-to-end neural network can effectively model complex temporal dynamics, this approach lacks explicit supervision and struggles to generate high-quality transformation fields. On the other hand, using time-conditioned polynomial functions to model Gaussian trajectories and orientations provides a more explicit and interpretable solution, but requires significant handcrafted effort and lacks generalizability across diverse scenes. To overcome these limitations, this paper introduces a novel approach based on a learnable infinite Taylor Formula to model the temporal evolution of Gaussians. This method offers both the flexibility of an implicit network-based approach and the interpretability of explicit polynomial functions, allowing for more robust and generalizable modeling of Gaussian dynamics across various dynamic scenes.
</p>

## 馃搵 TODO List
- [x] *Repo* - Create repo for [TaylorGaussian](https://ellisonking.github.io/TaylorGaussian).
- [x] *Clean* - Clean the system
- [x] *Test* - Test the system on different Ubuntu servershub
- [x] *Code* - Release code to the community


# 馃敡 Installation

### 1. Clone the TaylorGaussian Repo.
```
conda create -n taylorgaussian python=3.10
conda activate taylorgaussian
```

### 2. Environment setup. 
```
pip install -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install setuptools==69.5.1  
pip install numpy==1.23.5
pip install Cmake

pip install git+https://github.com/Po-Hsun-Su/pytorch-ssim.git
pip install git+https://github.com/facebookresearch/pytorch3d.git
```

Additionally, please install *diff-gaussian-rasterization* and *simple-knn* following comments: 
```
cd submodules
git clone https://github.com/ashawkey/diff-gaussian-rasterization.git
cd ./diff-gaussian-rasterization/
pip install -e .
cd ..
git clone https://github.com/camenduru/simple-knn.git
cd ./simple-knn/
pip install -e .
```

# 馃捑 Datasets

### 1. N3DV
Download the dataset from [here](https://github.com/facebookresearch/Neural_3D_Video.git).
```
python script/n3d_process.py --videopath ./data/Neural3D/cook_spinach
```

For each sequence, the structure is organized as follows:
cook_spinach
鈹溾攢鈹€ cam00
鈹溾攢鈹€ cam01
鈹溾攢鈹€ cam02
鈹溾攢鈹€ cam<....>
鈹溾攢鈹€ colmap_0
鈹溾攢鈹€ colmap_1
鈹溾攢鈹€ colmap_2
鈹溾攢鈹€ colmap_3
鈹溾攢鈹€ colmap_4
鈹溾攢鈹€ colmap_<....>
鈹斺攢鈹€ poses_bounds.npy


### 2. Technicolor
The dataset is provided by锛歔Dataset and Pipeline for Multi-View Light-Field Video](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w27/papers/Sabater_Dataset_and_Pipeline_CVPR_2017_paper.pdf)锛?```
python script/technicolor_process.py --videopath ./data/Technicolor/Birthday

```
For each sequence, the structure is organized as follows:
Birthday
鈹溾攢鈹€ Birthday_undist_00001_00.png
鈹溾攢鈹€ Birthday_undist_00001_01.png
鈹溾攢鈹€ Birthday_undist_00001_02.png
鈹溾攢鈹€ Birthday_undist_<....>
鈹溾攢鈹€ cameras_parameters.txt
鈹溾攢鈹€ colmap_0
鈹溾攢鈹€ colmap_1
鈹溾攢鈹€ colmap_<....>


# 馃搳  Results

### 1. PSNR, SSIM, and LPIPS Results
<table>
<caption><strong>Table I.</strong> Comparison of rendering on the NV3D dataset. 鈫?indicates lower is better, 鈫?indicates higher is better. The best score is in <b><i>bold-italic</i></b>, the second best is in <i>italic</i>.</caption>
    <tr>
        <td colspan="1"><div align="center">Squence</div></td> 
        <td colspan="3"><div align="center">Cook Spinach</div></td> 
        <td colspan="3"><div align="center">Sear steak</div></td> 
        <td colspan="3"><div align="center">Flame Steak</div></td> 
        <td colspan="3"><div align="center">Cut Roast Beef</div></td> 
        <td colspan="3"><div align="center">Flame Salmon</div></td>
         <td colspan="3"><div align="center">Coffee Martini</div></td>
    </tr>
    <tr>
        <td>Metric </td>
        <td> PSNR鈫?/td> 
        <td> SSIM鈫?/td>
        <td>LPIPS鈫?/td>
        <td> PSNR鈫?/td> 
        <td> SSIM鈫?/td>
        <td>LPIPS鈫?/td>
        <td> PSNR鈫?/td> 
        <td> SSIM鈫?/td>
        <td>LPIPS鈫?/td>
        <td> PSNR鈫?/td> 
        <td> SSIM鈫?/td>
        <td>LPIPS鈫?/td>
        <td> PSNR鈫?/td> 
        <td> SSIM鈫?/td>
        <td>LPIPS鈫?/td>
        <td> PSNR鈫?/td> 
        <td> SSIM鈫?/td>
        <td>LPIPS鈫?/td>
    </tr>
    <tr>
        <td>D3DGS</td>
        <td>20.53</td>
        <td>0.881</td>
        <td>0.153</td>
        <td>25.02</td>
        <td>0.944</td>
        <td>0.072</td>
        <td>23.02</td>
        <td>0.919</td>
        <td>0.113</td>
        <td>22.35</td>
        <td>0.907</td>
        <td>0.125</td>
        <td>23.03</td>
        <td>0.863</td>
        <td>0.153</td>
        <td>23.42</td>
        <td>0.865</td>
        <td>0.138</td>
    </tr>
    <tr>
      <td>4DGS</td>
      <td>28.12</td>
      <td>0.940</td>
      <td>0.038</td>
      <td>29.07</td>
      <td>0.957</td>
      <td>0.028</td>
      <td>25.04</td>
      <td>0.918</td>
      <td>0.079</td>
      <td>29.71</td>
      <td>0.944</td>
      <td>0.033</td>
      <td>27.90</td>
      <td>0.916</td>
      <td>0.063</td>
      <td>28.18</td>
      <td>0.924</td>
      <td>0.049</td>
    </tr>
    <tr>
      <td>FSGS</td>
      <td>29.60</td>
      <td>0.919</td>
      <td>0.115</td>
      <td>29.87</td>
      <td>0.945</td>
      <td>0.116</td>
      <td>29.42</td>
      <td>0.943</td>
      <td>0.113</td>
      <td>28.46</td>
      <td>0.913</td>
      <td>0.122</td>
      <td>28.06</td>
      <td>0.926</td>
      <td>0.102</td>
      <td>26.57</td>
      <td>0.907</td>
      <td>0.169</td>
    </tr>
    <tr>
      <td>SCGS</td>
      <td>17.20</td>
      <td>0.734</td>
      <td>0.232</td>
      <td>28.77</td>
      <td>0.951</td>
      <td>0.056</td>
      <td>23.49</td>
      <td>0.902</td>
      <td>0.104</td>
      <td>6.29</td>
      <td>0.007</td>
      <td>0.683</td>
      <td>5.63</td>
      <td>0.002</td>
      <td>0.681</td>
      <td>22.81</td>
      <td>0.883</td>
      <td>0.142</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>32.59</td>
      <td>0.966</td>
      <td>0.054</td>
      <td>33.12</td>
      <td>0.973</td>
      <td>0.049</td>
      <td>33.34</td>
      <td>0.971</td>
      <td>0.052</td>
      <td>33.06</td>
      <td>0.969</td>
      <td>0.055</td>
      <td>28.93</td>
      <td>0.951</td>
      <td>0.089</td>
      <td>27.51</td>
      <td>0.945</td>
      <td>0.088</td>
    </tr>
</table>


<table>
<caption><strong>Table II.</strong> Methods comparison on the Technicolor dataset. Best results are in <b>bold</b>.</caption>
  <tr>
    <td rowspan="2"><div align="center">Method</div></td>
    <td colspan="3"><div align="center">Birthday</div></td>
    <td colspan="3"><div align="center">Painter</div></td>
    <td colspan="3"><div align="center">Train</div></td>
    <td colspan="3"><div align="center">Fatma</div></td>
  </tr>
  <tr>
    <td><div align="center">PSNR 鈫?/div></td>
    <td><div align="center">SSIM 鈫?/div></td>
    <td><div align="center">LPIPS 鈫?/div></td>
    <td><div align="center">PSNR 鈫?/div></td>
    <td><div align="center">SSIM 鈫?/div></td>
    <td><div align="center">LPIPS 鈫?/div></td>
    <td><div align="center">PSNR 鈫?/div></td>
    <td><div align="center">SSIM 鈫?/div></td>
    <td><div align="center">LPIPS 鈫?/div></td>
    <td><div align="center">PSNR 鈫?/div></td>
    <td><div align="center">SSIM 鈫?/div></td>
    <td><div align="center">LPIPS 鈫?/div></td>
  </tr>
  <tr>
    <td>D3DGS</td>
    <td>33.81</td><td>0.965</td><td>0.014</td>
    <td>37.38</td><td>0.957</td><td>0.036</td>
    <td>-</td><td>-</td><td>-</td>
    <td>38.40</td><td>0.911</td><td>0.093</td>
  </tr>
  <tr>
    <td>STG</td>
    <td>33.87</td><td>0.951</td><td>0.038</td>
    <td>37.30</td><td>0.928</td><td>0.095</td>
    <td>33.36</td><td>0.948</td><td>0.036</td>
    <td>37.28</td><td>0.906</td><td>0.155</td>
  </tr>
  <tr>
    <td>FSGS</td>
    <td>26.26</td><td>0.920</td><td>0.068</td>
    <td>34.36</td><td>0.958</td><td>0.063</td>
    <td>30.39</td><td>0.965</td><td>0.032</td>
    <td>27.62</td><td>0.825</td><td>0.276</td>
  </tr>
  <tr>
    <td>4DGS</td>
    <td>21.94</td><td>0.902</td><td>0.071</td>
    <td>28.61</td><td>0.940</td><td>0.058</td>
    <td>22.36</td><td>0.878</td><td>0.124</td>
    <td>23.42</td><td>0.763</td><td>0.236</td>
  </tr>
  <tr>
    <td><b>Ours</b></td>
    <td><b>34.72</b></td><td><b>0.988</b></td><td><b>0.013</b></td>
    <td><b>38.37</b></td><td><b>0.985</b></td><td><b>0.022</b></td>
    <td><b>35.30</b></td><td><b>0.990</b></td><td><b>0.008</b></td>
    <td><b>38.91</b></td><td><b>0.945</b></td><td><b>0.071</b></td>
  </tr>
</table>


### 2. Erratum

In Table 1 ( tested on the N3DV Dataset) of the previous arxiv version, the experimentation results of our method have problems since the test dataset was polluted. Specifically, during the computation process, we discovered that a key parameter was set incorrectly (specifically, eval was mistakenly set to False), leading to these problem results in PSNR, SSIM, and LPIPS metrics. If you use the same code in your testing, please make sure the correct setting. For details, please refer to ...



# 猸曪笍 Acknowledgement
This work incorporates many open-source codes. We extend our gratitude to the authors of the software.
- [SCGS](https://github.com/CVMI-Lab/SC-GS)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)
- [SpacetimeGaussians](https://github.com/oppo-us-research/SpacetimeGaussians)


# 鉁夛笍 License and Citation
This project is released under the Gaussian-Splatting License.


If you find this code/work useful for your own research, please consider citing:

```
@article{hu2024learnable,
  title={Learnable Infinite Taylor Gaussian for Dynamic View Rendering},
  author={Hu, Bingbing and Li, Yanyan and Xie, Rui and Xu, Bo and Dong, Haoye and Yao, Junfeng and Lee, Gim Hee},
  journal={arXiv preprint arXiv:2412.04282},
  year={2024}
}

```