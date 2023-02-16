# MCNN-pytorch
This is an simple and clean implemention of CVPR 2016 paper ["Single-Image Crowd Counting via Multi-Column Convolutional Neural Network."](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)  
# Installation
&emsp;1. Install pytorch 1.0.0 later and python 3.6 later  
&emsp;2. Install visdom  
```pip
pip install visdom
```
&emsp;3. Clone this repository  

We'll call the directory that you cloned MCNN-pytorch as ROOT.
# Data Setup
&emsp;1. Download ShanghaiTech Dataset from kaggle.
  
&emsp;2. Put ShanghaiTech Dataset in ROOT and use "data_preparation/k_nearest_gaussian_kernel.py" to generate ground truth density-map. (Mind that you need modify the root_path in the main function of "data_preparation/k_nearest_gaussian_kernel.py")  
# Training
&emsp;1. Modify the root path in "train.py" according to your dataset position.   
&emsp;2. In command line:
```
python -m visdom.server
```  
&emsp;3. Run train.py
# Testing
&emsp;1. Modify the root path in "test.py" according to your dataset position.  
&emsp;2. Run test.py for calculate MAE of test images or just show an estimated density-map.  
