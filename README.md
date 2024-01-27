This is the PyTorch implementation of [SmRNet: Scalable Multiresolution Feature Extraction Network](https://ieeexplore.ieee.org/document/10389571). Serving as a versatile backbone, the network integrates the discrete wavelet transform (DWT) and its inverse (IDWT) to cater to various computer vision tasks, including detection, classification, and tracking.
![Upsampling_Downsampling](upsample_downsample_blocks.png)

![SmRNet](full_arch.png)


If you find this work useful, please cite:


```bash
@INPROCEEDINGS{10389571,
  author={Alaba, Simegnew Yihunie and Ball, John E.},
  booktitle={2023 International Conference on Electrical, Computer and Energy Technologies (ICECET)}, 
  title={SmRNet: Scalable Multiresolution Feature Extraction Network}, 
  year={2023},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ICECET58911.2023.10389571}}

```
## Getting Started
#### 1. Clone code

```bash
git clone https://github.com/Simeon340703/SmRNet.git
```
#### 2. Install Python packages
Install PyTorch and related.
#### 3. How to Run
#The default batch size is 128. model choices=['SmRNet_l', 'SmRNet_m', 'SmRNet_s']. dataset choices=['cifar10', 'cifar100'],
```bash
python main.py --batch-size 128 --lr 0.1 --model SmRNet_s --dataset cifar100 --epochs 100
```
#### 4 To DO
1. Add Object Detection
2.  Add Semantic Segmentation
