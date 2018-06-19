# Rotation Invariant Local Binary Convolution Neural Networks
By Xin Zhang, Li Liu, Yuxiang Xie, Jie Chen, Lingda Wu, Matti Pietikäinen

## Abstract
Although CNNs are unprecedentedly powerful to learn effective representations, they are still parameter expensive and limited by the lack of ability to handle with the orientation transformation of the input data. To alleviate this problem, we propose a deep architecture named Rotation Invariant Local Binary Convolution Neural Networks(RI-LBCNNs). RI-LBCNN is a deep convolution neural network consisting of Local Binary orientation Module(LBoM). A LBoM is composed of two parts, i.e., three layers steerable module (two layers for the first and one for the second part), which is a combination of Local Binary Convolution (LBConv) and Active Rotating Filters (ARFs). Through replacing the basic convolution layer in DCNNs with LBoM, RI-LBCNNs can be easy implemented and LBoM can be naturally inserted to popular models without any extra modification to the optimisation process. Meanwhile, The proposed RI-LBCNNs thus can be easily trained end to end. Extensive experiments show that the updating with the proposed LBoMs leads to significant reduction of learnable parameters and the reasonable performance on three benchmarks.

RI-LBCNN appeared in 2017 ICCV Workshop.

## Citation
If you use the code in your research, please cite:
```bibtex
@InProceedings{Zhang_2017_ICCV_Workshops,
author = {Zhang, Xin and Liu, Li and Xie, Yuxiang and Chen, Jie and Wu, Lingda and Pietikainen, Matti},
title = {Rotation Invariant Local Binary Convolution Neural Networks},
booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshop},
month = {Oct},
year = {2017}
}
```

The installation of RILBCNN is the same with ORN, please refer to: [installation](https://github.com/ZhouYanzhao/ORN/blob/master/README.md) for more information on installation.