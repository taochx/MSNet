Code for the following Paper:

Tao, C., Meng, Y., Li, J., Yang, B., Hu, F., Li, Y.,Cui, C., Zhang, W., 2022. MSNet: multispectral semantic segmentation network for remote sensing images. GIScience & Remote Sensing 59, 1177–1198. https://doi.org/10.1080/15481603.2022.2101728

[[Paper](https://www.tandfonline.com/doi/full/10.1080/15481603.2022.2101728)]

# Abstract

In the research of automatic interpretation of remote sensing images, semantic segmentation based on deep convolutional neural networks has been rapidly developed and applied, and the feature segmentation accuracy and network model generalization ability have been gradually improved. However, most of the network designs are mainly oriented to the three visible RGB bands of remote sensing images, aiming to be able to directly borrow the mature natural image semantic segmentation networks and pre-trained models, but simultaneously causing the waste and loss of spectral information in the invisible light bands such as near-infrared (NIR) of remote sensing images. Combining the advantages of multispectral data in distinguishing typical features such as water and vegetation, we propose a novel deep neural network structure called the multispectral semantic segmentation network (MSNet) for semantic segmentation of multi-classified feature scenes. The multispectral remote sensing image bands are split into two groups, visible and invisible, and ResNet-50 is used for feature extraction in both coding stages, and cascaded upsampling is used to recover feature map resolution in the decoding stage, and the multi-scale image features and spectral features from the upsampling process are fused layer by layer using the feature pyramid structure to finally obtain semantic segmentation results. The training and validation results on two publicly available datasets show that MSNet has competitive performance. The code is available: [GitHub - taochx/MSNet](https://github.com/taochx/MSNet).

# The overall architecture

We present a new deep neural network structure called MSNet for multispectral remote sensing images for semantic segmentation of multiple feature scenes. [Figure 1](https://www.tandfonline.com/doi/full/10.1080/15481603.2022.2101728#f0001) shows the overall structure of MSNet, which mainly consists of two parts: 1) Band splitting and simultaneous feature extraction, split the multispectral remote sensing image bands into visible and invisible light, and both draw support from ResNet-50 for feature extraction in the coding stage; 2) Feature fusion decoding, adopt the cascaded upsampling method to recover the feature map resolution in the decoding stage, and uses the feature pyramid structure to layer-by-layer fusion of multi-scale image features and spectral features in the upsampling process to finally obtain semantic segmentation results.

![fig1](.\figs\fig1.png)

# Comparison

We use the GID and Potsdam datasets, respectively, to experimentally compare MSNet of both data models with some mature semantic segmentation network models DeepLabV3+, FPN, PSPNet, UNet, and RTFNet.

![fig5a](.\figs\fig5a.png)

![fig7a](.\figs\fig7a.png)

# Credits

If you find this work useful, please consider citing:

```bibtex
  @article{tao_msnet_2022,
    title={{MSNet}: multispectral semantic segmentation network for remote sensing images},
    author={Tao, Chongxin and Meng, Yizhuo and Li, Junjie and Yang, Beibei and Hu, Fengmin and Li, Yuanxi and Cui, Changlu and Zhang, Wen},
    journal={{GIScience} \& Remote Sensing},
    volume={59},
    pages={1177--1198},
    year={2022},
    publisher={Taylor \& Francis}
  }
```
