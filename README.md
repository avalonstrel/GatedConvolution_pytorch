# GatedConvolution_pytorch
A modified reimplemented in pytorch of inpainting model in Free-Form Image Inpainting with Gated Convolution [http://jiahuiyu.com/deepfill2/] 
This repo is transfered from the https://github.com/avalonstrel/GatedConvolution and https://github.com/JiahuiYu/neuralgym. 
It is a model for image inpainting task. I implement the network structure and gated convolution in Free-Form Image Inpainting with Gated Convolution,
but a little difference about the original structure described in Free-Form Image Inpainting with Gated Convolution. 
* In refine network, I do not employ the contextual attention but a self-attention layer instead.
* I add batch norm to each layer.


### How to test images by pre-trained model?
I provide a pre-trained model on Places2 256x256 dataset, (but unfortunately only the coarse network can be loaded since I
change the network structure after the pre-train process, in fact the coarse network also work). 


### How to train your own model?


### Some tips about mask generation?
