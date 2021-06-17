# GatedConvolution_pytorch
A modified reimplemented in pytorch of inpainting model in Free-Form Image Inpainting with Gated Convolution [http://jiahuiyu.com/deepfill2/]
This repo is transfered from the https://github.com/avalonstrel/GatedConvolution and https://github.com/JiahuiYu/generative_inpainting.

It is a model for image inpainting task. I implement the network structure and gated convolution in Free-Form Image Inpainting with Gated Convolution,
but a little difference about the original structure described in Free-Form Image Inpainting with Gated Convolution.

* In refine network, I do not employ the contextual attention but a self-attention layer instead.
* I add batch norm to each layer.

## Some results
BenchMark data and Mask data can be found in [Google Drive](https://drive.google.com/file/d/1xZxH6g7K3W7UKhd9DGW9EHa_AiIzBi33/view?usp=sharing)
![Result](result.png?raw=true "Title")
## How to test images by pre-trained model?
I provide a pre-trained [Baidu](https://pan.baidu.com/s/1bpHm9YoEV8isJz3S9bCziA), [Google](https://drive.google.com/file/d/1nMDb2REcfdLNd_HXpGaleuIGgr8QzLNZ/view?usp=sharing)  model on Places2 256x256 dataset, (but unfortunately only the coarse network can be loaded since I change the network structure after the pre-train process, in fact the coarse network also work).

Run `bash scripts/test_inpaint.sh`

You should provide a file containing file paths you want to test following the form of

test1.png

test2.png

...
...

Change the parameters in config/test_places2_sagan.yml
About the image

places2: 

    [

      'flist_file_for_train',
      'flist_file_for_test'
  
     ]
About the mask

val:

    [
    
      'mask_flist_file_for_train',
      
      'mask_flist_file_for_test'
      
    ]

The mask file should be a pkl file containing a numpy.array.

The MODEL_RESTORE should be set to the path of the pre-trained model.
After successfully running, you can find the results in result_logs/MODEL_RESTORE

## How to train your own model?
To train your own model with some other dataset you can

Run `bash scripts/run_inpaint_sa.sh`

By providing the

places2: 

    [

      'flist_file_for_train',
      'flist_file_for_test'
  
     ]

About the mask

val:

    [
    
      'mask_flist_file_for_train',
      
      'mask_flist_file_for_test'
      
    ]

And in training you can use random free-form mask or random rectangular mask. I use random free-form mask. If you want use random rectangular mask you need to change the process in train_sagan.py(line 163) and set MASK_TYPES: ['random_bbox'].

Some detials about the training parameters is easy to understand as shown in config file.

## Tensorboard

Run `tensorboard --logdir model_logs --port 6006` to view training progress.

## Some tips about mask generation?

We provide two random mask generation function.
* random free form masks
    
    The parameters about this function are
    
    RANDOM_FF_SETTING:
    
      img_shape: [256,256]
    
      mv: 5
    
      ma: 4.0
    
      ml: 40
    
      mbw: 10

    Following the meaning in http://jiahuiyu.com/deepfill2/.
* random rectangular masks

    RANDOM_BBOX_SHAPE: [32, 32]
    
    RANDOM_BBOX_MARGIN: [64, 64]
    
    means the shape of the random bbox and the margin between the boarder. (The number of rectangulars can be set in inpaint_dataset.py random_bbox_number=5)
## LICENSE
CC 4.0 Attribution-NonCommercial International

The software is for educational and academic research purposes only.

## Acknowledgments
My project acknowledge the official code DeepFillv1 and SNGAN. Especially, thanks for the authors of this amazing algorithm.
