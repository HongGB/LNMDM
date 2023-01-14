# LNMDM Algorithm 

## Introduction

## Requirements

- Python packages
  - Pytorch==1.9.0

## Train
To train the segmentation model, you add the image to ./img_path/ and the corresponding mask in ./mask_path/.

Then you can run the following command to train the model:

```bash
python train.py --cfg experiments/seg_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
```

## Test 
To test the segmentation model, you add the image to ./img_path/ and the corresponding mask in ./mask_path/. If the input image with size more than 2048, the code will slice the input image
into patch of size 2048, and input all the patches into the model.

You can run the following command to test the model:

```bash
python test.py --cfg experiments/seg_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
```

The results of predicted label and visualization can be found in './result/seg_label' and './result/seg_mask'.

