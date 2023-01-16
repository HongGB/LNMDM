# LNMDM Algorithm 

## Introduction
Aimed at assisting pathologists in the identification of metastatic tumor cells in lymph nodes removed after radical cystectomy, we developed a supervised learning model called the lymph node metastases diagnostic model (LNMDM). 6,375 images are used for training and internal validation of the model, and 1,616 images are used for external testing. Based on the convolutional neural network HRNet-w18 architecture, LNMDM is trained in whole-slide pathological images (WSIs) with pixel-level labels, in order to provide heatmaps and slide-level predictions of WSIs. Firstly, the masks of WSIs are obtained based on the pixel-level annotations of the training data. Secondly, patches are randomly cropped in the annotated regions of WSIs, and the prediction images which are obtained by sliding convolution of the convolution kernels are upsampled and input into the sigmoid function to form the prediction probability of each pixel. Finally, the annotated masks are combined with the binary cross-entropy loss function to obtain the slide-level predictions and heatmaps.


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

