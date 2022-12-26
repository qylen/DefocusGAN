# DefocusGAN
Learnable Blur Kernel for Single-Image Defocus Deblurring in the Wild

This is the github space of DefocusGAN.

We released all code.

The link to the pre-trained model of DefocusGAN is: https://drive.google.com/file/d/1t46JVs9GXqVyNWONXYPrHx2XVIN42Ggz/view?usp=sharing

The link for unsupervised estimation of the defocus map is: https://drive.google.com/file/d/1qUtrV7dvzoBZTfS3cQqeLaSytF9PasN-/view?usp=sharing

After downloading the file, put the models in 

experiments\pretrain_model_for_DefocusGAN

experiments\pretrain_model_for_learnable_blur_kernel,respectively.

for testing, modify \option\test\Deblur_Dataset_Test.yaml for your dataset. 
Run test_final.py.

for training, modify \option\test\Defocus_GAN_Trained.yaml for your dataset.
First use blur_kernel_test.py to generate defocus map.
And then you can use train_final.py for training.

