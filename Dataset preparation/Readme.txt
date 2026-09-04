************************************************************************************ Project name: FSRCNN ************************************************************************************ 

This repository is implementation of the ["Accelerating the Super-Resolution Convolutional Neural Network"](https://arxiv.org/abs/1608.00367).

===========================================================================================================================================================================

## Differences from the original

- Added the zero-padding
- Used the Adam instead of the SGD

===========================================================================================================================================================================

## Requirements

- PyTorch 1.13.1
- Numpy 1.21.6
- matplotlib 3.5.3
- h5py 3.8.0
- tqdm 4.67.1
- numpy 1.21.6
- pandas 1.3.5

===========================================================================================================================================================================

# datasets.py: In this file, the Image Dataset class is defined, which can be used to input images into the network and train the network. Of course, it should be noted that the LR and HR images must first be placed in the format of H5
files and the necessary codes for doing this are placed in the Dataset preparation folder.

# utils.py: In this file, the necessary functions for training the network are defined.

# models.py: In this file, the structure of the Fast Super Resolution Convolutional Neural Network (FSRCNN) model is defined in the form of a class, which is used to increase the quality of LR images.

===========================================================================================================================================================================

************************************************************************************ Main Code: FSRCNN.py *********************************************************************************** 

1) In line 45, the address of the H5 file related to the HR images should be written.

2) In line 46, the address of the H5 file related to the LR images should be written.

3) In line 51, the value of learning rate can be set.

4) In line 52, the value of batch-size can be set.

5) In line 53, the number of epochs can be set.

6) In line 79, the type of loss function can be set.

7) In line 81, the type of optimizer can be set.

===========================================================================================================================================================================
#FSRCNN_Test.py: 

Using this file, the network can be tested. For this purpose, the address of the file containing the best weights of the trained network must be written in line 144.
===========================================================================================================================================================================