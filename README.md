# CNN ensemble using recursive Chernoff fusion
By Or Tslil, Nadav Leherer and Avishy Carmi

## Algorithm
This package demonstrates the application of recursive Chernoff fusion for classifiers ensemble.
For technical details refer to [1], which is currently under review.

## Dependencies
* keras
* imgaug
* sklearn

## Usage
* To train and test the ensemble of multiple CNNs (here we use VGG architecture) please run '''vgg_fusion_train.py'''.
* To ensemble multiple observaations (using augmentation of the same input) please run '''vgg_fusion_aug.py'''.
* The fusion scheme, presented in [1] are implemented in '''fusion_utils.py'''.

## Referance
[1] "Approaches to generalized Chernoff fusion with applications to distributed estimation".

