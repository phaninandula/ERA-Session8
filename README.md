# ERA-Session8

## Table of Contents
- [Problem Statement](#problem-statement)
- [Summary of Results](#summary-of-results)
- [Findings](#findings)

## Problem Statement
Assignment is:  
  
1. Change the dataset to CIFAR10  
2. Make this network:  
    1. C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP c10  
    2. Keep the parameter count less than 50000  
    3. Try and add one layer to another  
    4. Max Epochs is 20  
3. You are making 3 versions of the above code (in each case achieve above 70% accuracy):  
    1. Network with Group Normalization  
    2. Network with Layer Normalization  
    3. Network with Batch Normalization  
4. Share these details  
    1. Training accuracy for 3 models  
    2. Test accuracy for 3 models  
    3. Find 10 misclassified images for the BN model, and show them as a 5x2 image matrix in 3 separately annotated images.  
5. write an explanatory README file that explains:
    1. what is your code all about,
    2. your findings for normalization techniques,
    3. add all your graphs
    4. your collection-of-misclassified-images 
6. Upload your complete assignment on GitHub and share the link on LMS

## Summary of Results

Below table summarizes all the model results. 

| Model | Total params | Notebook Link | Train-Epoch when Acc>70% | Test-Epoch when Acc>70% | Overall Train Accuracy | Overall Test Accuracy |
|-------|--------------|---------------|-------------------|---------------|---------------------|-----------------|
| Group Normalization | xxxx | [GN Notebook](./ERA1_S8_CIFAR10_GN.ipynb) | - | - | --- % | --- % |
| Layer Normalization | 44,364 | [LN Notebook](./ERA1_S8_CIFAR10_LN.ipynb) | 5 | 5 | 83.11% | 77.04% |
| Batch Normalization | 44,364 | [BN Notebook](./ERA1_S8_CIFAR10_BN.ipynb) | 4 | 4 | 85.31% | 77.44% |

Total parameters dont change since the Normalization is just to keep the weights within 0 & 1 but doesnt reduce the parameters.
## Findings
1. Batch Normalization achieved the highest training and test accuracies.
2. Layer Normalization also performed on par with BatchNormalization, with slightly lower accuracies.
3. Group Normalization, however, needed the num of groups such that 'num_channels must be divisible by num_groups'.
