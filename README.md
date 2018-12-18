# End-to-End Incremental Learning

Code for our published paper http://arxiv.org/abs/1807.09536

# Abstract
Although deep learning approaches have stood out in recent years due to their state-of-the-art results, they continue to suffer from catastrophic forgetting, a dramatic decrease in overall performance when training with new classes added incrementally. This is due to current neural network architectures requiring the entire dataset, consisting of all the samples from the old as well as the new classes, to update the model -a requirement that becomes easily unsustainable as the number of classes grows. We address this issue with our approach to learn deep neural networks incrementally, using new data and only a small exemplar set corresponding to samples from the old classes. This is based on a loss composed of a distillation measure to retain the knowledge acquired from the old classes, and a cross-entropy loss to learn the new classes. Our incremental training is achieved while keeping the entire framework end-to-end, i.e., learning the data representation and the classifier jointly, unlike recent methods with no such guarantees. We evaluate our method extensively on the CIFAR-100 and ImageNet (ILSVRC 2012) image classification datasets, and show state-of-the-art performance.

# Code
### Prerequsites:
   - Matlab 2017b.
   - VlFeat (0.9.20). https://github.com/vlfeat/vlfeat
   - MatConvNet (1.0-beta25). https://github.com/vlfeat/matconvnet
   - ResNet-Matconvnet. https://github.com/zhanghang1989/ResNet-Matconvnet
   
   All these libraries are required to use the code. In parentheses appears the version we have used.
   
### Installation:
 1. Let's assume that you have placed this code in folder `<EndToEndIncrementalLearning>`.
 2. Let's assume that you have placed and installed ResNet-Matconvnet in folder `<ResNet-Matconvnet>`.
 3. Change path of ResNet-Matconvnet in startup.m
 4. Start Matlab.
 5. Now you can use the code.
 
### Usage:
1. Change required paths in build_imdbs.m and execute the code to generate the imdbs.
2. Train the first model with source code of ResNet-Matconvnet but using the first imdb for each step size and iteration. Don't forget to change the number of classes of the model and parse the labels to use eqlabs like in our incremental_training.m - lines 123-133. 
3. Change required paths and parameters in experiment_cifar.m and run the code to train the models.
4. Change required paths and parameters in test_cifar.m and run the code to test the models.

For ImageNet, you only have to change the number of classes and relative paths to the dataset.

Note that the parameters that must be changed are followed by the comment '% Edit me!'
 
