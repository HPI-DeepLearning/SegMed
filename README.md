# SegMed
This repository contains the code to solve the [Multimodal Brain Tumor Segmentation Challenge 2017](http://braintumorsegmentation.org/) as part of the seminar "Practical Video Analysis" during the summer term 2017 at Hasso Plattner Institute for Software Systems Engineering. Most of the model is based on [pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow).

- The introduction tasks can be found inside the [brain_health_classification](https://github.com/HPI-DeepLearning/SegMed/tree/master/brain_health_classification) folder. We familiarized with the BraTS challenge and Deep Learning by first using simple Keras models to distinguish between healthy and unhealthy brains.
- BraTS related tasks
    - The classification algorithm can be found inside the [classification](https://github.com/HPI-DeepLearning/SegMed/tree/master/classification) folder. The task given by the supervisor demanded to distinguish between low- and high-grade gliomas. We used a vanilla CIFAR10 model which was able to distinguish between the mentioned tumor types with ease. 
    - The segmentation algorithm can be found inside the [pix2pix-ultimate](https://github.com/HPI-DeepLearning/SegMed/tree/master/pix2pix-ultimate) folder. The approach is based on [pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow) and uses a Conditional Generative Adversarial Network to distinguish between real segmented brain images and fakes ones.   
    - The survival rate prediction can be found inside the [survival_rate](https://github.com/HPI-DeepLearning/SegMed/tree/master/survival_rate) folder. The preferred approach uses a CNN to predict the survival rate of the patient. The [CNN model](https://github.com/HPI-DeepLearning/SegMed/blob/master/survival_rate/Model.ipynb) outperforms the [Support Vector Regression](https://github.com/HPI-DeepLearning/SegMed/blob/master/survival_rate/SVR_wg.ipynb) model in precision as well as variance of the error. 
# Demo
The demo code is located in the [server](https://github.com/HPI-DeepLearning/SegMed/tree/master/server) directory. It requires a trained segmentation model.

![](https://user-images.githubusercontent.com/6676439/29240031-278b6162-7f5d-11e7-8846-b0e8049191ff.gif)
