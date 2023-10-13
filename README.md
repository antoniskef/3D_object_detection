# Object Detection 

## Introduction
The goal of this thesis is to improve the performance of the publication 
"TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with 
Transformers" which relies on Transformers for object detection. The way to do this 
is by pre-training the network, as presented in the paper "3D Object Detection with a 
Self-supervised Lidar Scene Flow Backbone". More specifically, with self-supervised 
learning and pretext task the scene flow, the parameters of the TransFusion backbone 
network are initialized. To make this possible, modifications are made to the 
algorithms in both publications, while mixed precision is chosen for the computations 
due to limitations in the available graphics card memory. The Nuscenes dataset was 
used to train and test the networks. The above actions led to improved performance 
based on the mAP and NDS metrics. Additionally, a second attempt was made using a 
different learning rate scheduler and led to a slight further increase in the metrics.

## Pretraining 
The publication "3D Object Detection with a Self-supervised Lidar Scene Flow
Backbone" makes use of self-supervised learning to initialize the 
network parameters with pretext task the scene flow. Through this the network 
recognizes the motion characteristics of objects and exploits the 
different motion patterns they have resulting in better performance in the downstream task.
