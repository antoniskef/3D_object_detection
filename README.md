# Object detection in point cloud

## Introduction
The goal of this project is to improve the performance of the publication 
"TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with 
Transformers" which relies on Transformers for 3D object detection. The way to do this 
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
The scene flow method used from the paper is FlowNet3D.

## TransFusion 
TransFusion makes use of both LiDAR points and 
RGB images. It consists of two backbone networks, one for points and one for 
images, which are based on convolutional networks and generate their features. 
There are two layers of transformers.The first one makes use of 
features of the points and predicts some bounding boxes, which in 
second layer are refined by the contributing features from the images. 
The attention mechanism enables the model to determine which parts 
of the images are useful for improving the results.

At the beginning, LiDAR points are entered into a backbone convolutional network 
and specifically the paper states that this is the same as the one 
used in VoxelNet. In the supplementary material at the end of the paper 
there is an appendix that refers to the use of PointPillars as an alternative backbone network. 
PointPillars is chosen because it is also used from the paper of the pretraining. 
After the generation of the features of the points, follows the query 
initialization. Each object query provides a query position that defines the location 
of the object and a query feature that defines attributes of the box such as 
its size and its orientation. The final locations of the bounding boxes 
provided are relative to the query positions, as was done 
with anchors where no absolute values were predicted. It is emphasized that in 
earlier works the query positions were randomly generated or initialized as 
parameters of the network without any consideration of the input data, which 
created the need for additional decoder layers. The beginning was made in 
object detection applications in images and in particular in DETR where 
better initialization of queries is done. So initialization is applied 
influenced by the input data and based on a center heatmap
producing very good results with a single decoder layer.
The goal is for the initial object queries to be exactly or very close to the centers 
of the objects and so not many 
decoders be needed to improve the location.


