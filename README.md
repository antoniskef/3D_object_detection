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

## Changes 
In the pretext task, which is the scene flow, the network learns to recognize the motion representations of objects and initializes 
the backbone network so that object detection, after changing the heads, yields better results. However, for this to happen, the two implementations 
must have exactly the same backbone networks. If the architecture differs, for example, if the number and type of layers or the number of weights
are not common, the initialization does not provide any advantage. The first thing to consider is that in the network of pretraining, each point is of four dimensions,
while in transfusion, it is five. Specifically, each point in both architectures has three coordinates, x, y, z, and one more related to differences in timestamps. 
On the other hand, the model for object detection has one extra dimension that defines intensity. So, in order for the input to be of the same dimension and
the first layer of the same size, the number of coordinates of the points must change for one of the implementations. It is chosen for both to have points with five coordinates, 
which means to add intensity to the points in self-supervised learning. For the next change, it is known that in PointPillars, the points additionally take the following values: xc, yc, zc, xp, yp, 
where the first three define the distance of the point from the center of all points belonging to the same pillar, while the next two show the distance from the center of the pillar itself. 
In the case of pretraining, the values indicating the distance from the center of the pillar are three, as the z coordinate is added. To continue to have a common input, the detector is adjusted accordingly.

As described both TransFusion and the pretraining architecture are based on PointPillars. In this architecture after creating the pillars and the three-dimensional tensors, a simplified version of PointNet
is applied to generate features. According to the original publications of both architectures, the linear layers used before applying batch normalization and ReLU consist of one layer. Since in some other 
papers two layers are used resulting in better performance, I add an extra layer in the sharing backbone network. 

Apart from those, since I had only one graphics card for this work, the models could not be trainedbecause of an "Out Of Memory" (OOM) error. To solve this problem, I used mixed precision. 
This means that for certain processes, half precision (FP16) is used instead of single precision (FP32). The problem that arises with this is that the gradients may become zero due to the reduced precision.
The solution is to apply loss scaling after the forward propagation and then perform the backward propagation with the resulting gradients also scaled. The scaling can be a fixed number by which the loss is multiplied,
or it can be dynamically determined. This means that the scaling factor starts with a large value and increases further if there is no overflow for a certain number of iterations. Conversely, if an overflow occurs,
the weights are not updated, and the scaling factor is reduced.

