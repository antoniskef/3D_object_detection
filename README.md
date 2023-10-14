# 3D object detection

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
Mixing points with images almost always surpasses methods that use one of them,
as both contribute differently to the final result. Adding images to predictions
with points alone helps detect small objects or objects located far from the source.
This is due to the sparsity of point clouds, which makes it challenging to discern fine details.
However, object detection is a task that places high demands on the graphics card's memory, 
and the use of both points and images further increases these requirements. 
Due to the lack of a graphics card with sufficient memory for training this specific architecture, 
it is chosen not to use images but only points. This is feasible because the first transformer
is followed by a feedforward network (FFN) that predicts the bounding boxes and object classes.

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

## Two approaches 
After the initialisation of the parameters with the pretext task, the training of the object detection architecture starts. This happens twice with two different learning rate schedulers. 
The authors of Transfusion use 8 graphics cards with 2 samples per card, resulting in a batch size of 16. In my implementation, using mixed precision I achieve having 4 samples per card but only on one card. So, in my case,
the batch size is 4 times smaller, which necessitates a corresponding reduction in the learning rate. More specifically, the authors use a one-cycle scheduler [17], in which the learning rate is initialized with a value and 
increases linearly until it reaches a maximum. Then, a linear decrease starts, reaching very small values by the end of training, surpassing even the initial value. In the case of Transfusion, they choose an initial value of 0.0001, 
while the maximum is ten times greater, i.e., 0.001. Therefore, in the first approach I use one- cycle scheduler and I choose corresponding values of 0.0001/4 and 0.001/4. Additionally, they select the increase to last for 40% of 
the iterations, meaning that by the end of the 8th epoch (20 epochs in total), it will have reached the maximum. First approach in the image below.The accuracy stored in the logs is limited to the fourth decimal place, resulting 
in the specific discrete format when visualized. In reality, the learning rate accuracy is much higher, and in this analysis, if the complete accuracy was stored in the logs, there should be no discernible corners or discrete jumps.
![Screenshot 2023-10-13 191336](https://github.com/antoniskef/3D_object_detection/assets/93796754/16fd7ed6-7622-48cb-9580-cb6c25509b1f)

With the aim of achieving more accurate results, a different approach for adjusting the learning rate is tested, while keeping all other hyperparameters the same. Specifically, instead of the one-cycle scheduler, a combination of 
the cyclical learning rate and the one-cycle scheduler is used. The learning rate follows two cycles, with the second having a lower maximum and finishing, like the one-cycle scheduler, with very small, almost zero, values in the last epochs.
As previously, the initial value is 0.0001/4, and the maximum for the first cycle is 0.001/4. Once the learning rate reaches linearly the maximum and returns to the initial value, the second cycle begins with a value of 0.0001/4 and a maximum of 0.0006.
The first cycle lasts for 13 epochs, and the second one for 7, with a total sum equal to the first approach. Second approach in the image below
![Screenshot 2023-10-13 191417](https://github.com/antoniskef/3D_object_detection/assets/93796754/ca83a71c-caef-4b6d-a7a8-d8c36f398f90)

## Results 
The table below displays the metrics for both the first and second approaches in comparison to TransFusion. It's evident that there is an improvement compared to the publication for both approaches. It's also important to note that the model achieved
TransFusion's performance in the 14th epoch in the second approach, while in the first approach, this occurred after the 16th epoch.
![Screenshot 2023-10-13 191903](https://github.com/antoniskef/3D_object_detection/assets/93796754/78f93f07-87c5-4e00-85fd-da9d2eda8384)

NDS and mAP are the two metrics that are being used. Below there are two graphs showing the values of the metrics for each epoch for both approaches.

![Screenshot 2023-10-13 192058](https://github.com/antoniskef/3D_object_detection/assets/93796754/c7a927b1-291a-4375-9ec0-58eb9b26a394)
![Screenshot 2023-10-14 161338](https://github.com/antoniskef/3D_object_detection/assets/93796754/9354dd13-398f-4d75-8a66-f22174c81f53)

The two images below show the gradual reduction of losses during training. The total loss, is the weighted sum of three losses: one for classification, one for bounding boxes, and one for heatmap prediction, which contributes to the initialization of object queries and aims to position them as close as possible to the centers of objects. The weight of the heatmap loss is 1, while the other two have a weight of 0.25 each. The horizontal axis contains the iterations, which are approximately 32,000 per epoch. So, in total, there are a little over 600,000 iterations for the 20 epochs. The sudden reduction in loss at a certain point in the graph is due to the removal of a data augmentation technique from the dataset. In this techique a database is created that contains all the objects along with their labels. During training, random objects are selected from this database and added to the point cloud in use. This method, by merging point clouds from detached objects with the point cloud of the scene being processed at a given time, increases the number of objects per point cloud. To avoid unrealistic scenarios, collision checks are performed on the point clouds, and objects that overlap with others are removed.
![Screenshot 2023-10-13 192157](https://github.com/antoniskef/3D_object_detection/assets/93796754/8f035200-b814-4201-958a-e9cf6e1f8964)
![Screenshot 2023-10-14 161426](https://github.com/antoniskef/3D_object_detection/assets/93796754/cc20c20c-d131-4268-8de4-c8ada3114186)

Here are the graphs showing the mean Intersection over Union (IoU) for each iteration of the predicted bounding boxes matched with ground truth bounding boxes. As training progresses, higher IoU values are calculated, indicating that the predicted boxes are approaching in size, position, and orientation the manually created bounding boxes by the Nuscenes dataset creators. The sudden increase in IoU values at a certain point is also due to the removal of data augmentation techniques from the dataset, as mentioned earlier.
![Screenshot 2023-10-13 192224](https://github.com/antoniskef/3D_object_detection/assets/93796754/c2768741-c1d3-4efe-a805-461183a1a9ac)
![Screenshot 2023-10-14 161614](https://github.com/antoniskef/3D_object_detection/assets/93796754/85987c60-50c6-4bc7-a7b9-7fea007964a5)

## Visualize results
In this subsection, there are images of point clouds along with the model's predictions. The bounding boxes in blue are the ground truth, while the green ones represent the model's predictions.
![Screenshot 2023-10-14 161859](https://github.com/antoniskef/3D_object_detection/assets/93796754/eda256c5-8f81-451d-82bb-1bf3c493b9ce)
![Screenshot 2023-10-14 164353](https://github.com/antoniskef/3D_object_detection/assets/93796754/2152524c-f42e-4a5d-9e00-f2b8b7663bc0)




