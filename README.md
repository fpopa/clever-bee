# clever-bee
Drone object detection and tracking.

### Training & detection
I've created my own dataset for a remote controlled helicopter.

Used image augmentation in order to enhance the dataset.

Training dataset consists of frames extracted from videos recorded both inside and outside

##### results:

![2.jpg](https://github.com/fpopa/clever-bee/blob/master/results/images/2.jpg)
![3.jpg](https://github.com/fpopa/clever-bee/blob/master/results/images/3.jpg)
![4.jpg](https://github.com/fpopa/clever-bee/blob/master/results/images/4.jpg)
![1.jpg](https://github.com/fpopa/clever-bee/blob/master/results/images/1.jpg)


The most surprising result can be seen in the last picture, in which the network detects the helicopter shadow.

### Tracking

Movement commands are issued based on how far off the tracked object is from the center of the image. The farther away, the more aggressive the movement will be.

For tracking I've used a pretrained network (VOC) as controlling the helicopter and making sure the drone wouldn't crash at the same time would have been a challenge.

[![Autonomous drone, object detection & tracking](http://img.youtube.com/vi/RtYwD74qlWQ/0.jpg)](http://www.youtube.com/watch?v=RtYwD74qlWQ)

Spaghetti code, feel free to ask anything.

##### Work based on:

https://github.com/weiliu89/caffe/tree/ssd

https://github.com/chuanqi305/MobileNet-SSD

https://github.com/srianant/kalman_filter_multi_object_tracking

https://github.com/brean/python-ardrone

##### Used libraries:

https://github.com/duangenquan/YoloV2NCS

https://github.com/aleju/imgaug
