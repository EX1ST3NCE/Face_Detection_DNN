# Frontal and Non-Frontal Face Detection using Deep Neural Networks (DNN)

A fast and accurate face detection with OpenCV using a pre-trained deep learning face detector model shipped with the library. Two files are required for using OpenCV's deep neural network module with Caffe models.

- The .prototxt file (define the model architecture i.e., the layers themselves)
- The .caffemodel file (which contains the weights for the actual layers)

You can find Caffe-based face detector prototxt files in the face_detector sub-directory of the [dnn samples](https://github.com/opencv/opencv/tree/master/samples/dnn).

### Requirements

To install the requirements, open cmd and type:

```
pip install -r requirements.txt
```

### Getting Started

- How to use
```
git clone https://github.com/prasadnitin05/Face_Detection_DNN.git
cd Face_Detection_DNN
```

### 

- To detect faces from photos 
 
 ```
 Open face_detect in IDLE3
 Press f5 or run module from the tab
 
 Then type the code:
 
detectDNN('./data/img.jpg')
detectDNN('./data/negative.jpg')
detectDNN('./data/tony.jpg')
detectDNN('./data/group.jpg')
 ```

- To detect faces from videos
 
 ```
 Open detect_video in IDLE3
 Press f5 or run module from the tab
 
 Then type the code:
 
detectVidDNN('./data/scientist.mp4')
detectVidDNN('./data/sample.mp4')
 ```

### Results 

![alt text](https://github.com/prasadnitin05/Face_Detection_DNN/results/fig1.png?raw=true)
