# Frontal and Non-Frontal Face Detection using Deep Neural Networks (DNN)

Research Paper: http://www.riejournal.com/article_122236.html


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

- Non-Frontal Face Detection in images

![alt Detection of Non-Frontal Face from an image](https://github.com/prasadnitin05/Face_Detection_DNN/blob/master/results/fig1.png?raw=true)

![alt Face detection of people from different age groups](https://github.com/prasadnitin05/Face_Detection_DNN/blob/master/results/fig2.png?raw=true)

![alt Detection of Non-Frontal Face from an image](https://github.com/prasadnitin05/Face_Detection_DNN/blob/master/results/fig5.png?raw=true)

![alt Detection of Non-Frontal Face from an image](https://github.com/prasadnitin05/Face_Detection_DNN/blob/master/results/out_pos.png?raw=true)

![alt Detection of Non-Frontal Face from an video](https://github.com/prasadnitin05/Face_Detection_DNN/blob/master/results/fig3.png?raw=true)

