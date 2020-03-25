# AM-Traffic-I-Phase-2-Iteration-2-Task2-Pytorch
Prepare and develop end-to-end pipeline (for a road condition classification light-weight neural network).

As an input this model should take a video sequence from CCTV camera; As an output model should classify road condition (Dry, Moist, Wet, Wet & Salty, Frost, Snow, Ice, Probably moist/salty, Slushy).

-------------------------------------------------------------------------------------------------------------------------------
# Requirements
To be able to install Fastai on Valohai, part of the requirements could be found in *requirements.txt* and you can find the rest of them in the yaml file instructions.

# Data
The data was collected during task4. As described in task4, the images were downloaded in AWS S3 bucket and the labels are included in the images’s names whose format is as follows:<br/>
 *'camera-id'\_r'roadConditionCategory'\_w'weatherConditionCategory'\_'measuredTime'*<br/>
 eg. "C1255201_r7_w0_2020-01-29_21-00-39"<br/>
 The road conditions to classify are:<br/>
 * Dry (0)
 * Moist (1)
 * Wet (2)
 * Wet & Salty (3)
 * Frost (4)
 * Snow (5)
 * Ice (6)
 * Probably moist/salty (7)
 * Slushy (8)
 
Unfortunately the labels are not accurate and have many mistakes and that’s due to different reasons such as the quality of the image, the distance between camera and weather station, sensors errors… so manually checking the labels was necessary. 
# Training the model (train.py)
The training was made using 1xGPU NVIDIA Tesla K80 (on Microsoft Azure NC6).

Once the data was ready, a model was built with Fastai (Pytorch). I used the resnet34 architecture pretrained on imagenet dataset. The choice of the architecture was based on the fact that the model must be light weighted in order to be run in realtime on a Jetson Nano device. Therefore, I had to make a compromise between accuracy and lesser number of parameters. Since depth-wise convolutions are known of low accuracy, I didn’t opt for mobilenet. So I found that resnet34 is the best candidate.<br/>  
The data was augmented using Fastai library.<br/>

The best accuracy I got is **0.79** (execution **#3** in Valohai). 
This model was obtained with one cycle policy, batch size of *64* sample, image with *(224x224)* size and no layer fine tuned.
# Testing the model (predict.py)
To test the performance of the model we run the model on images not included in training and validation datasets.
## Prediction on images
You can predict on images using *predict_images_tf.py* script:
```sh
python3 predict_images_torch.py --model ./models/pytorch/road_model.pt --weights  ./models/pytorch/weights_road.pth --input ./input --output ./output_road_torch --labels ./road_labels.json
```
Where:
* **'model'**: the path of the training model architecture.
* **weights**: the path to the parameters of the model.
* **'input'**: the path of your input images.
* **'output'**: the path of the output images.
* **'labels'**: the path of labels json file.
### Results
The predictions are displayed on images as follows:

<p align="center">
  <img src="figures/C0150409_r6_w0_2020-02-06_14-26-26.jpg">
</p>

<p align="center">
  <img src="figures/C0150802_r1_w0_2020-02-03_08-25-46.jpg">
</p>

Predictions metrics calculated on the test dataset:

<p align="center">
  <img src="figures/class_report_torch.png">
</p>

## Prediction on video
You can predict on video using *predict_video_tf.py* script:
```sh
python3 predict_video_torch.py --model ./models/pytorch/road_model.pt --weights  ./models/pytorch/weights_road.pth --input ./test_video.mp4 --labels ./road_labels.json --output ./output_torch/road_conditions.avi --size 1
```
Where:
* **'model'**: the path of the training model.
* **weights**: the path to the parameters of the model.
* **'input'**: the path of your input vdeo (you have to mention the input video name).
* **'output'**: the path of the output video (you have to mention the output video name).
* **'labels'**: the path of labels json file.
* **'size'**: size of queue for averaging (128 by default). Set the size to 1 if you  don't want to perform any averaging.
# Conversion to TensorRT
Conversion of the built pytorch model to ONNX model to TensorRT model.
## Requirement
* tensorflow-gp~=1.15.0
* Keras~=2.2.5
* argparse~=1.4.0
* onnx2keras~=0.018
* onnx~=1.6.0
* torch~=1.4.0
## Installation
```sh
pip3 install onnx
pip3 install onnx2keras
git clone https://github.com/onnx/onnx-tensorflow.git
cd onnx-tensorflow
sudo python3 setup.py install
```
## Conversion from pytorch model to keras model
Use the script *convert_pt_to_keras.py* as follows:
```sh
python3 convert_pt_to_keras.py --model ./models/torch/road_model.pt --weights ./models/torch/weights_road.pth --keras_path ./models/torch_trt/
```
Where:
* **model**: path to trained serialized model.
* **weights**: path to the parameters of the model.
* **keras_path**: path where to save our converted keras model.

Once the script in executed, onnx model and keras model are saved in *keras_path*.
## conversion from keras model to tensorRT model
Use the script *convert_keras_to_trt.py* as follows:
```sh
python3 convert_keras_to_trt.py --trt_path ./models/keras_trt --model ./models/tensorflow/road_model.h5 --output_node  test_output/BiasAdd
```
Where:
* **trt_path**: path where to save the converted models.
* **model**: path to trained serialized keras model.
* **output_node**:  name of the output node (*test_output/BiasAdd* in our case).

After running this script successfully, in trt_path you will have:
*checkpoints, tf_model.meta, frozen_model.pb and tensorrt_model.pb.* 

