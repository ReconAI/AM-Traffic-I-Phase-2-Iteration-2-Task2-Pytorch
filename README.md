# AM-Traffic-I-Phase-2-Iteration-2-Task2-Pytorch
Prepare and develop end-to-end pipeline (for a road condition classification light-weight neural network).

As an input this model should take a video sequence from CCTV camera; As an output model should classify road condition (Dry, Moist, Wet, Wet & Salty, Frost, Snow, Ice, Probably moist/salty, Slushy).

-------------------------------------------------------------------------------------------------------------------------------

# Data
The data was collected during task4. As described in task4, the images were downloaded in AWS S3 bucket and the labels are included in the images’s names whose format is as follows:<br/>
 *'camera-id'\_r'roadConditionCategory'\_w'weatherConditionCategory'\_'measuredTime'*<br/>
 eg. "C1255201_r7_w0_2020-01-29_21-00-39"<br/>
 The weather conditions to classify are:<br/>
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

## Prediction on video
You can predict on video using *predict_video_tf.py* script:
```sh
python3 predict_video_torch.py --model ./models/pytorch/road_model.pt --weights  ./models/pytorch/weights_road.pth --input ./test_video.mp4 --labels ./road_labels.json --output ./output_road_torch/road_conditions_pie.avi --size 1
```
Where:
* **'model'**: the path of the training model.
* **weights**: the path to the parameters of the model.
* **'input'**: the path of your input vdeo (you have to mention the input video name).
* **'output'**: the path of the output video (you have to mention the output video name).
* **'labels'**: the path of labels json file.
* **'size'**: size of queue for averaging (128 by default). Set the size to 1 if you  don't want to perform any averaging.

