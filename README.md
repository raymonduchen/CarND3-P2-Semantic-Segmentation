# CarND3-P2 Semantic Segmentation

## Description

**This my 2nd project result of Udacity self-driving car nanodegree (CarND) term 3. It's required to train a Fully Convolutional Network (FCN) to refer pixels of a road in images. The model is first transferred from a pretrained VGG16 model and Kitti road dataset is used for training.**

**The following demonstrates several inference results where green regions are referred as road regions:** 
   
![alt text][image]

* Udacity self-driving car nanodegree (CarND) :

  https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
  
* VGG16 model download (Fully convolutional version) :

  [VGG16 model](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
  
* Kitti Road dataset download :

  [Kitti Road dataset](http://www.cvlibs.net/download.php?file=data_road.zip)

[//]: # (Image References)
[image]: ./images/image.gif
[image1]: ./images/1.png
[image2]: ./images/2.png
[image3]: ./images/3.png
[image4]: ./images/4.png
[image5]: ./images/5.png

## Setup
#### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
#### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Run
Run the following command to run the project:
```
python main.py
```

### File structure

* The implementations are in `./main.py` file, which includes  `load_vgg` function for loading a pretrained VGG16 model, `layers` function for FCNs layer design, `optimize` function for defining loss and optimizer used and `train_nn` function for training based on different epochs and batches.

* Some helper functions used for downloading a pretrained vgg model, generating batches, generating test output and saving inference samples are already provided in `./helper.py` file. Besides, function test cases are provided in `project_test.py` file.

* Inference samples will be saved in `runs` folder after running the project.

## Architecture
### Fully Convolutional Network (FCN) layers
In this project, FCN model is based on a pretrained VGG-16 model. First, input layer, keep probability layer, layer 3, layer 4 and layer 7 are grabbed from the model. 

Next, layer 7 is connected to convolutional layer with (kernel, stride) = (1, 1) and then connected to deconvolutional layer with (kernel, stride) = (4, 2). Layer 4 is also connected to convolutional layer with (kernel, stride) = (1, 1). The above two output layers are added to form the first skip layer.

Then, the first skip layer is connected to convolutional layer with (kernel, stride) = (4, 2). The output of layer 3 connecting to convolutional layer with (kernel, stride) = (1, 1) add the above output layer to form the second skip layer.

Finally, the second skip layer is connected to deconvolutional layer with (kernel, stride) = (16, 8) to form the final output layer.

All convolution and deconvolution layer using kernel initializer with standard deviation 0.01 and L2 regularizer 0.001.

### Optimizer

Cross-entropy loss function and Adam optimizer are used in optimizer.

### Hyperparameter 

Learning rate is fixed at 0.0001 and dropout is set as 50%. Different hyperparamters used are epochs and batch size :

* Batch size : 5, 10
* Epochs : 11, 21, 31, 41

For batch size 5, mean losses of epochs 11, 21, 31 and 41 are 
0.0541, 0.0300, 0.0205 and 0.0180, respectively.

For batch size 10, mean losses of epochs 11, 21, 31 and 41 are 
0.0517, 0.0365, 0.0277 and 0.0235, respectively.

Mean losses of both cases decrease and saturate to a certain level. The final model is trained under batch size 5 and epochs 41.

### Inference result

The following images are several inference results of the final model : 

![alt text][image1]

![alt text][image2]

![alt text][image3]

![alt text][image4]

![alt text][image5]



