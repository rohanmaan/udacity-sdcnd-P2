#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

![visualization](https://github.com/rohanmaan/udacity-sdcnd-P2/blob/master/examples/visualization.jpg)
![random_noise](https://github.com/rohanmaan/udacity-sdcnd-P2/blob/master/examples/random_noise.jpg)


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rohanmaan/udacity-sdcnd-P2/blob/master/Traffic_Signs_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

![data explore](https://github.com/rohanmaan/udacity-sdcnd-P2/blob/master/examples/dataExploreIMG.png)

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this results in a slightly better performance. In addition, this will make our network easier to train, since the input is a 32x32x1 image instead of 32x32x3. Besides, by examining the different signs to identify, we observe that there is sufficient information in the shape of the sign and its contents to rely on grayscale information. A different case would have been to classify between 'normal' signs and 'on-road work' signs, which are distinguished only by the background color in Europe (white/yellow).

Here is an example of a traffic sign image before and after grayscaling.

![graysacale](https://github.com/rohanmaan/udacity-sdcnd-P2/blob/master/examples/grayscale.jpg)

We implement a min-max scaling in the same way as it was done in the TensorFlow lab, given the good performance obtained. Another approach would have been to make each test image of zero mean and unit variance, but this would have taken considerably more time. Considering that the dataset (after generating fake data) is relatively big, we opted for a computationally cheaper solution.

I decided to generate additional data because after training on the raw dataset, we observed that the network had a not good enough performance on the test set (around 87%). As some students pointed out in the forums, the main reason for this is insufficient training data. Therefore we generated an additional number of jittered versions of each image. This proves to be a very powerful solution against overfitting and provides robustness against affine transformations in the image. The fake dataset could have been extended by jittering the color, adding noise to the image, etc. We generate a different number of extra images for each class, so that the overall training dataset is balanced.

Here is an exploratory visualization of the new data set. 

![additionalData](https://github.com/rohanmaan/udacity-sdcnd-P2/blob/master/examples/additionalDate.png)


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

![cnn_layer](https://github.com/rohanmaan/udacity-sdcnd-P2/blob/master/examples/cnn_layer.png)

It can be seen that it is a convolutional neural network, composed to:
Preprocessing step. The input image (RGB) of 32x32x3 pixels is transformed into grayscale and normalized.
1. First Layer: convolutional. Takes an input of 32x32x1 pixels and convolves it with 16 5x5x1 filters. Maxpooling reduces the size of the
output by a half, obtaining 16x16x16. Dropout is used.
2. Second layer: convolutional. Takes an input of 16x16x16 pixels, and convolves with 32 5x5x1 filters. Again, maxpooling and dropout is used. The output is 8x8x32.
3. Third layer: convolutional. Takes an input of 8x8x32 pixels, and convolves with 64 5x5x1 filters. Maxpooling and dropout are used. The output is 4x4x64. As can be seen, the convolutional layers form a pyramid, where the image size is reduced but the depth is increased.
4. Fourth layer: fully connected. We first flatten the previous input to a 1024 feature vector. The layer has 512 neurons.
5. Fifth layer: takes the previous 512-vector and outputs 43 neurons, the number of classes. ReLU and dropout are not used, since the output is directly connected to the classifier.

Classifier, consisting on a softmax function followed by argmax to determine the most likely class for the given input.
It's also worth mentioning that the filtering operations were performed using 'SAME' padding, in order to keep the dimensions of the original image, with stride of 1. The max pooling operation had stride 2 in order to reduce by 2 the size of the image and create a pyramid-like network.
The model is relatively simple (only 3 convolutional and 2 fully-connected layers), but it proves itself to be very powerful.


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model was trained using:

* **Cost function**: cross-entropy with logits, following the example from the MNIST TensorFlow Lab.

* **AdamOptimizer**. After watching lectures CS231n, it was observed that the GradientDescentOptimizer is not the best one, and actually pretty slow. AdamOptimizer was then recommended.

* **Learning rate**: 0.0001, as suggested in the following [TensorFlow tutorial](https://www.youtube.com/watch?v=HMcx-zY8JSg&list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ). The documentation of AdapOptimizer suggests 0.001, but we found a smaller learning rate to provide slightly better results. It's already a fast optimizer, so it is better to step a bit slower and make sure the learning happens in a proper manner. 

* **Batch size**: 128 for both training and testing, in order to fit in GPU memory (4 GB). A power of 2 was chosen since GPU architectures can take advantage of this for more efficient processing.

* **Dropout keep probability**: 0.5 for training and 1.0 for testing, as suggested in the lectures. This was extremely powerful to make the network more robust and prevent from overfitting.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The approach to solving the traffic sign recognition problem was to use a convolutional neural network. These type of networks are well suited for image processing problems (since convolutions are typical operations on them), 
and the number of required parameters is much smaller (since the weights are shared for every pixel in the image).

The main reference was the paper cited above, where they propose a 2 conv + 2 FC network architecture. Many TensorFlow tutorials used this approach so it seemed like a robust and well-studied solution. 

The following additions over the basic network pushed the test accuracy from around 87% to 93.4%:

* Added an extra convolutional layer. This bumped the accuracy to around 89%.

* Added dropout, which significantly improved the test accuracy, bumping it to around 92%.

* Generated fake data, giving the final push on the accuracy to 93.4%. It also helped a lot to increase the accuracy in my own images, since the training distribution was balanced.

My final model results were:
* training set accuracy of 99.2 %
* validation set accuracy of 99.5 % 
* test set accuracy of 93.4 %



If a well known architecture was chosen:
* What architecture was chosen?
For reference on how to build a deep neural network using TensorFlow, see [Deep Neural Network in TensorFlow
](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/83a3a2a2-a9bd-4b7b-95b0-eb924ab14432) from the classroom._
 

###Test a Model on New Images

####1. Choose few German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are few German traffic signs that I found on the web:

![newimages](https://github.com/rohanmaan/udacity-sdcnd-P2/blob/master/examples/testNewImage_1.png()

We have taken 10 images from the street, as shown in the picture above. The first two of them are pretty standard and should be easy to identify (stop, keep right). The third one has an sticker on it and a tree in the background which could pose some difficulty. The next two signs are speed limit signs, but with yellow background (indicating road work). This could pose problems for the network as it hasn't seen road work speed limits before. The following 3 signs have triangular shape but different symbols inside, so the network must recognize not only the external shape. 

Finally, the last two images correspond to night pictures, where the signs show up much blurrier and with less illumination. This tests the network under a completely different illumination condition, a typical problem in Computer Vision.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        	|     Prediction	        					| 
|:---------------------:	|:---------------------------------------------:| 
| Stop     					| Stop    										| 
| Keep right     			| Keep right 									|
| Priority road				| Priority road									|
| Speed limit (30km/h)		| Speed limit (20km/h)		 					|
| Speed limit (80km/h)		|Speed limit (80km/h)     						|
| Yield						| Yield 										|
| Road narrows on the right	| Road narrows on the right      				|
| Road work					| Road work     								|
| Priority road				| Priority road      							|
| Keep right				| Keep right      								|


We notice that the correct predictions had a huge confidence, very close to 100%, which is very positive.

The only missclasiffied image was the 30 km/h speed sign, so the network did not learn the numbers very well. However we can observe the top 5 probabilities and realize that the network was not that far anyway, since the correct option is the second with highest probability. In addition, all the top-5 outputs are speed limits, so at least the network learnt that.

The positive aspect is that the reported prediction confidence for the missclassified sign is not close to 100%, compared to the successfully predicted classes, so we would normally not take this as a valid prediction and would let the network try again with a new image.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is absolutely sure that this is a stop sign (probability of 0.99), and the image is a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop    										| 
| .0     				| Keep right 									|
| .0					| No entry										|
| .0	      			| Turn left ahead					 			|
| .0				    | Priority road      							|

For the second image, the model is absolutely sure that this is a Keep right sign (probability of 1.0), and the image is a Keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Keep right    								| 
| .0     				| Priority road 								|
| .0					| Turn left ahead								|
| .0	      			| Go straight or right					 		|
| .0				    | Roundabout mandatory      					|

For the third image, the model is absolutely sure that this is a Priority road sign (probability of 0.985), and the image is a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.985         		| Priority road    								| 
| .006    				| Roundabout mandatory 							|
| .002					| Speed limit (30km/h)							|
| .002	      			| Speed limit (80km/h)					 		|
| .001				    | Speed limit (50km/h)      					|

For the fourth image, the model is sure that this is a Speed limit (20km/h)sign (probability of 0.886), but the image is a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.985         		| Speed limit (20km/h)    						| 
| .043    				| Speed limit (30km/h) 							|
| .022					| Speed limit (50km/h)							|
| .018	      			| Roundabout mandatory					 		|
| .012				    | Speed limit (100km/h)      					|

For the fifth image, the model is sure that this is a Speed limit (80km/h)sign (probability of 0.741), and the image is a Speed limit (80km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.741         		| Speed limit (80km/h)    						| 
| .099    				| Speed limit (50km/h) 							|
| .088					| Speed limit (60km/h)							|
| .036	      			| Right-of-way at the next intersection			|
| .009				    | Dangerous curve to the right      			|

For the fifth image, the model is sure that this is a Speed limit (80km/h)sign (probability of 0.741), and the image is a Speed limit (80km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.741         		| Speed limit (80km/h)    						| 
| .099    				| Speed limit (50km/h) 							|
| .088					| Speed limit (60km/h)							|
| .036	      			| Right-of-way at the next intersection			|
| .009				    | Dangerous curve to the right      			|

For the sixth image, the model is absolutely sure that this is a Yield sign (probability of 1.0), and the image is a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield    										| 
| .0     				| Go straight or right 							|
| .0					| Priority road									|
| .0	      			| Ahead only					 				|
| .0				    | End of all speed and passing limits      		|

For the seventh image, the model is sure that this is a Road narrows on the right sign (probability of 0.881), and the image is a Road narrows on the right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.881         		| Road narrows on the right     				| 
| .066    				| Right-of-way at the next intersection 		|
| .038					| Road work										|
| .007	      			| Pedestrians					 				|
| .002				    | Beware of ice/snow      						|

For the eighth image, the model is sure that this is a Road work sign (probability of 0.847), and the image is a Road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.847         		| Road work    									| 
| .096    				| Go straight or left 							|
| .017					| Road narrows on the right						|
| .010	      			| Right-of-way at the next intersection			|
| .009				    | Roundabout mandatory      					|

For the ninth image, the model is absolutely sure that this is a Priority road sign (probability of 1.0), and the image is a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority road    								| 
| .0     				| Speed limit (30km/h) 							|
| .0					| Keep right									|
| .0	      			| Speed limit (50km/h)					 		|
| .0				    | Speed limit (60km/h)      					|

For the tenth image, the model is absolutely sure that this is a Keep right sign (probability of 1.0), and the image is a Keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Keep right    								| 
| .0     				| Turn left ahead 								|
| .0					| Priority road									|
| .0	      			| Go straight or right					 		|
| .0				    | Stop      									|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


