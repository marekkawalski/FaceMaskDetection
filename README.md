# FaceMaskDetection

It is said that need is the mother for advancement. Covid-19 pandemic made us think what we could do to bring about a change. The idea we came up with was creating this project which provides a solution that can help institutions to verify whether a person is wearing a mask or not. 

To achieve our goal, we created two models- one for binary classification of whether someone is wearing a mask and the other to detect 4 states (the initial classification has extended to 4 possible scenarios, namely: mask on, mask off, mask on chin, mask covers mouth). The models were then trained couple of times. The last step that we performed was creating an executable program which makes use of previously mentioned models and a built-in camera to detect masks in real life video stream.

## Analysis of the task

a) Possible approaches to solve the problem
Having got acquainted with the topic, we dived into a problem how to teach the neural network to differentiate how mask is worn in images. One of the ideas that draw our attention was utilizing a pretrained model – transfer learning. It has multiple pros: it’s not mandatory to start the learning process from scratch, it’s efficient in terms of data, it’s reusable. 
It's worth mentioning that all training was done on GPU. It took a tremendous amount of time to get it up and running but in the end, we succeeded and saved some time compared to performing all the training on a CPU.

b)	Datasets
We used two datasets in our project. The first one for binary detection whether someone is wearing a mask or not. In that dataset: 
-	Pictures are divided into folders, and each has almost two thousand photos.
-	Some of the photos are of someone facing forward or sideways, wearing glasses, headgear, etc.
-	In the folder without the mask, there are also pictures like a person with their hand partially covered with their mouth or with a cup at their mouth.

The second dataset was used to provide model with extra classes. In that dataset:
-	There are four types of photos: with no mask on the face, with the mask worn under the nose, with the mask worn on the chin, with no mask.
-	The original size of this dataset is 500GB. For our purposes we extracted 12 thousand photos from the original dataset and placed them in 4 different folders so that 3 thousand photos go to each of them.
-	It’s made in such a way that one person is shown in four different ways according to classes.
There were multiple datasets containing people with mask or without but only a handful of datasets divided into 4 categories.

c)	Tools
There were multiple tools to choose from. We choose the ones listed below:
IDE:
- Jupyter Notebook that works on a GPU by using Cuda toolkit and CudaNN
Libraries/Frameworks: 
-	Tensorflow- open-source software for machine learning.
-	Keras (MobileNetV2)- a deep learning API written in Python, running on top of the machine learning platform TensorFlow. We use transfer learning so the base model in our project is MobileNetV2 which is a type of convolutional neural network designed for mobile and embedded vision applications.
-	Opencv- used in video stream, to capture frame with a face and outline the result
-	Imutils- used for video stream
-	Matplotlib- used to create plots for training loss and accuracy

# Internal and external specification of the software solution
a)	Classes/objects/methods/scripts

To split the second dataset we wrote a script „split-data.ipynb” which changes folder depending on the string found in a picture name.
Firstly, we changed image parameters such as: rotation, zoom, width, height, shift and shear using ImageDataGenerator from Keras. 
In file “program” we created a function “detect_and_predict_mask” which finds frames with faces and using our model assesses whether mask is worn or not. 

b)	Data structures

Datasets were divided into folders. The number of folders is equivalent to number of classes. These folders can be found in folders „dataset” and „datasetV2”. First dataset was divided into classes right away. 
On Github, there are files “mask-detection-model.ipynb” and “mask-detection-model-4classes.ipynb”. The first one is used to train model containing 2 classes, the second one to train model with 4 classes. There are also files “program.ipynb and “ and “program-4classes.ipynb”. These files are executables which utilize built-in camera to detect how mask is worn using our models.
Referring to the models, they go by the name “mask_detectorINIT_LR_X,EPOCHS_Y,BS_Z.model” where: 
-	X is init learning, 
-	Y is number of epochs
-	Z is batch size
-	Additionally, model which detects 4 classes has a suffix “4classes”
Other files which can be found are .png files. Their naming convention is similar to models. These pictures represent training loss and accuracy.

c)	GUI

GUI is simple yet effective. It only consists of one window with a stream from camera. The output comprises a frame that goes around the face and depending on how mask is worn is either green or red. To quit the program, user is to press q.


