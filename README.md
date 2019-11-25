# Evolution-COnstructed Object Recognition (Under Dev.)

@author: Jia Geng

@email: jxg570@miami.edu  


## Dev Log 

10/15:  Init

10/17:  Implemented the Creature Class (still need to implement the perceptron)

10/19:  Implemented the perceptron classifier and the confusion matrix, fitness score

10/20:  Fixed some bugs. The fitness can be overwhelmed by imbalanced data. 

10/21:  Implemented the population operator.

10/23:  Added mutation function for creature patch coors; Added reproduce related functions.

10/31:  Fixed a bug. Now the child will have at least 1 gene from parent 1 and ata most all gene from the 2 parents.

10/31:  Refactored the train/validate and lock strategy. Perceptron now using early stopping.

11/1:   Added a method to save & load the params into/from json file

11/4:   Refactored the framework. Now should work with any defined number of classes.

11/11:  Changed the img processing pipeline from img -> cropped patch -> processed patch to img -> processed img -> cropped patch for better compatibility

11/16:  Refactored method with Enum

11/21:  Refactored the perceptron, now use sklearn packages instead of scratch model.

11/22:  Added a small gauss number to the fitness score to prevent unstable behaviour. Now the result will be identical if using fixed random seed.

## Dev. Plan

This framework is currently under development for my research project. The framework was (and will continue to be) tested on my research project dataset (full dataset will not be released at least in the next two years).

The current weak classifier is multiclass perceptron, regularized simply by early stopping. During the boosting (SAMME), the prediction vote will be ecoded in: 1 for predicted class and -1/(C-1) for other classes. One drawback for the multi-class perceptron is that it assume each class is independed to each other including the boosting stage.
 
- implement visualization function 
- implement non-linear SVM 
- clean up the depreciated code


## Introduction

This is a genetic programming based computational framework for constructing Evolution-COnstructed (ECO) features for object detection. This method was initially proposed by __Lillywhite et al. (2013)__ and extended by __Zayyan et al. (2018)__. 

Genetic programming provides an efficient way to combine specialized basic image feature filters to build a more complex image feature extractors. The compound feature extractors can be used for training weak classifiers such as perceptron, SVM, ANN etc. which can then be boosted to a stronger classifer. 

## How does it work?

### Some key objects in this framework:

__Creature__: Creature is the basic ECO feature extraction unit. Creatures are constructed with randomly generated feature filters, and randomly generated coordinates for cropping an image. They are able to crop the input image and apply compound features on the subimage to extract the image features. Creatures can be trained with perceptron/SVN/etc. so that they will become a weak classifier. Here, the creature was implemented as an python object class. The field contains variables such as feature filter functions (chromosomes), cropping coordinates, weights, confusion matrix, etc so that it will be capable of provide essenstial computational functions.

__Chromosomes__: A chromosome is a list of image feature extractors.

__Generation__: A generation is a batch of creatures, e.g. 500 randomly generated creatures. 

### Pipeline:

__A. Weak Classifiers Generation:__

1 - Init: Start with a large population of creatures (1st generation)

2 - Train: Train all creatures on the training data so they become weak classifiers. Validate all the creatures.

3 - Eliminate: Eliminate the underperformed creatures. The rest become parent candidate pool.

4 - Reproduce: Randomly select a pair of creatures from the parent candidate pool. Cross the paresnts chromosome to generate child creature (reproduce). Each children creature have a small chance to mutate (changing the parameters of the filters). Reproduce enough number of children creature as the next generation.
5 - go to step 2.

From step 2 -> step 4 is one generation. Repeat 10 generations and this genetic framework should be able to select well performed creatures, i.e. weak classifiers.

__B. Boosting__

Adaboost the weak classifers. 

### Example

__1 - Data Preparation__

- Put your image data named with integer id (__.png format__, e.g. 1.png, 2.png. 3.png) into a folder. Image must be
 in same size.
- You need to encode the image category (label) into 0, 1, 2, 3 etc.
- For training/testing/validation data, you need to prepare the data as a list of tuples. E.g. `[(img1_id, img1_label
), (img2_id, img2_label), ...]`. 
- The program will read each tuple and find the corresponding image using the image id and the image src folder
 directory you provide. The label will be feed into the model as ground truth for training or testing. 

__2 - Weak Classifier Training__


```
from src.ga import PopulationOperator as po

# prepare your  data and your img folder directory
train_data = [...]  # for training weak classifier
hol_data = [...]    # for validating the weak classifier and eliminate the bad performer
boost_data = [...]  # for train the booster, you can also use train_data + hol_data for training the booster
img_src = '/path/to/img/folder'


# create a new populations of weak classifier. Assume your images are in 49x49
# use 500 weak classifiers, the more classifiers you have, the more time will be needed to train the model
first_generation = po.new_population(img_shape=(49, 49), num=500)

# train the weak classifiers
# e is a parameter for early stopping, epoch limit is the maximum number of epoch
po.train_population(first_generation, train_data, img_src, e=0.025, epoch_limit=100)

#

```


## What kind of data does it work well with?

This method tend to converge on some sensitive sub-area of an image along with some useful feature extractor. Intuitively, if the object of interest always locate on the center of the image, this method should be able to work stablely and nicely. 
Besides, this method provides highly interpretable classifiers. 

This method achieved very high accuracy for some early time dataset, e.g. Caltech101. But this framework might not work well on the more challenging datasets, especially when the object of interest does not have a certain location pattern in the image (no reports are available). 

This method might work well on the moving object proposals detected by the moving object algorithms, e.g., background subtraction as it can generate proposals with object of interest on the image center. 


## Reference

[1] Lillywhite, K., Lee, D.-J., Tippetts, B., & Archibald, J. (2013). A feature construction method for general object recognition. _Pattern Recognition_, 46, 3300–3314.

[2] Zayyan, M. H., AlRahmawy, M. F., & Elmougy, S. (2018). A new framework to enhance Evolution-COnstructed object recognition method. _Ain Shams Engineering Journal_, 9, 2795–2805.
