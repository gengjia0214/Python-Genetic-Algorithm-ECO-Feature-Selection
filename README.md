# Evolution-COnstructed Object Recognition (Under Dev.)

## Introduction

This is a genetic programming based computational framework for constructing Evolution-COnstructed (ECO) features for object detection. This method was initially proposed by __Lillywhite et al. (2013)__ and extended by __Zayyan et al. (2018)__. 

Genetic programming provides an efficient way to combine specialized basic image feature filters to build a more complex image feature extractors. The compound feature extractors can be used for training weak classifiers such as perceptron, SVM, ANN etc. which can then be boosted to a stronger classifer. 

## How does it work?

Some key objects in this framework:

__Creature__: Creature is the basic ECO feature extraction unit. Creatures are constructed with randomly generated feature filters, and randomly generated coordinates for cropping an image. They are able to crop the input image and apply compound features on the subimage to extract the image features. Creatures can be trained with perceptron/SVN/etc. so that they will become a weak classifier. Here, the creature was implemented as an python object class. The field contains variables such as feature filter functions (chromosomes), cropping coordinates, weights, confusion matrix, etc so that it will be capable of provide essenstial computational functions.

__Chromosomes__: A chromosome is a list of image features.

__Generation__: A generation is a batch of creatures, e.g. 500 randomly generated creatures. 

Pipeline:

__A. Weak Classifiers Generation:__

1 - Init: Start with a large population of creatures (1st generation)
2 - Train: Train all creatures on the training data so they become weak classifiers. Validate all the creatures.
3 - Eliminate: Eliminate the underperformed creatures. The rest become parent candidate pool.
4 - Reproduce: Randomly select a pair of creatures from the parent candidate pool. Cross the paresnts chromosome to generate child creature (reproduce). Each children creature have a small chance to mutate (changing the parameters of the filters). Reproduce enough number of children creature as the next generation.
5 - go to step 2.

From step 2 -> step 4 is one generation. Repeat 10 generations and this genetic framework should be able to select well performed creatures, i.e. weak classifiers.

__B. Boosting__

Adaboost the weak classifers. 

## What kind of data does it work well with?

This method tend to converge on some sensitive sub-area of an image along with some useful feature extractor. Intuitively, if the object of interest always locate on the center of the image, this method should be able to work stablely and nicely. 
Besides, this method provides highly intepertable classifiers. 

This method achieved very high accuracy for some early time dataset, e.g. Caltech101. But this framework might not work well on the more challenging datasets, especially when the object of interest does not have a certain location pattern in the image (no reports are available). 

This method might work well on the moving object proposals detected by the moving object algorithms, e.g., background subtraction as it can generate proposals with object of interest on the image center. 


## Dev Log 

Currently the implementation is for my own research so it is specialized for classify 4 different categories (encoded into 0, 1, 2, 3). 

10/15:  Init

10/17:  Implemented the Creature Class (still need to implement the perceptron)

10/19:  Implemented the perceptron classifier and the confusion matrix, fitness score

10/20:  Fixed some bugs. The fitness can be overwhelmed by imbalanced data. 

10/21:  Implemented the population operator.

10/23:  Added mutation function for creature patch coors; Added reproduce related functions.

10/31:  Fixed a bug. Now the child will have at least 1 gene from parent 1 and ata most all gene from the 2 parents.

10/31:  Refactored the train/validate and lock strategy. Perceptron now using early stopping.

11/1:   Added a method to save & load the params into/from json file


## Reference

[1] Lillywhite, K., Lee, D.-J., Tippetts, B., & Archibald, J. (2013). A feature construction method for general object recognition. _Pattern Recognition_, 46, 3300–3314.

[2] Zayyan, M. H., AlRahmawy, M. F., & Elmougy, S. (2018). A new framework to enhance Evolution-COnstructed object recognition method. _Ain Shams Engineering Journal_, 9, 2795–2805.
