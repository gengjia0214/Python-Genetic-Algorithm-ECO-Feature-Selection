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

11/22:  Added a small gauss number to the fitness score to prevent unstable behaviour. Now the result will be identical if using fixed random seed


## Introduction

This is a genetic programming based computational framework for constructing Evolution-COnstructed (ECO) features for object detection. This method was initially proposed by __Lillywhite et al. (2013)__ and extended by __Zayyan et al. (2018)__. 

Genetic programming using evolutionary strategy to combine the basic image feature filters and select the sub-region of an image for feature extraction. The compound feature extractor can be used for training weak classifiers such as perceptron, SVM, ANN etc., which can be further boosted into a stronger classifer. 

## How does it work?

### Some key objects in this framework:

__Creature__: Creature is the basic ECO feature extraction unit. Creatures are constructed with randomly generated feature filters, and randomly generated coordinates for cropping the image. They are able to crop the input image and apply combined features on the imge subregion to extract local features. Creatures can be trained with perceptron/SVM/etc. as weak classifiers. 

__Chromosomes__: A chromosome is a sequence of image feature filters.

__Generation__: A generation is a batch of creatures, e.g. 500 randomly generated creatures. 

### Pipeline:

__A. Weak Classifiers Generation:__

1 - Init: Start with a large population of creatures (1st generation)

2 - Train: Train all creatures on the training data. Validate all the creatures on a holdout dataset.

3 - Eliminate: Eliminate the underperformed creatures based on performance on the holdout set. The rest of creatures will be in the parents pool.

4 - Reproduce: Randomly select a pair of creatures from the parents pool. Cross the paresnts chromosome to generate child creature (reproduce). Each children creature have a small chance to mutate (changing the parameters of the filters). Reproduce enough number of children creature as the next generation.
5 - go to step 2.

From step 2 -> step 4 is one generation. Repeat 10 generations and this genetic framework should be able to select well performed creatures, i.e. weak classifiers.

__B. Boosting__

Adaboost the weak classifers. 
Check https://github.com/gengjia0214/Python-Multiclass-AdaBoost-SAMME.git

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
from src.ga import PopulationOperator as po, Eliminiation, Model
import os

# prepare your  data and your img folder directory
img_src = '/path/to/img/folder'
dst_dir = 'path/to/dst'
train_data = [...]  # for training weak classifier
val_data = [...]    # for validating the weak classifier and eliminate the bad performer
boost_data = [...]  # for train the booster, you can  use train_data + val_data for training the booster
num_gen = 10        # 10 generations



# create a new populations of weak classifier. Assume your images are in 49x49
# use 500 weak classifiers, the more classifiers you have, the more time will be needed to train the model
first_generation = ga.new_population(img_shape=(49, 49), num=500)

# train the weak classifiers using genetic algorithms for feature evolution
# prepare the args (E.g. using the logistic classifier)
args = {'penalty': penalty, 'solver': 'saga', 'max_iter': 100, 'n_jobs': -1, 'random_state': np.random.RandomState(77), 'multi_class': 'multinomial', "C": c, 'dual': False} 
model = ga.Model.LOGIT_MUL
elimination_mode = ga.Elimination.BY_CAT

curr_gen = first_gen
for i in range(num_gen):
    print("\n======================================================")
    print("Start Generation {}".format(i))
    print("======================================================\n")
    model_log = "{}_fold={}_gen={}.json".format(trial_id, fold, i)
    out_path = os.path.join(dst_dir, model_log)
    ga.PopulationOperator.train_population(curr_gen, train_data, img_src, mode=model, args=args,
                                           silence=silence)
    ga.PopulationOperator.validate_population(curr_gen, val_data, img_src, mode=model)
    ga.PopulationOperator.save_population(curr_gen, dst_file=os.path.join(dst_dir, "{}.json".format(i))
    ga.PopulationOperator.eliminate_population(curr_gen, mode=elimination_mode, t=0.25)
    curr_gen = ga.PopulationOperator.reproduce(curr_gen, num=500)
```
__3 - Boosting__

Check https://github.com/gengjia0214/Python-Multiclass-AdaBoost-SAMME.git
The serializtion and de-serialization of the model ensemble was implemented but currently for internal usage only.
It is fairly easy to modify the SAMME class and implement the serialization and de-serialzation for the ensemble.

Simply clip the serialized eco model and the boosting params together. `ga.PopulationOperator.save_population()` will return the serilized model when `dst_file=None` 

## What kind of data does it work well with?

The weak classifiers should evolve toward meaningful sub-area of an image and also generate some useful combination of feature filters. Intuitively, if the objects of interest are always at a certain location of the image (E.g. the center), this method should be able to work stablely and nicely. Besides, this method provides highly interpretable classifiers. 

This method achieved good performance for some early-time dataset, e.g. Caltech101 (tested by the orginal author). But this framework might not work well on the more challenging datasets, especially when the object(s) of interest will not appear at certain location of the image (still need to be tested). This method might work well on the moving object proposals detected by the background subtraction algorithms as it can generate bounding boxes with center-located object proposals. One draw back of this method is the slow training and testing speed when the population gets large or image gets large (the evolution does not contribute to better result when population is low). A GPU implementation of both the feature filters and the training algorithms could greatly accelerate this method.

## Dev. Plan

This framework is currently under development for my research project. The framework was (and will continue to be) tested on my research project dataset (full dataset will not be released before the end of 2020).

The current weak classifiers include perceptron, multinomial logistic regression and one-vs-all SVM. For the boosting algorithm (SAMME, implemented at another repo), the prediction vote will be ecoded in: 1 for predicted class and -1/(C-1) for other classes. 

Future plan includes:
- improve the speed of the framework
- implement visualization function 
- try more powerful weak classifers

## Reference

[1] Lillywhite, K., Lee, D.-J., Tippetts, B., & Archibald, J. (2013). A feature construction method for general object recognition. _Pattern Recognition_, 46, 3300–3314.

[2] Zayyan, M. H., AlRahmawy, M. F., & Elmougy, S. (2018). A new framework to enhance Evolution-COnstructed object recognition method. _Ain Shams Engineering Journal_, 9, 2795–2805.
