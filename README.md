# Evolution-COnstructed Object Recognition

## Introduction

This is a genetic programming based computational framework for constructing Evolution-COnstructed (ECO) features for object detection. This method was initially proposed by Lillywhite et al. (2013) and extended by Zayyan et al. (2018). 


This method provide an efficient way to combine specialized baisc image feature filters to build a more complex image feature extractors. The compound feature extractors can be used for training weak classifiers such as perceptron, SVM, ANN etc. which can then be boosted to a stronger classifer. 

This method achieved high accuracy for some early time dataset, e.g. Caltech101. Although this framework might not work well on the more challenging datasets (no reports are available). This method provides a very intepertable classifier and might works well on image data that focuses on the target object. 

## Dev Log

10/15:  Init
10/17:  Implemented the Creature Class (still need to implement the perceptron)
10/19:  Implemented the perceptron classifier and the confusion matrix, fitness score
10/20:  Fixed some bugs. The fitness can be overwhelmed by imbalanced data. 
        Up-sampling (collect more samples or generate simulated data) or increase the weight of NECR for the fitness 
        score
10/21:  Implemented the population operator.

## Reference
[1] Lillywhite, K., Lee, D.-J., Tippetts, B., & Archibald, J. (2013). A feature construction method for general object recognition. _Pattern Recognition_, 46, 3300–3314.
[2] Zayyan, M. H., AlRahmawy, M. F., & Elmougy, S. (2018). A new framework to enhance Evolution-COnstructed object recognition method. _Ain Shams Engineering Journal_, 9, 2795–2805.
