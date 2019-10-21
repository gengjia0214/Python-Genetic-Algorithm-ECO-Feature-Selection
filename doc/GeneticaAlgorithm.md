# Genetic Algorithm

## Building Blocks

### Terms and Elements

- __chromosome__: a solution
- __gene__: an element of a solution, usually marked and represented by alphabetic character 
- __locus__: the alphabetic character that marks the position of the gene
- __parents__: chromosomes with acceptable performance which are selected to produce off springs
- __offspring__: new chromosomes produced by the selection -> crossover -> mutation

### GA Operators

- __selection__: operator that select chromosomes to reproduce the offsprings
- __crossover__: operator that randomly choose a locus and exchange the subsequence before and after that locus between
 two chromosomes to produce two off springs
- __mutation__: randomly (small prob e.g. 0.001) change some of the gene in a chromosome. 

### Algorithm

```
1. Calculate the compute_fitness f(x) of each string x in the population.
2. Choose (with replacement) two parents from the current population with probability proportional to each string's
 relative compute_fitness in the population.
3. Cross over the two parents (at a single randomly chosen point) with probability p c to form two offspring. (If no
 crossover occurs, the offspring are exact copies of the parents.) Select one of the offspring at random and discard
 the other.
4. Mutate each bit in the selected offspring with probability p m , and place it in the new population.
5. Go to step 2 until a new population is complete.
6. Go to step 1.
```

__Generation__

Each iteration of this process is called a generation. A GA is typically iterated for anywhere from 50 to 500 or more
generations. The entire set of generations is called a run. At the end of a run there are often one or more highly
fit chromosomes in the population. Since randomness plays a large role in each run, two runs with different random
number seeds will generally produce different detailed behaviors. GA researchers often report statistics (such as
the best compute_fitness found in a run and the generation at which the individual with that best compute_fitness was discovered
) averaged over many different runs of the GA on the same problem.

__Selection Approach__

- Roulette Wheel Sampling
- Stochastic Universal Sampling
- Sigma Scaling
- Rank Selection
- Tournament Selection: Two individuals are chosen at random from the population. A random number r is then chosen
 between 0 and 1. If r < k (where k is a parameter, for example 0.75), the fitter of the two individuals is selected
  to be a parent; otherwise the less fit individual is selected.

__Cross Over__

- one point cross over -> positional bias
- two point cross over -> less positional bias
- parameterized uniform cross over -> no positional bias but could disrupt schema

Two point and parameterized uniform are more commonly used. p_c usually 0.7-0.8. 

__Hyper-parameters__

|GA Params|Value|
| --- | --- |
|No. Generations|10|
|Size of Pop.|100 (maybe higher)|
|Max Gene Len.|8|
|Selector|Tournament|
|Tournament Pool Size|2|
|Cross Over Operator|Cut and Slice|
|Cross Over Rate|0.9|
|Mutate Rate|0.0005|

### Transformer Candidates

|Transformer|Intention|Param|Input dtype|Output dtype|
|---|---|---|---|---|
|AdaptiveThreshold|Thresholding|3(4)|CV_8U|Filtered Img CV_32F
|CannyEdge|Edging|1(2)|CV_8U|Edge Pixels CV_32F
|CensusTransformation|Pixel Local Ranking|0|CV_32F|Census Image CV_32F
|CLAHistogram|Image Equalization|2(2)|CV_8U|Equalized Image CV_32F
|HistogramEqualization|Image Equalization|0|CV_8U|Equalized Image CV_32F
|DistanceTransformation|Image Metrics|2(2)|CV_32F|Distance Metrics CV_32F
|Dilate|Thicken Image|1(2)|CV_32F|Thickened Image CV_32F
|Erode|Thinning Image|1(2)|CV_32F|Thinned Image CV_32F
|DifferenceGaussian|Edge Filter|2(4)|CV_32F|Normalized Filtered Image CV_32F
|GaussianBlur|Smoothing|2(2)|CV_32F|Smoothed Image CV_32F
|Gradient|Gradient Filter (Edge)|x(y)|CV_32F|Normalized Filtered Image CV_32F
|HarrisCorner|Corner Detection|2(2)|CV_32F|Normalized Corner Prob. Map CV_32F
|IntegralTransformation|Image Integral|0|CV_32F|Normalized Integral CV_32F
|LaplacianEdge|Edge Detection|1(1)|CV_32F|Normalized Filtered Image CV_32F
|Log|Contrast Enhancement|0(1)|CV_8U|Contrast Enhanced Image CV_32F
|MediumBlur|Smoothing|1(1)|CV_8U/CV_32F|Smoothed Image CV_32F
|SquareRoot|Contrast Enhancement|0|CV_32F|Contrast Enhanced Image CV_32F

## Reference
@reference: 

[1] M. Mitchell, An Introduction to Genetic Algorithms, The MIT press, 1998.

[2] Lillywhite, K., Lee, D.-J., Tippetts, B., & Archibald, J. (2013). A feature construction method for general
 object recognition. _Pattern Recognition_, 46, 3300–3314.

[3] Zayyan, M. H., AlRahmawy, M. F., & Elmougy, S. (2018). A new framework to enhance Evolution-COnstructed object
 recognition method. _Ain Shams Engineering Journal_, 9, 2795–2805.