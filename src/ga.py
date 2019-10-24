import numpy as np
from src import transformer as tfm
import random
import cv2 as cv
import os

"""
Dev Log
10/17:  Implemented the Creature Class (still need to implement the perceptron)
10/19:  Implemented the perceptron classifier and the confusion matrix, fitness score
10/20:  Fixed some bugs. The fitness can be overwhelmed by imbalanced data. 
        Up-sampling (collect more samples or generate simulated data) or increase the weight of NECR for the fitness 
        score
10/21:  Implemented the population operator.
10/23:  Added mutation function for creature patch coors; Added reproduce related functions.
"""


class Creature:
    """
    Creature - the basic classifier that taking random set of feature extractor and a random cropped subpatch of an
    image
    """

    def __init__(self, img_shape: tuple):
        """
        Constructor.
        Create a creature with random patch window and empty chromosome
        :param img_shape: original image size (height x weight)
        """

        self.img_shape = img_shape
        # sub-patch should be at least 3x3
        # below could use some memory if the image is large?
        x1, y1 = random.randrange(self.img_shape[0]), random.randrange(self.img_shape[1])  # 0 ~ 48
        x2_range = [x2 for x2 in range(self.img_shape[0]) if abs(x2 - x1) >= 3]
        y2_range = [y2 for y2 in range(self.img_shape[1]) if abs(y2 - y1) >= 3]
        x2, y2 = random.choice(x2_range), random.choice(y2_range)

        # x2, y2 are not included
        # for the sub patch just call arr[x1:x2, y1:y2]
        self.x1, self.x2 = min(x1, x2), max(x1, x2)
        self.y1, self.y2 = min(y1, y2), max(y1, y2)
        self.height, self.width = self.x2 - self.x1, self.y2 - self.y1

        # TODO: need a better initialization?
        # TODO: customized cats in the future
        self.weights = np.zeros((4, self.height * self.width + 1))
        self.chromosome = []
        self.fitness_score = None
        self.confusion = np.zeros((4, 4), dtype=np.int16)
        self.lock = False
        self.survive = True

    def give_birth(self):
        """
        # TODO: decide whether use the shallow chromosome copy or deep copy. Currently use shallow.
        # TODO: shallow should be fine as the parent generation's info will be saved before the new generations
        Return a child creature that have same setup as the parent creature except for the weights, etc.
        Child's chromosome will be a shallow copy from the parent.
        I.E. list objects are different object but the tfm elements in the list are shared between parents and child
        :return:
        """
        child = Creature(self.img_shape)
        child.x1, child.x2, child.y1, child.y2 = self.x1, self.x2, self.y1, self.y2
        child.height, child.width = self.height, self.width
        child.weights = np.zeros((4, self.height * self.width + 1))
        child.chromosome = self.chromosome.copy()  # shallow copy, tfm objects in the list are shared between generation
        return child

    def build_chromosome(self, gene_pool_size=17, length_limit=8):
        """
        Build the chromosome with random length
        :return:
        """

        n = random.randrange(1, length_limit+1)  # n is the chromosome length
        gene_seq = random.sample(range(1, gene_pool_size+1), n)  # randomly take n gene from the pool

        for gene in gene_seq:
            self.chromosome.append(self.create_tfm(gene))  # build the chromosome using the randomly generated gene

    def mutate(self, r=0.0005):
        """
        Mutate. Only happen when generating offsprings.
        This does not reset the weights
        :return: void
        """

        # mutate the patch coordinates
        img_h = self.img_shape[0]
        img_w = self.img_shape[1]

        if np.random.choice(2, 1, p=[1 - r, r]) == 1:
            x1_range = [x1 for x1 in range(img_h) if abs(x1 - self.x2) >= 3]
            y1_range = [y1 for y1 in range(img_w) if abs(y1 - self.y2) >= 3]
            self.x1, self.y1 = random.choice(x1_range), random.choice(y1_range)

        if np.random.choice(2, 1, p=[1 - r, r]) == 1:
            x2_range = [x2 for x2 in range(img_h) if abs(x2 - self.x1) >= 3]
            y2_range = [y2 for y2 in range(img_w) if abs(y2 - self.y1) >= 3]
            self.x2, self.y2 = random.choice(x2_range), random.choice(y2_range)

        # need to check the order after the mutate on the coordinates
        self.x1, self.x2 = min(self.x1, self.x2), max(self.x1, self.x2)
        self.y1, self.y2 = min(self.y1, self.y2), max(self.y1, self.y2)
        self.height, self.width = self.x2 - self.x1, self.y2 - self.y1

        # mutate the gene
        for gene in self.chromosome:
            gene.mutate(r=r)

    def reset_confusion(self):
        """
        Reset the confusion matrix and the weights
        :return:
        """
        self.confusion = np.zeros((4, 4), dtype=np.int16)

    def reset_weights(self):
        """
        Reset the weights
        :return: void
        """

        self.unlock_weights()  # when reset the weight, also need to unlock the weights
        self.weights = np.zeros((4, self.height*self.width + 1))

    def lock_weights(self):
        """
        Freeze the weights
        :return: void
        """

        self.lock = True

    def unlock_weights(self):
        """
        Unlock the weights
        :return: void
        """

        self.lock = False

    def process(self, img):
        """
        Process image using sequence of filters
        :param img: input image can be uint8
        :return: Processed image in float32
        """

        # TODO: set this method to private in the future
        if img.dtype != np.uint8:
            raise Exception("Input Image need to be unit8 but was {}".format(img.dtype))

        img = (img/255).astype(np.float32)
        sub_patch = img[self.x1:self.x2, self.y1:self.y2]
        for processor in self.chromosome:
            sub_patch = processor.transform(sub_patch)

        return sub_patch

    def train_perceptron(self, img: np.ndarray, label, lr=1):
        """
        Train the perceptron for one entry.
        Also for validate the perceptron
        :param img: input processed image, no need to be flattened
        :param label: ground truth label
        :param lr: learning rate
        :return:
        """

        # apply filters
        subpatch = self.process(img)

        # check input
        if subpatch.shape != (self.height, self.width):
            print("Shape does not match. Shape should be {} but was {}".format((self.height, self.width),
                                                                               subpatch.shape))

        # prepare the arrays
        bias = np.ones(1)
        arr = subpatch.flatten()
        arr = np.hstack((arr, bias))  # need to append a bias node

        # compute scores and prediction
        scores = self.weights @ arr
        predicted_cat = np.argmax(scores)

        # update weights if the prediction is wrong
        # if the weights are locked, this become validation
        if not self.lock and label != predicted_cat:
            sign = np.zeros((4, 1))  # no update for the true negatives
            sign[predicted_cat] = -1  # punish the false positive
            sign[label] = 1  # increase the weight for false negative
            update = np.vstack((arr, arr, arr, arr)) * sign
            self.weights = self.weights + lr * update  # do not normalize the weight

        # update the confusion matrix
        self.confusion[label, predicted_cat] += 1

    def compute_fitness(self):
        """
        Validate on a holding image set and get the compute_fitness score
        :return: the compute_fitness score
        """

        # count the true positive, true negative, false positive and false negative
        # in format of size n (num of cats) 1d array
        tp = self.confusion.diagonal()  # tp on diagonal
        fp = np.sum(self.confusion, axis=0) - tp  # fp are each column sum minus the tp
        fn = np.sum(self.confusion, axis=1) - tp  # fn are each row sum minus the tp
        tn = - 1 * (tp + fp + fn) + self.confusion.sum()  # tn are the sum of the rest grids

        # precision
        precision = tp/(fn + tp)

        # true negative rate (need to be normalized by # of cats - 1)
        tn_norm = tn/3
        tn_rate = tn_norm/(fp + tn_norm)

        # score need to be normalized by # of cat
        self.fitness_score = np.sum(precision + tn_rate) * 125  # score ranged from 0 to 1000

    def info(self):
        """
        Return the creature information including:
        patch coordinates
        gene code and parameters
        perceptron weights
        compute_fitness score
        :return: a list of string that describe the creature
        """
        info = {"patch": (self.x1, self.x2, self.y1, self.y2), "genes": {x.code: x.params for x in self.chromosome},
                "weights": self.weights, "compute_fitness": self.fitness_score}
        return info

    def create_tfm(self, gene):
        """
        Helper method to get the transformer
        :param gene: gene code
        :return: transformer with random parameters
        """

        # TODO: include several more tfms
        if gene == 1:
            x = tfm.AdaptiveThreshold(width=self.width, height=self.height)
        elif gene == 2:
            x = tfm.CannyEdge(width=self.width, height=self.height)
        elif gene == 3:
            x = tfm.CensusTransformation(width=self.width, height=self.height)
        elif gene == 4:
            x = tfm.CLAHistogram(width=self.width, height=self.height)
        elif gene == 5:
            x = tfm.DifferenceGaussian(width=self.width, height=self.height)
        elif gene == 6:
            x = tfm.Dilate(width=self.width, height=self.height)
        elif gene == 7:
            x = tfm.DistanceTransformation(width=self.width, height=self.height)
        elif gene == 8:
            x = tfm.Erode(width=self.width, height=self.height)
        elif gene == 9:
            x = tfm.GaussianBlur(width=self.width, height=self.height)
        elif gene == 10:
            x = tfm.Gradient(width=self.width, height=self.height)
        elif gene == 11:
            x = tfm.HarrisCorner(width=self.width, height=self.height)
        elif gene == 12:
            x = tfm.HistogramEqualization(width=self.width, height=self.height)
        elif gene == 13:
            x = tfm.IntegralTransformation(width=self.width, height=self.height)
        elif gene == 14:
            x = tfm.LaplacianEdge(width=self.width, height=self.height)
        elif gene == 15:
            x = tfm.Log(width=self.width, height=self.height)
        elif gene == 16:
            x = tfm.MediumBlur(width=self.width, height=self.height)
        elif gene == 17:
            x = tfm.SquareRoot(width=self.width, height=self.height)
        else:
            raise Exception("Invalid Gene Number {}".format(gene))
        return x


class PopulationOperator:
    """
    Operators for handling population activities such as select, cross, mutate.
    Pipeline:
    1 - new population
    2 - train population -> weights will be locked after the training
    3 - validate population -> compute the fitness
    4 - report -> report the population
    5 - eliminate population -> all underperformed creatures will be removed from population
    6 - update report? -> add one line of survived creatures?
    7 - reproduce ->  generate a new generation via a series of select -> cross -> mutate
    8 - reset_population_weights
    8 - ... (go to step 2)
    """

    @staticmethod
    def new_population(num, img_shape):
        """
        Initialize a new population
        :param num: number of creatures for the generation
        :param img_shape: image shape
        """

        # generate a population of creatures
        population = []
        for i in range(num):
            population.append(Creature(img_shape=img_shape))
        return population

    @staticmethod
    def train_population(population: list, train_data: list, src_dir: str, epoch_num=100, lr=1):
        """
        Train the population.
        # TODO: in the future, implement an epoch report, show avg confusion matrix stats
        # TODO: in the future, store the midway results in case the training get interrupted
        :param population: population
        :param train_data: list of the training data in tuple format [(id, cat)]. id and cat must be int
        :param src_dir: the directory for placing the training images
        :param epoch_num: number of iterations for the training the perceptron
        :param lr: learning rate
        :return: void
        """

        # reset the weights
        for creature in population:
            creature.reset_weights()

        # train perceptron for each creature
        for i in range(epoch_num):
            # reset the confusion matrix for each creature at the beginning of each training epoch
            # TODO: confusion matrix does not need to be updated/reset during the training period
            # TODO: or it does need to be reset to provide report on per training iteration
            PopulationOperator.reset_population_confusion(population)

            # train on each data entry
            for entry in train_data:
                img_id = entry[0]
                label = entry[1]
                img_path = os.path.join(src_dir, "{}.png".format(img_id))
                img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY)
                for creature in population:
                    # TODO: in the future implement more classifier e.g. SVM, KNN etc
                    creature.train_perceptron(img=img, label=label, lr=lr)  # train perceptron

        # when training is done, lock the weights of each creature
        for creature in population:
            creature.lock_weights()

    @staticmethod
    def reset_population_weights(population):
        """
        reset the weights for the creatures in a population
        also unlock the weights
        :param population:
        :return:
        """

        for creature in population:
            creature.reset_weights()

    @staticmethod
    def validate_population(population: list, hol_data: list, src_dir: str):
        """
        Validate the creatures on hold out data set and compute the compute_fitness score.
        :param population: population
        :param hol_data: list of the holdout data in tuple format [(id, cat)]. id and cat must be int
        :param src_dir: the directory for placing the holdout images
        :return: void
        """

        # reset the confusion matrix for each creature at the beginning of each training epoch
        # TODO: confusion matrix does not need to be updated/reset during the training period
        # TODO: or it does need to be reset to provide report on per training iteration
        PopulationOperator.reset_population_confusion(population)

        # validate on the holding data
        for entry in hol_data:
            img_id = entry[0]
            label = entry[1]
            img_path = os.path.join(src_dir, "{}.png".format(img_id))
            img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY)
            for creature in population:
                # TODO: in the future implement more classifier e.g. SVM, KNN etc
                creature.train_perceptron(img=img, label=label, lr=1)  # validate perceptron

        # compute the compute_fitness scores for creatures in population
        for creature in population:
            creature.compute_fitness()

    @staticmethod
    def eliminate_population(population: list, t=0.25):
        """
        # TODO: need to decide whether use hard threshold or the flexible threshold (i.e. eliminate lowest 25%)
        # TODO: hard threshold could prevent some bad gene contamination
        # TODO: for now, eliminate lowest 25%.
        Eliminate the underperformed creatures in place.
        Due to the current fitness score set up, the validating data need to be balanced, otherwise it could be bias
        to the dominate cat.
        :param population: population of creatures
        :param t: threshold
        :return:
        """

        # remove the last
        population.sort(key=lambda x: x.fitness_score, reverse=True)  # O(nlgn)
        n = int(len(population) * t)
        for i in range(n):
            population.pop()  # pop last item is O(1)

    @staticmethod
    def reproduce(parents_pool: list, num, cross_rate=0.9, mutate_rate=0.0005):
        """
        Reproduce a new generation
        :param parents_pool: parent pool
        :param num: number of creatures in new generation
        :param cross_rate: cross rate, default 0.9
        :param mutate_rate: mutate rate default 0.0005
        :return: list of new generation of creatures
        """

        offsprings = []
        for i in range(num):
            p1, p2 = PopulationOperator.tournament(parents_pool)
            child = PopulationOperator.cross(parents_pool[p1], parents_pool[p2], cross_rate, mutate_rate)
            offsprings.append(child)
        return offsprings

    @staticmethod
    def tournament(population: list, p=0.75):
        """
        Tournament selection method
        :param population: creature population
        :param p: chance that the fitter creature is selected
        :return: the parents indexes in tuple
        """

        p_idx = [-1, -1]
        n = len(population)

        # find two parents indexes
        for i in range(2):
            idx1 = random.randrange(n)
            idx2 = random.randrange(n)
            fit1 = population[idx1].fitness_score
            fit2 = population[idx2].fitness_score

            # Tournament
            if np.random.choice(2, 1, p=[1-p, p]) == 1:
                p_idx[i] = idx1 if fit1 > fit2 else idx2
            else:
                p_idx[i] = idx1 if fit1 < fit2 else idx2

        return p_idx[0], p_idx[1]

    @staticmethod
    def cross(parent1: Creature, parent2: Creature, cross_rate, mutate_rate):
        """
        Cross operation. It is possible to create children that is longer than 8.
        # TODO: if the creature is too long, it might not be a good thing.
        # TODO: because it is possible that there are children with duplicated gene (could be a bad thing)
        :param parent1: first parent
        :param parent2: secondary parent
        :param cross_rate: cross over rate
        :param mutate_rate: mutate rate
        :return:
        """

        # if cross does not happen, take the first random parent to the next generation
        # child is a new creature object but with shared tmf objects in chromosome lis
        n1, n2 = len(parent1.chromosome), len(parent2.chromosome)
        child = parent1.give_birth()

        if np.random.choice(2, 1, p=[1-cross_rate, cross_rate]) == 1:
            # first parent for the first half, second parent for the remaining half slice ~ [0, n-1]
            # since slice ~ [0, n] -> range(0) ~ range(n) at least pop 0 out, at most pop n-1 out
            # this guarantee that the children at least have one gene from the parents
            slice1, slice2 = random.randrange(n1), random.randrange(n2)

            # pop the gene after cp1 from the first parent
            for i in range(slice1):
                child.chromosome.pop()

            # pop the gene after cp2 from parent2 then append to first parent
            for i in range(slice2):
                child.chromosome.append(parent2.chromosome.pop())  # O(1) for pop the last item

        # children's weights shape should match with patch shape, all weights are 0 and unlocked.
        child.mutate(r=mutate_rate)
        return child

    @staticmethod
    def reset_population_confusion(population):
        """
        Reset the confusion matrix of the population
        :param population:
        :return:
        """

        # TODO: confusion matrix does not need to be updated/reset during the training period
        for creature in population:
            creature.reset_confusion()

    @staticmethod
    def output_model(population):
        # TODO: out put the whole model: creature gene, params, weights, fitness score, confusion matrix
        # TODO: maybe the weights and confusion matrix in separated .npy file
        pass

    @staticmethod
    def report(population):
        # TODO: report the validation results?
        pass


a = Creature(img_shape=(49, 49))
a.build_chromosome()
b = a.give_birth()

print(a.chromosome)
print(b.chromosome)
