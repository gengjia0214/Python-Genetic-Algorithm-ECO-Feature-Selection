from src import transformer as tfm
from sklearn.linear_model import LogisticRegression as Logit
from sklearn.svm import LinearSVC as SVm
from sklearn.linear_model import Perceptron as Pct
from enum import Enum
import numpy as np
import cv2 as cv
import random
import warnings
import json
import copy
import os

warnings.filterwarnings("ignore")

"""
@author: Jia Geng
@email: jxg570@miami.edu
 
Dev Log
10/17:  Implemented the Creature Class (still need to implement the perceptron)
10/19:  Implemented the perceptron classifier and the confusion matrix, fitness score
10/20:  Fixed some bugs. The fitness can be overwhelmed by imbalanced data. 
        Up-sampling (collect more samples or generate simulated data) or increase the weight of NECR for the fitness 
        score
10/21:  Implemented the population operator.
10/23:  Added mutation function for creature patch coors; Added reproduce related functions.
10/24:  Use deep copy instead of shallow copy for cross operation. Implement process report bar.
10/24:  Seems like just use the edge will very easily converge during the training param = [0.5984, 1] (make sense 
cause it is linear separable) but it does not perform well for validating set.
10/31:  Fixed a bug. Now the child will have at least 1 gene from parent 1 and ata most all gene from the 2 parents.
10/31:  Refactored the train/validate and lock strategy. Perceptron now using early stopping.
11/1:   Added a method to save & load the params into/from json file
11/4:   Refactored the method. Now is compatible for any defined number of classes
11/11:  Changed the image process pipeline from img -> subpatch -> processed subpatch to img -> processed image -> 
subpatch to alleviate the potential kernel oversize issue
11/16:  Refactored method with Enum
11/21:  Refactored the perceptron, now use sklearn
11/22:  Added a random gauss small number to the fitness score to prevent unstable behaviour
"""


class Elimination(Enum):
    """
    ENUMs for elimination
    """
    OVERALL = 18
    BY_CAT = 19


class Model(Enum):
    """
    ENUMS for model selection
    """

    PERCEPTRON_MUL = 0
    PERCEPTRON_HIER = 1

    LOGIT_MUL = 2
    LOGIT_HIER = 3

    SVM_MUL = 4
    SVM_HIER = 5


class Creature:
    """
    Creature - the basic classifier that taking random set of feature extractor and a random cropped subpatch of an
    image
    """

    def __init__(self, img_shape: tuple, creature_id, num_cat=4):
        """
        Constructor.
        Create a creature with random patch window and empty chromosome
        :param img_shape: original image size (height x weight)
        :param creature_id: creature id
        :param num_cat: number of classes
        """

        self.id = creature_id
        self.num_cat = num_cat
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
        self.subpath_height, self.subpatch_width = self.x2 - self.x1, self.y2 - self.y1
        self.weights = np.zeros((self.num_cat, self.subpath_height * self.subpatch_width + 1))
        self.chromosome = []
        self.fitness_score = {'avg': 0}
        for i in range(self.num_cat):
            self.fitness_score[str(i)] = 0
        self.confusion = np.zeros((self.num_cat, self.num_cat), dtype=np.int16)
        self.lock = False

    def build_chromosome(self, gene_pool_size=17, length_limit=8):
        """
        Build the chromosome with random length
        :return:
        """

        n = random.randrange(1, length_limit + 1)  # n is the chromosome length
        gene_seq = random.sample(range(1, gene_pool_size + 1), n)  # randomly take n gene from the pool with
        # replacement

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
        self.subpath_height, self.subpatch_width = self.x2 - self.x1, self.y2 - self.y1

        # need to reset the weights after mutate
        self.reset_weights()

        # mutate the gene
        for gene in self.chromosome:
            gene.mutate(r=r)

    def reset_confusion(self):
        """
        Reset the confusion matrix and the weights
        :return:
        """
        self.confusion = np.zeros((self.num_cat, self.num_cat), dtype=np.int16)

    def reset_weights(self):
        """
        Reset the weights
        :return: void
        """

        self.unlock_weights()  # when reset the weight, also need to unlock the weights
        self.weights = np.zeros((self.num_cat, self.subpath_height * self.subpatch_width + 1))

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

    def __process(self, img):
        """
        Process image using sequence of filters
        :param img: input image (n_row, n_col) need to be uint8
        :return: Processed cropped sub_images in flattened array (n_pixels, )
        """
        img_f = cv.normalize(img, None, 1e-8, 1.0, cv.NORM_MINMAX, cv.CV_64F)
        for gene in self.chromosome:
            img_f = gene.transform(img_f)
        sub_patch = img_f[self.x1:self.x2, self.y1:self.y2]
        arr = sub_patch.flatten()
        return arr

    def __imgs2X(self, imgs):
        """
        Method to apply feature extraction on a batch of images
        :param imgs: imgs
        :return: features: (n_sample, n_subrow * n_subcol)
        """

        X = np.empty((imgs.shape[0], self.subpath_height * self.subpatch_width))
        for i in range(imgs.shape[0]):
            X[i, :] = self.__process(imgs[i])
        return X

    def train_perceptron_multi(self, imgs: np.ndarray, y: np.ndarray, args: dict):
        """
        Train the logistic regression. After the training, this method will save the parameter to the creature
        :param imgs: training img need to be flattened and v-stacked to (n_entries, n_row, n_col)
        :param y: labels need to be h-stacked to (n_labels, )
        :param args: args to be passed to the logistic model fit method
        :return: void
        """
        if not self.lock:
            # prepare the args
            p_args = {'penalty': None, 'alpha': 1e-4, 'max_iter': 100, 'n_jobs': -1, 'random_state': None}
            if args is not None:
                for arg, value in args.items():
                    if arg in p_args:
                        p_args[arg] = value
                    else:
                        raise Exception("[{}] is not a supported arg for sklearn.Perceptron.".format(arg))
            X = self.__imgs2X(imgs)

            # fit model
            pct_model = Pct(penalty=p_args['penalty'], alpha=p_args['alpha'], max_iter=p_args['max_iter'],
                            n_jobs=p_args['n_jobs'], random_state=p_args['random_state'])
            pct_model.fit(X, y)

            # save the parameters
            feature_weights = pct_model.coef_  # (n_class, n_feature)
            intercept_weight = pct_model.intercept_  # (n_class, )
            weights = np.hstack((intercept_weight[:, np.newaxis], feature_weights))
            self.weights = weights
            self.lock_weights()

    def train_logit_multi(self, imgs: np.ndarray, y: np.ndarray, args: dict):
        """
        Train the logistic regression. After the training, this method will save the parameter to the creature
        :param imgs: training img need to be flattened and v-stacked to (n_entries, n_row, n_col)
        :param y: labels need to be h-stacked to (n_labels, )
        :param args: args to be passed to the logistic model fit method
        :return: void
        """

        if not self.lock:
            # prepare the args
            p_args = {'penalty': None, 'solver': 'saga', 'max_iter': 100, 'n_jobs': -1, 'random_state': None,
                      'multi_class': 'multinomial', "C": 1.0, 'dual': False}
            if args is not None:
                for arg, value in args.items():
                    if arg in p_args:
                        p_args[arg] = value
                    else:
                        raise Exception("[{}] is not a supported arg for sklearn Logit Regression.".format(arg))
            X = self.__imgs2X(imgs)

            # fit model
            lg_model = Logit(penalty=p_args['penalty'], random_state=p_args['random_state'], C=p_args['C'],
                             solver=p_args['solver'], multi_class=p_args['multi_class'], n_jobs=p_args['n_jobs'],
                             dual=p_args['dual'], max_iter=p_args['max_iter'])
            lg_model.fit(X, y)

            # save the parameters
            feature_weights = lg_model.coef_  # (n_class, n_feature)
            intercept_weight = lg_model.intercept_  # (n_class, )
            weights = np.hstack((intercept_weight[:, np.newaxis], feature_weights))
            self.weights = weights
            self.lock_weights()

    def train_svm_multi(self, imgs: np.ndarray, y: np.ndarray, args: dict):
        """
        Train the logistic regression. After the training, this method will save the parameter to the creature
        :param imgs: training img need to be flattened and v-stacked to (n_entries, n_row, n_col)
        :param y: labels need to be h-stacked to (n_labels, )
        :param args: args to be passed to the logistic model fit method
        :return: void
        """

        if not self.lock:
            # prepare the args
            p_args = {'penalty': 'l2', 'loss': 'hinge', 'max_iter': 1000, 'random_state': None, 'multi_class': 'ovr',
                      "C": 1.0, 'dual': True}
            if args is not None:
                for arg, value in args.items():
                    if arg in p_args:
                        p_args[arg] = value
                    else:
                        raise Exception("[{}] is not a supported arg for sklearn linear SVM.".format(arg))
            X = self.__imgs2X(imgs)

            # fit model
            # fit model
            svm_model = SVm(penalty=p_args['penalty'], random_state=p_args['random_state'], C=p_args['C'],
                            loss=p_args['loss'], multi_class=p_args['multi_class'], dual=p_args['dual'],
                            max_iter=p_args['max_iter'])
            svm_model.fit(X, y)

            # save the parameters
            # save the parameters
            feature_weights = svm_model.coef_  # (n_class, n_feature)
            intercept_weight = svm_model.intercept_  # (n_class, )
            weights = np.hstack((intercept_weight[:, np.newaxis], feature_weights))
            self.weights = weights
            self.lock_weights()

    def train_svm_hierarchy(self, img: np.ndarray, label, lr=1):
        pass

    def train_perceptron_hierarchy(self, img: np.ndarray, label, lr=1):
        pass

    def train_logit_hierarchy(self, img: np.ndarray, label, lr=1):
        pass

    def model_validation(self, imgs: np.ndarray, y: np.ndarray, mode: Model):
        """
        Validate the perceptron
        :param imgs: validation data (n_samples, n_row, n_col)
        :param y: truth
        :param mode: model mode
        :return: void
        """

        # must lock the weights before validate
        if not self.lock:
            raise Exception("Weight need to be locked before validation!")

        # make prediction

        predictions = self.predict(imgs, mode=mode)

        # update the confusion matrix
        self.reset_confusion()
        for i, predicated_cat in enumerate(predictions):
            self.confusion[y[i], predicated_cat] += 1

    def predict(self, imgs: np.ndarray, mode: Model):
        """
        Validate the perceptron on a img
        :param imgs: input images in (n_sample, n_row, n_col)
        :param mode: mode for prediction
        :return: predict cat
        """

        # compute scores and prediction

        X = self.__imgs2X(imgs)
        if mode == Model.PERCEPTRON_MUL:
            pct = Pct()
            pct.intercept_ = self.weights[:, 0]
            pct.coef_ = self.weights[:, 1:]
            pct.classes_ = np.arange(4).astype(np.int8)
            return pct.predict(X).astype(np.int8)
        elif mode == Model.LOGIT_MUL:
            logit = Logit()
            logit.intercept_ = self.weights[:, 0]
            logit.coef_ = self.weights[:, 1:]
            logit.classes_ = np.arange(4).astype(np.int8)
            return logit.predict(X).astype(np.int8)
        elif mode == Model.SVM_MUL:
            svm = SVm()
            svm.intercept_ = self.weights[:, 0]
            svm.coef_ = self.weights[:, 1:]
            svm.classes_ = np.arange(4).astype(np.int8)
            return svm.predict(X).astype(np.int8)
        elif mode == Model.PERCEPTRON_HIER:
            pass
        elif mode == Model.LOGIT_HIER:
            pass
        elif mode == Model.SVM_HIER:
            pass

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
        precision = tp / (fn + tp)

        # true negative rate (need to be normalized by # of cats - 1)
        tn_norm = tn / (self.num_cat - 1)
        tn_rate = tn_norm / (fp + tn_norm)

        # fitness by cat
        fitness_score_by_cat = (precision + tn_rate) * 500
        for i in range(fitness_score_by_cat.shape[0]):
            self.fitness_score[str(i)] = fitness_score_by_cat[i]

        # overall score need to be normalized by # of cat
        self.fitness_score['avg'] = np.sum(precision + tn_rate) * 500 / self.num_cat  # score ranged from 0 to 1000
        for key in self.fitness_score:
            self.fitness_score[key] += random.gauss(mu=1e-6, sigma=1e-7)

    def info(self):
        """
        Return the creature information including:
        patch coordinates
        gene code and parameters
        perceptron weights
        compute_fitness score
        :return: a list of string that describe the creature
        """

        img_shape = self.img_shape
        patch = (self.x1, self.x2, self.y1, self.y2)
        height, width = self.subpath_height, self.subpatch_width
        genes = [{'code': x.code, 'params': x.params, 'h': x.height, 'w': x.width} for x in self.chromosome]
        weights = self.weights
        confusion = self.confusion
        fitness_score = self.fitness_score
        info = {'img_shape': img_shape, 'patch': patch, 'height': height, 'width': width, 'genes': genes, 'weights':
                weights.tolist(), 'confusion': confusion.tolist(), 'fitness': fitness_score}

        return info

    def create_tfm(self, gene):
        """
        Helper method to get the transformer
        :param gene: gene code
        :return: transformer with random parameters
        """

        return tfm.Transformer.get_tfm(img_height=self.img_shape[0], img_width=self.img_shape[1], gene=gene)

    def depreciated_train_perceptron_multi(self, imgs: np.ndarray, y: np.ndarray, args: dict, silence=True):
        """
        Train the logistic regression. After the training, this method will save the parameter to the creature
        :param imgs: training imgs (n_samples, n_row, n_col), should be in uint8
        :param y: labels need to be h-stacked to (n_labels, )
        :param args: args lr: learning rate, theta: stopping criteria
        :param silence: whether report epoch
        :return: void
        """

        # prepare args

        n_samples = imgs.shape[0]
        error_prev, error_curr = len(imgs), 0
        lr = args['lr']
        theta = args['theta']

        # training
        for epoch in range(100):
            # reset the confusion matrix for each creature at the beginning of each training epoch
            self.reset_confusion()
            randomize = np.arange(n_samples)
            np.random.shuffle(randomize)
            imgs = imgs[randomize]
            y = y[randomize]

            for i in range(imgs.shape[0]):
                img = imgs[i]
                label = y[i]
                # make prediction
                bias = np.ones((1,))
                arr = np.hstack((bias, self.__process(img)))
                predicted_cat = self.predict(arr[np.newaxis, :], mode=Model.PERCEPTRON_MUL)
                self.confusion[label, predicted_cat] += 1
                # update weights if the prediction is wrong
                # if the weights are locked, this become validation
                if label != predicted_cat:
                    sign = np.zeros((self.num_cat, 1))  # no update for the true negatives
                    sign[predicted_cat] = -1  # punish the false positive
                    sign[label] = 1  # increase the weight for false negative
                    update = np.vstack(tuple([arr for i in range(self.num_cat)])) * sign
                    self.weights = self.weights + lr * update  # do not normalize the weight

            # converge detection
            error_curr = n_samples - np.sum(self.confusion.diagonal())
            # intuitively, theta is how many improvement do you need at least for each epoch training
            t = abs(error_curr - error_prev) * 2 / ((error_prev + error_curr) + 1e-8)

            if t < theta and epoch > 3:
                err = round(error_curr / n_samples, 4)
                self.lock_weights()
                if not silence:
                    print("Creature {} locked at epoch {}. Error rate {}".format(self.id, epoch, err))
                break
            error_prev = error_curr
        self.lock_weights()


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
    7 - reproduce ->  generate a new generation via a series of select -> cross -> mutate (also reset weights)
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
            creature = Creature(img_shape=img_shape, creature_id=i)
            creature.build_chromosome()
            population.append(creature)
        return population

    @staticmethod
    def train_population(population: list, train_data: list, src_dir: str, mode: Model, args: dict, silence=False):
        """
        Train the population.
        :param population: population
        :param train_data: list of the training data in tuple format [(id, cat)]. id and cat must be int
        :param src_dir: the directory for placing the training images
        :param args: args for training methods
        :param silence: whether report the training stats
        :param mode: model mode
        :return: void
        """

        # reset the weights
        for creature in population:
            creature.reset_weights()

        # prepare the training data
        n_samples = len(train_data)
        n_row, n_col = population[0].img_shape
        imgs = np.empty((n_samples, n_row, n_col))
        y = np.full((n_samples,), fill_value=-1, dtype=np.int8)
        for j in range(len(train_data)):
            img_id = train_data[j][0]
            img_path = os.path.join(src_dir, "{}.png".format(img_id))
            img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY)
            imgs[j, :, :] = img
            y[j] = train_data[j][1]

        # train each creature
        for creature in population:
            # train perceptron pass if locked
            if not silence:
                print("Training Creature {}".format(creature.id))
            if creature.lock:
                continue
            else:
                # epoch
                if mode == Model.PERCEPTRON_MUL:
                    creature.train_perceptron_multi(imgs=imgs, y=y, args=args)
                elif mode == Model.LOGIT_MUL:
                    creature.train_logit_multi(imgs=imgs, y=y, args=args)
                elif mode == Model.SVM_MUL:
                    creature.train_svm_multi(imgs=imgs, y=y, args=args)
                elif mode == Model.PERCEPTRON_HIER:
                    pass
                elif mode == Model.LOGIT_HIER:
                    pass
                elif mode == Model.SVM_HIER:
                    pass
        print("All creatures trained, weights locked.")
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
    def validate_population(population: list, hol_data: list, src_dir: str, mode: Model):
        """
        Validate the creatures on hold out data set and compute the compute_fitness score.
        :param population: population
        :param hol_data: list of the holdout data in tuple format [(id, cat)]. id and cat must be int
        :param src_dir: the directory for placing the holdout images
        :param mode: model mode
        :return: void
        """

        # reset the confusion matrix for each creature at the beginning of each training epoch
        PopulationOperator.reset_population_confusion(population)

        # validate on the holding data
        # prepare the data
        n_samples = len(hol_data)
        n_row, n_col = population[0].img_shape
        imgs = np.empty((n_samples, n_row, n_col))
        y = np.full((n_samples,), fill_value=-1, dtype=np.int8)
        for i in range(n_samples):
            img_id = hol_data[i][0]
            label = hol_data[i][1]
            img_path = os.path.join(src_dir, "{}.png".format(img_id))
            img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY)
            imgs[i, :, :] = img
            y[i] = int(label)

        # validate creatures
        for creature in population:
            creature.model_validation(imgs=imgs, y=y, mode=mode)  # validate perceptron

        # compute the compute_fitness scores for creatures in population
        for creature in population:
            creature.compute_fitness()
        print("Validation done.")
        PopulationOperator.report_population(population)

    @staticmethod
    def eliminate_population(population: list, mode: Elimination, t=0.25):
        """
        Eliminate the underperformed creatures in place.
        Due to the current fitness score set up, the validating data need to be balanced, otherwise it could be bias
        to the dominate cat.
        :param population: population of creatures
        :param mode: OVERALL: eliminate overall bad performer by threshold; BY_CAT: for each class, keep top (1 - sqrt(
        sqrt(t))).
        :param t: threshold
        :return: void
        """

        # num_cat
        num_cat = population[0].num_cat

        # remove the last
        if mode == Elimination.OVERALL:
            population.sort(key=lambda x: x.fitness_score['avg'], reverse=True)  # O(nlgn)
            n = int(len(population) * t)
            for i in range(n):
                population.pop()  # pop last item is O(1)
        elif mode == Elimination.BY_CAT:
            p = np.sqrt(np.sqrt(t))
            keep_num = int(round(len(population) * (1 - p)))
            collector = []
            for i in range(num_cat):
                cat = str(i)
                population.sort(key=lambda x: x.fitness_score[cat], reverse=True)
                for creature in population[:keep_num + 1]:
                    if creature not in collector:
                        collector.append(creature)
            population[:] = collector
        else:
            raise Exception("Invalid input {}".format(mode))

        print("Eliminated bad performers, now have {} in pool".format(len(population)))

    @staticmethod
    def reproduce(parents_pool: list, num, cross_rate=0.9, mutate_rate=0.0005):
        """
        Reproduce a new generation
        :param parents_pool: parent pool
        :param num: number of creatures in new generation
        :param random_gen: random generator

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

        num_cat = population[0].num_cat
        p_idx = [-1, -1]
        n = len(population)

        # find two parents indexes
        for i in range(2):
            idx1 = random.randrange(n)
            idx2 = random.randrange(n)

            # Tournament based on random criteria
            criteria = random.randrange(num_cat + 1)
            if criteria == num_cat:
                criteria = 'avg'
            else:
                criteria = str(criteria)
            fit1 = population[idx1].fitness_score[criteria]
            fit2 = population[idx2].fitness_score[criteria]
            if random.choices(population=[0, 1], weights=[1-p, p]) == 1:
                p_idx[i] = idx1 if fit1 > fit2 else idx2
            else:
                p_idx[i] = idx1 if fit1 <= fit2 else idx2

        return p_idx[0], p_idx[1]

    @staticmethod
    def cross(parent1: Creature, parent2: Creature, cross_rate, mutate_rate):
        """
        Cross operation. It is possible to create children that is longer than 8.
        Usually a long creature will be more likely to be eliminated because it tends to perform bad (lost too much
        information)
        :param parent1: first parent
        :param parent2: secondary parent
        :param cross_rate: cross over rate
        :param mutate_rate: mutate rate
        :return:
        """

        # if cross does not happen, take the first random parent to the next generation
        # child is a new creature object but with shared tmf objects in chromosome lis

        n1, n2 = len(parent1.chromosome), len(parent2.chromosome)
        child = copy.deepcopy(parent1)  # deep copy

        if random.choices(population=[0, 1], weights=[1 - cross_rate, cross_rate]) == 1:
            # first parent for the first half, second parent for the remaining half
            # need to make sure at least one element from each parent
            # at most get all the elements from both parents
            slice1 = random.randrange(n1)  # slice1 ~ [0, n1-1]
            for i in range(slice1):  # pop out [0 ~ n1-1] number of elements
                child.chromosome.pop()

            # append gene from parent2 to the child
            slice2 = random.randrange(n2 + 1)  # slice2 ~ [0, n2]
            if slice2 != 0:
                child.chromosome = child.chromosome + copy.deepcopy(parent2.chromosome[-slice2:])

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
        for creature in population:
            creature.reset_confusion()

    @staticmethod
    def save_population(population, dst_file=None):
        meta_data = []
        for creature in population:
            meta_data.append(creature.info())
        if dst_file:
            with open(dst_file, 'w') as f:
                json.dump(meta_data, f)
            print("Model params saved at {}".format(dst_file))
        else:
            return meta_data

    @staticmethod
    def load_population(src):
        """
        Load a population from a json file
        :param src: input src, either a list or a json path
        :return:
        """

        loaded_population = []

        if type(src) == str:
            with open(src, 'r') as f:
                meta_data = json.load(f)
        elif type(src) == list:
            meta_data = src
        else:
            raise Exception("Input source need to be either list or json file path but was {}".format(type(src)))

        for i, creature_info in enumerate(meta_data):
            creature = Creature(img_shape=(10, 10), creature_id=i)
            creature.img_shape = tuple(creature_info['img_shape'])
            creature.x1, creature.x2, creature.y1, creature.y2 = creature_info['patch']
            creature.subpath_height = creature_info['height']
            creature.subpatch_width = creature_info['width']
            h, w = creature.img_shape[0], creature.img_shape[1]
            for gene in creature_info['genes']:
                code = gene['code']
                params = gene['params']
                tfm_x = tfm.Transformer.get_tfm(img_width=w, img_height=h, gene=code)
                tfm_x.params = params
                tfm_x.code = code
                creature.chromosome.append(tfm_x)
            creature.weights = np.asarray(creature_info['weights'])
            creature.confusion = np.asarray(creature_info['confusion'])
            creature.fitness_score = creature_info['fitness']
            loaded_population.append(creature)
        print("All {} creatures loaded.".format(len(meta_data)))
        return loaded_population

    @staticmethod
    def report_creature(creature: Creature):
        """
        Report the best precision, recall from a population
        :param creature:
        :return:
        """

        num_cat = creature.num_cat
        tp = creature.confusion.diagonal()  # tp on diagonal
        fp = np.sum(creature.confusion, axis=0) - tp  # fp are each column sum minus the tp
        fn = np.sum(creature.confusion, axis=1) - tp  # fn are each row sum minus the tp
        precision = np.round(tp / (fp + tp + 0.001), 4)  # in case the nan value
        recall = np.round(tp / (tp + fn + 0.001),  4)  # in case the nan value
        error = np.round(tp.sum() / creature.confusion.sum(), 4)
        line1 = "{:>10}".format("Class")
        line2 = "{:>10}".format("Precision")
        line3 = "{:>10}".format("Recall")
        for i in range(num_cat):
            line1 += "{:>10}".format(i)
            line2 += "{:>10}".format(precision[i])
            line3 += "{:>10}".format(recall[i])
        print(line1)
        print(line2)
        print(line3)
        print("Overall Error Rate {}".format(error))
        print("-------------------------------------------------------\n")

    @staticmethod
    def report_population(population: list):
        """
        Report the best precision, recall from a population
        :param population:
        :return:
        """

        num_cat = population[0].num_cat
        precision = np.zeros((num_cat,), dtype=np.float32)
        recall = np.zeros((num_cat,), dtype=np.float32)
        for creature in population:
            tp = creature.confusion.diagonal()  # tp on diagonal
            fp = np.sum(creature.confusion, axis=0) - tp  # fp are each column sum minus the tp
            fn = np.sum(creature.confusion, axis=1) - tp  # fn are each row sum minus the tp
            temp_precision = np.round(tp / (fp + tp + 0.001), num_cat)  # in case the nan value
            temp_recall = np.round(tp / (tp + fn + 0.001),  num_cat)  # in case the nan value
            precision = np.maximum(temp_precision, precision)
            recall = np.maximum(temp_recall, recall)
        line1 = "{:>10}".format("Class")
        line2 = "{:>10}".format("Precision")
        line3 = "{:>10}".format("Recall")
        for i in range(num_cat):
            line1 += "{:>10}".format(i)
            line2 += "{:>10}".format(precision[i])
            line3 += "{:>10}".format(recall[i])
        print(line1)
        print(line2)
        print(line3)
        print("-------------------------------------------------------\n")

