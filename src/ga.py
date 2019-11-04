from src.models.eco import transformer as tfm
import numpy as np
import cv2 as cv
import random
import json
import copy
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
10/24:  Use deep copy instead of shallow copy for cross operation. Implement process report bar.
10/24:  Seems like just use the edge will very easily converge during the training param = [0.5984, 1] (make sense 
cause it is linear separable) but it does not perform well for validating set.
10/31:  Fixed a bug. Now the child will have at least 1 gene from parent 1 and ata most all gene from the 2 parents.
10/31:  Refactored the train/validate and lock strategy. Perceptron now using early stopping.
11/1:   Added a method to save & load the params into/from json file
"""


# TODO: after the crossing or the mutate of the crop, the kernel size might not suit the sub-patch size any more.
#  will this cause problem? Maybe add a checking, if the it is oversize, then regenerate the params, or use the
#  largest kernel size.
# TODO: Implement the perceptron binary strategy (low priority)

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
        self.fitness_score = {'0': 0, '1': 0, '2': 0, '3': 0, 'avg': 0}
        self.confusion = np.zeros((4, 4), dtype=np.int16)
        self.lock = False

    def build_chromosome(self, gene_pool_size=17, length_limit=8):
        """
        Build the chromosome with random length
        :return:
        """

        n = random.randrange(1, length_limit + 1)  # n is the chromosome length
        gene_seq = random.sample(range(1, gene_pool_size + 1), n)  # randomly take n gene from the pool

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

        # need to reset the weights after mutate
        self.reset_weights()

        # check the gene's width and height and mutate the gene
        # TODO: Add a checking method to check the kernel size matching
        for gene in self.chromosome:
            gene.width = self.width
            gene.height = self.height
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
        self.weights = np.zeros((4, self.height * self.width + 1))

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

        img = (img / 255).astype(np.float32)
        sub_patch = img[self.x1:self.x2, self.y1:self.y2]
        for processor in self.chromosome:
            sub_patch = processor.transform(sub_patch)

        return sub_patch

    def train_perceptron_binary(self, img: np.ndarray, label, lr=1):
        """
        Train the perceptron for one entry. Using multiple binary perceptron strategy
        Also for validate the perceptron
        :param img: input processed image, no need to be flattened
        :param label: ground truth label
        :param lr: learning rate
        :return: void
        """

        # TODO: implement the binary strategy
        # does not train on locked perceptron
        if self.lock:
            pass

        # make prediction
        predicted_cat, arr = self.predict(img, label)

    def train_perceptron_multiclass(self, img: np.ndarray, label, lr=1):
        """
        Train the perceptron for one entry using multiclass strategy
        Also for validate the perceptron
        :param img: input processed image, no need to be flattened
        :param label: ground truth label
        :param lr: learning rate
        :return: void
        """

        # does not train on locked perceptron
        if self.lock:
            pass

        # make prediction
        predicted_cat, arr = self.predict(img, mode='multiclass_perceptron_train')

        # update the confusion matrix
        self.confusion[label, predicted_cat] += 1

        # update weights if the prediction is wrong
        # if the weights are locked, this become validation
        if label != predicted_cat:
            sign = np.zeros((4, 1))  # no update for the true negatives
            sign[predicted_cat] = -1  # punish the false positive
            sign[label] = 1  # increase the weight for false negative
            update = np.vstack((arr, arr, arr, arr)) * sign
            self.weights = self.weights + lr * update  # do not normalize the weight

    def validate_perceptron(self, img: np.ndarray, label):
        """
        Validate the perceptron
        :param img: input image
        :param label: truth
        :return: void
        """

        # must lock the weights before validate
        if not self.lock:
            raise Exception("Weight need to be locked before validation!")

        # make prediction
        predicted_cat = self.predict(img, label)

        # update the confusion matrix
        self.confusion[label, predicted_cat] += 1

    def predict(self, img: np.ndarray, mode='multiclass_perceptron_predict'):
        """
        Validate the perceptron on a img
        :param img: input image
        :param mode: 'multiclass_perceptron_predict', 'multiclass_perceptron_train', 'binary_perceptron'
        :return: predict cat
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
        if mode == 'multiclass_perceptron_predict' or mode == 'multiclass_perceptron_train':
            scores = self.weights @ arr
            predicted_cat = np.argmax(scores)
        elif mode == 'binary_perceptron':
            score = self.weights[0] @ arr
            if score > 0:
                predicted_cat = 0
            else:
                score = self.weights[1] @ arr
                if score > 0:
                    predicted_cat = 3
                else:
                    score = self.weights[2] @ arr
                    predicted_cat = 1 if score > 0 else 2
        else:
            raise Exception('Mode must be either \'multiclass\' or \'binary\' but was {}'.format(mode))

        if mode == 'multiclass_perceptron_train':
            return predicted_cat, arr
        else:
            return predicted_cat

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
        tn_norm = tn / 3
        tn_rate = tn_norm / (fp + tn_norm)

        # fitness by cat
        fitness_score_by_cat = (precision + tn_rate) * 500
        self.fitness_score['0'] = fitness_score_by_cat[0]
        self.fitness_score['1'] = fitness_score_by_cat[1]
        self.fitness_score['2'] = fitness_score_by_cat[2]
        self.fitness_score['3'] = fitness_score_by_cat[3]

        # overall score need to be normalized by # of cat
        self.fitness_score['avg'] = np.sum(precision + tn_rate) * 125  # score ranged from 0 to 1000

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
        height, width = self.height, self.width
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

        return tfm.Transformer.get_tfm(self.width, self.height, gene=gene)


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
            creature = Creature(img_shape=img_shape)
            creature.build_chromosome()
            population.append(creature)
        return population

    @staticmethod
    def train_population(population: list, train_data: list, src_dir: str, e, epoch_limit=100, lr=1, silence=False):
        """
        Train the population.
        # TODO: in the future, implement an epoch report, show avg confusion matrix stats
        # TODO: in the future, store the midway results in case the training get interrupted
        :param population: population
        :param train_data: list of the training data in tuple format [(id, cat)]. id and cat must be int
        :param e: stopping coefficient: intuitively, how many improvement (%) do you if you made 100 errors in two
        consecutive training iterations
        :param src_dir: the directory for placing the training images
        :param epoch_limit: number of iterations for the training the perceptron
        :param lr: learning rate
        :param silence: whether report the training stats
        :return: void
        """

        # reset the weights
        for creature in population:
            creature.reset_weights()

        # train perceptron for each creature
        n = len(population)
        error_prev, error_curr = np.full((500,), fill_value=100), np.zeros((500,))
        for i in range(epoch_limit):

            # reset the confusion matrix for each creature at the beginning of each training epoch
            print("Training Epoch = {}\n".format(i))
            PopulationOperator.reset_population_confusion(population)

            # train on each data entry
            for j in range(len(train_data)):
                img_id = train_data[j][0]
                label = train_data[j][1]
                img_path = os.path.join(src_dir, "{}.png".format(img_id))
                img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY)
                for creature in population:
                    # TODO: in the future implement more classifier e.g. SVM, KNN etc
                    creature.train_perceptron_multiclass(img=img, label=label, lr=lr)  # train perceptron

            # early stop
            for k, creature in enumerate(population):
                if not creature.lock:
                    error_curr[k] = n - np.sum(creature.confusion.diagonal())
                    # intuitively, theta is how many improvement do you need for
                    theta = abs(error_curr[k] - error_prev[k]) * 2 / ((error_prev[k] + error_curr[k]) + 10e-6)
                    if theta < e and i > 5:
                        creature.lock_weights()
                        print("Creature {} weights locked at epoch {}. Error {}".format(k, i, error_curr[k]/n))
                    error_prev[k] = error_curr[k]

            # report for the current epoch
            if not silence:
                PopulationOperator.report(population)

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
                creature.validate_perceptron(img=img, label=label)  # validate perceptron

        # compute the compute_fitness scores for creatures in population
        for creature in population:
            creature.compute_fitness()

        print("Validation done.")
        PopulationOperator.report(population)

    @staticmethod
    def eliminate_population(population: list, mode, t=0.25):
        """
        # TODO: compare two strategy
        # TODO: strategy - 0: eliminate lowest t by overall fitness score
        # TODO: strategy - 1: eliminate lowest t by keeping top p% of good performer from each cat
        # TODO: strategy - 1 should always give a good solution as they will be boosted at the end
        # TODO: uninformative prior -> keep top (1 - sqrt(sqrt(t))) from each class
        # TODO: strategy - 2: combine two -> keep t -> alpha portion of t from top performer rest from each class
        Eliminate the underperformed creatures in place.
        Due to the current fitness score set up, the validating data need to be balanced, otherwise it could be bias
        to the dominate cat.
        :param population: population of creatures
        :param mode: 0: eliminate overall bad performer by threshold; 1: keep % of the good performer by each cat
        :param t: threshold
        :return:
        """

        # remove the last
        if mode == '0':
            population.sort(key=lambda x: x.fitness_score['overall'], reverse=True)  # O(nlgn)
            n = int(len(population) * t)
            for i in range(n):
                population.pop()  # pop last item is O(1)
        elif mode == '1':
            p = np.sqrt(np.sqrt(t))
            keep_num = int(round(len(population) * (1 - p)))
            collector = []
            for i in range(4):
                cat = str(i)
                population.sort(key=lambda x: x.fitness_score[cat], reverse=True)
                for gene in population[:keep_num + 1]:
                    if gene not in collector:
                        collector.append(gene)
            population[:] = collector
        else:
            raise Exception("Mode must be '0': eliminate by overall fitness; '1': by cat but was {}".format(mode))

        print("Eliminated bad performers, now have {} in pool".format(len(population)))
        PopulationOperator.report(population)

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

            # Tournament based on random criteria
            criteria = random.randrange(5)
            if criteria == 4:
                criteria = 'avg'
            else:
                criteria = str(criteria)
            fit1 = population[idx1].fitness_score[criteria]
            fit2 = population[idx2].fitness_score[criteria]
            if np.random.choice(2, 1, p=[1 - p, p]) == 1:
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
        child = copy.deepcopy(parent1)  # deep copy

        if np.random.choice(2, 1, p=[1 - cross_rate, cross_rate]) == 1:
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

        # TODO: confusion matrix does not need to be updated/reset during the training period
        for creature in population:
            creature.reset_confusion()

    @staticmethod
    def save_population(population, dst_file):
        meta_data = []
        for creature in population:
            meta_data.append(creature.info())

        with open(dst_file, 'w') as f:
            json.dump(meta_data, f)

        print("Model params saved at {}".format(dst_file))

    @staticmethod
    def load_population(src_file):
        """
        Load a population from a json file
        :param src_file:
        :return:
        """

        loaded_population = []

        with open(src_file, 'r') as f:
            meta_data = json.load(f)
        for creature_info in meta_data:
            creature = Creature(img_shape=(10, 10))
            creature.img_shape = tuple(creature_info['img_shape'])
            creature.x1, creature.x2, creature.y1, creature.y2 = creature_info['patch']
            creature.height = creature_info['height']
            creature.width = creature_info['width']
            for gene in creature_info['genes']:
                code = gene['code']
                params = gene['params']
                h, w = gene['h'], gene['w']
                tfm_x = tfm.Transformer.get_tfm(width=w, height=h, gene=code)
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
    def report(population: list):
        """
        Report the best precision, recall from a population
        :param population:
        :return:
        """
        precision = np.array([0, 0, 0, 0], dtype=np.float32)
        recall = np.array([0, 0, 0, 0], dtype=np.float32)
        for creature in population:
            tp = creature.confusion.diagonal()  # tp on diagonal
            fp = np.sum(creature.confusion, axis=0) - tp  # fp are each column sum minus the tp
            fn = np.sum(creature.confusion, axis=1) - tp  # fn are each row sum minus the tp
            temp_precision = np.round(tp / (fp + tp + 0.001), 4)  # in case the nan value
            temp_recall = np.round(tp / (tp + fn + 0.001), 4)  # in case the nan value
            precision = np.maximum(temp_precision, precision)
            recall = np.maximum(temp_recall, recall)
        print("{:>10}: {:>10} {:>10} {:>10} {:>10}".format("Class", 0, 1, 2, 3))
        print("{:>10}: {:>10} {:>10} {:>10} {:>10}".format("Precision", precision[0], precision[1],
                                                           precision[2], precision[3]))
        print("{:>10}: {:>10} {:>10} {:>10} {:>10}".format("Recall", recall[0], recall[1], recall[2],
                                                           recall[3]))
        print("-------------------------------------------------------\n")

