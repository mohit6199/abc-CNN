import keras
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
import keras.backend
from particle import foodSource
from population import Population

import numpy as np

from copy import deepcopy

class abcCNN:
    def __init__(self, dataset, n_iter, pop_size, batch_size, epochs, min_layer, max_layer, \
        conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, dropout_rate,limit):
        
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.epochs = epochs

        self.batch_size = batch_size
        self.gBest_acc = np.zeros(n_iter)
        self.gBest_test_acc = np.zeros(n_iter)
        self.probability = [None] * pop_size


        self.fitness = [None] * pop_size
        self.trial = [None] * pop_size
        self.discardedBestSolution = None
        self.allLosses = []
        self.allAccuracies = []
        self.limit = limit
        self.discardedSolutions = []


        if dataset == "mnist":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        
        if dataset == "fashion-mnist":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()

            self.x_train = self.x_train.astype('float32')
            self.x_test = self.x_test.astype('float32')
            self.x_train /= 255
            self.x_test /= 255

        if dataset == "mnist-background-images":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            train = np.loadtxt("./datasets/mnist-background-images/mnist_background_images_train.amat")
            test = np.loadtxt("./datasets/mnist-background-images/mnist_background_images_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]

        if dataset == "mnist-rotated-digits":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            train = np.loadtxt("./datasets/mnist-rotated-digits/mnist_all_rotation_normalized_float_train_valid.amat")
            test = np.loadtxt("./datasets/mnist-rotated-digits/mnist_all_rotation_normalized_float_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]

        if dataset == "mnist-random-background":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            train = np.loadtxt("./datasets/mnist-random-background/mnist_background_random_train.amat")
            test = np.loadtxt("./datasets/mnist-random-background/mnist_background_random_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]

        if dataset == "mnist-rotated-with-background":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            train = np.loadtxt("./datasets/mnist-rotated-with-background/mnist_all_background_images_rotation_normalized_train_valid.amat")
            test = np.loadtxt("./datasets/mnist-rotated-with-background/mnist_all_background_images_rotation_normalized_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]

        if dataset == "rectangles":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 2

            train = np.loadtxt("./datasets/rectangles/rectangles_train.amat")
            test = np.loadtxt("./datasets/rectangles/rectangles_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]

        if dataset == "rectangles-images":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 2

            train = np.loadtxt("./datasets/rectangles-images/rectangles_im_train.amat")
            test = np.loadtxt("./datasets/rectangles-images/rectangles_im_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]

        if dataset == "convex":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 2

            train = np.loadtxt("./datasets/convex/convex_train.amat")
            test = np.loadtxt("./datasets/convex/convex_test.amat")

            self.x_train = train[:, :-1]
            self.x_test = test[:, :-1]

            # Reshape images to 28x28
            self.x_train = np.reshape(self.x_train, (-1, 28, 28))
            self.x_test = np.reshape(self.x_test, (-1, 28, 28))

            self.y_train = train[:, -1]
            self.y_test = test[:, -1]

        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2], input_channels)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2], input_channels)

        self.y_train = keras.utils.to_categorical(self.y_train, output_dim)
        self.y_test = keras.utils.to_categorical(self.y_test, output_dim)

        print("Initializing population...")
        self.population = Population(pop_size, min_layer, max_layer, input_width, input_height, input_channels, conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, output_dim)

        #there are 3 phases of the ABC algorithm 1) Emplolyed bee 2) onlooker bee and 3) scout bee

        for iteration in range(0,self.n_iter):

            self.loss = [None] * pop_size
            self.accuracy = [None] * pop_size

            print("\n------------ITERATION = ", iteration+1 , "---------------------\n")
            ####-------- EMPLOYED BEE PAHSE -------#########
            print("\n####-------- EMPLOYED BEE PHASE -------#########\n")
            i = 0
            for foodSrc in self.population.particle:
                print("Food Source = ",i)
                foodSrc.generateNewParticle(self.population.particle[np.random.randint(0,pop_size)].layers)
                foodSrc.generateModel(dropout_rate)
                foodSrc.model_fit(self.x_train,self.y_train,batch_size,epochs)
                foodSrc.calculateFitness()
                self.loss[i] = foodSrc.loss
                self.accuracy[i] = foodSrc.accuracy
                self.fitness[i] = foodSrc.fitness
                self.trial[i] = foodSrc.trial
                print("Loss = ", self.loss[i])
                i += 1
                if i == pop_size:
                    i = 0

            print("\n####-------- EMPLOYED BEE PHASE ENDS -------#########\n")

            self.probability = [0.9 * x / max(self.fitness) + 0.1 for x in self.fitness]
            print("Probabilities = \n",self.probability)

            #########----- ONLOOKER BEE PHASE ---------##########
            print("\n####-------- ONLOOKER BEE PHASE -------#########\n")

            i = 0#to cunt pop_size bees
            x = 0 # to keep tract of the food sources
            while i < pop_size:
                r = np.random.uniform()
                print("bee = ", i)
                print("foodSource = ", x)
                if r < self.probability[x]:
                    self.population.particle[x].generateNewParticle(self.population.particle[np.random.randint(0,pop_size)].layers)
                    self.population.particle[x].generateModel(dropout_rate)
                    self.population.particle[x].model_fit(self.x_train,self.y_train,batch_size,epochs)
                    self.population.particle[x].calculateFitness()
                    self.loss[x] = self.population.particle[x].loss
                    self.accuracy[x]=self.population.particle[x].accuracy
                    self.fitness[x] = self.population.particle[x].fitness
                    self.trial[x] = self.population.particle[x].trial
                    i += 1
                x += 1
                if x == pop_size:
                    x = 0
            print("\n####-------- ONLOOKER BEE PHASE ENDS -------#########\n")


            print("Trial = ", self.trial)
            #remember to add the comcept of !LIMIT!
            ##############--------- SCOUT BEE PHASE --------------###############




            print("\n####-------- SCOUT BEE PHASE -------#########\n")
            if max(self.trial) >= self.limit:
                maxTrailPos = self.trial.index(max(self.trial))

                if self.discardedBestSolution == None:
                    self.discardedBestSolution = self.population.particle[maxTrailPos]
                else:
                    if self.discardedBestSolution.loss > self.population.particle[maxTrailPos].loss:
                        self.discardedBestSolution = self.population.particle[maxTrailPos]
                        print("Best Discarded Particle Updated with loss = ",self.discardedBestSolution.loss)

                self.population.particle[maxTrailPos] = foodSource(min_layer, max_layer, 0, input_width, input_height, input_channels, conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, output_dim)
                self.trial[maxTrailPos] = 0
                self.discardedSolutions.append(maxTrailPos)
                print(maxTrailPos , "th particle was Discarded from the population")
                print("\n####-------- SCOUT BEE PHASE ENDS -------#########\n")
            else:
                print("No trial is greater than the limit ")

            print("LOSS = ",self.loss)
            print("TRIAL = ",self.trial)
            self.allLosses.append(self.loss)
            self.allAccuracies.append(self.accuracy)
            print("ALL LOSSES = ",self.allLosses)
            print("ALL ACCURACIES = ",self.allAccuracies)
