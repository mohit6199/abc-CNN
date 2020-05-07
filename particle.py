import numpy as np
from copy import deepcopy

import utils

import keras.backend
from keras.models import Model, Sequential
from keras.layers import Input, Add, Dense, Dropout, Flatten
from keras.layers import Activation, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers 
from keras.optimizers import Adam, Nadam

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

import os
import tensorflow as tf

# Hide Tensorflow INFOS and WARNINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

class foodSource:
    def __init__(self, min_layer, max_layer, max_pool_layers, input_width, input_height, input_channels, conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, output_dim):
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels

        self.num_pool_layers = 0
        self.max_pool_layers = max_pool_layers

        self.feature_width = input_width
        self.feature_height = input_height

        self.depth = np.random.randint(min_layer, max_layer)
        self.conv_prob = conv_prob
        self.pool_prob = pool_prob
        self.fc_prob = fc_prob
        self.max_conv_kernel = max_conv_kernel
        self.max_out_ch = max_out_ch
        
        self.max_fc_neurons = max_fc_neurons
        self.output_dim = output_dim

        self.layers = []
        self.newParticleLayers = []
        self.fitness = 0
        self.loss = -1
        self.trial = 0
        self.accuracy = 0

        # Build particle architecture
        self.initialization()
        

        self.model = None
        self.model2 = None

    
    def __str__(self):
        string = ""
        for z in range(len(self.layers)):
            string = string + str(self.layers[z]["kernel"])+ "_" + self.layers[z]["type"] + "_" + str(self.layers[z]["ou_c"]) + " | "
        
        return string

    def printSecond(self):
        string = ""
        for z in range(len(self.newParticleLayers)):
            string = string + str(self.newParticleLayers[z]["kernel"]) +"_" + self.newParticleLayers[z]["type"] + "_" + str(self.newParticleLayers[z]["ou_c"]) + " | "
        return string

    def initialization(self):
        out_channel = np.random.randint(3, self.max_out_ch)
        conv_kernel = np.random.randint(3, self.max_conv_kernel)
        
        # First layer is always a convolution layer
        self.layers.append({"type": "conv", "ou_c": out_channel, "kernel": conv_kernel})

        conv_prob = self.conv_prob
        pool_prob = conv_prob + self.pool_prob
        fc_prob = pool_prob

        for i in range(1, self.depth):
            if self.layers[-1]["type"] == "fc":
                layer_type = 1.1
            else:
                layer_type = np.random.rand()#This is a random number that decides what type of layer will be formed

            if layer_type < conv_prob:
                self.layers = utils.add_conv(self.layers, self.max_out_ch, self.max_conv_kernel)

            elif layer_type >= conv_prob and layer_type <= pool_prob:
                self.layers, self.num_pool_layers = utils.add_pool(self.layers, self.fc_prob, self.num_pool_layers, self.max_pool_layers, self.max_out_ch, self.max_conv_kernel, self.max_fc_neurons, self.output_dim)
            
            elif layer_type >= fc_prob:
                self.layers = utils.add_fc(self.layers, self.max_fc_neurons)
            
        self.layers[-1] = {"type": "fc", "ou_c": self.output_dim, "kernel": -1}
    

    def generateNewParticle(self , p2):
        self.newParticleLayers = utils.generateNewSolution(self.layers , p2)




    ##### Model methods ####
    def generateModel(self,dropout_rate):

        if self.loss == -1:
            list_layers = self.layers
            self.model = Sequential()

            for i in range(len(list_layers)):
                if list_layers[i]["type"] == "conv":
                    n_out_filters = list_layers[i]["ou_c"]
                    kernel_size = list_layers[i]["kernel"]

                    if i == 0:
                        in_w = self.input_width
                        in_h = self.input_height
                        in_c = self.input_channels
                        self.model.add(Conv2D(n_out_filters, kernel_size, strides=(1,1), padding="same", data_format="channels_last", kernel_initializer='he_normal', bias_initializer='he_normal', activation=None, input_shape=(in_w, in_h, in_c)))
                        self.model.add(BatchNormalization())
                        self.model.add(Activation("relu"))
                    else:
                        self.model.add(Dropout(dropout_rate))
                        self.model.add(Conv2D(n_out_filters, kernel_size, strides=(1,1), padding="same", kernel_initializer='he_normal', bias_initializer='he_normal', activation=None))
                        self.model.add(BatchNormalization())
                        self.model.add(Activation("relu"))

                if list_layers[i]["type"] == "max_pool":
                    kernel_size = list_layers[i]["kernel"]

                    self.model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

                if list_layers[i]["type"] == "avg_pool":
                    kernel_size = list_layers[i]["kernel"]

                    self.model.add(AveragePooling2D(pool_size=(3, 3), strides=2))

                if list_layers[i]["type"] == "fc":
                    if list_layers[i-1]["type"] != "fc":
                        self.model.add(Flatten())

                    self.model.add(Dropout(dropout_rate))

                    if i == len(list_layers) - 1:
                        self.model.add(Dense(list_layers[i]["ou_c"], kernel_initializer='he_normal', bias_initializer='he_normal', activation=None))
                        self.model.add(BatchNormalization())
                        self.model.add(Activation("softmax"))
                    else:
                        self.model.add(Dense(list_layers[i]["ou_c"], kernel_initializer='he_normal', bias_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01), activation=None))
                        self.model.add(BatchNormalization())
                        self.model.add(Activation("relu"))

            adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)

            self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

        ##now generating a model for the modified solution

        list_layers = self.newParticleLayers
        self.model2 = Sequential()

        for i in range(len(list_layers)):
            if list_layers[i]["type"] == "conv":
                n_out_filters = list_layers[i]["ou_c"]
                kernel_size = list_layers[i]["kernel"]

                if i == 0:
                    in_w = self.input_width
                    in_h = self.input_height
                    in_c = self.input_channels
                    self.model2.add(
                        Conv2D(n_out_filters, kernel_size, strides=(1, 1), padding="same", data_format="channels_last",
                               kernel_initializer='he_normal', bias_initializer='he_normal', activation=None,
                               input_shape=(in_w, in_h, in_c)))
                    self.model2.add(BatchNormalization())
                    self.model2.add(Activation("relu"))
                else:
                    self.model2.add(Dropout(dropout_rate))
                    self.model2.add(Conv2D(n_out_filters, kernel_size, strides=(1, 1), padding="same",
                                          kernel_initializer='he_normal', bias_initializer='he_normal',
                                          activation=None))
                    self.model2.add(BatchNormalization())
                    self.model2.add(Activation("relu"))

            if list_layers[i]["type"] == "max_pool":
                kernel_size = list_layers[i]["kernel"]

                self.model2.add(MaxPooling2D(pool_size=(3, 3), strides=2))

            if list_layers[i]["type"] == "avg_pool":
                kernel_size = list_layers[i]["kernel"]

                self.model2.add(AveragePooling2D(pool_size=(3, 3), strides=2))

            if list_layers[i]["type"] == "fc":
                if list_layers[i - 1]["type"] != "fc":
                    self.model2.add(Flatten())

                self.model2.add(Dropout(dropout_rate))

                if i == len(list_layers) - 1:
                    self.model2.add(
                        Dense(list_layers[i]["ou_c"], kernel_initializer='he_normal', bias_initializer='he_normal',
                              activation=None))
                    self.model2.add(BatchNormalization())
                    self.model2.add(Activation("softmax"))
                else:
                    self.model2.add(
                        Dense(list_layers[i]["ou_c"], kernel_initializer='he_normal', bias_initializer='he_normal',
                              kernel_regularizer=regularizers.l2(0.01), activation=None))
                    self.model2.add(BatchNormalization())
                    self.model2.add(Activation("relu"))

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)

        self.model2.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

    def model_fit(self, x_train, y_train, batch_size, epochs):
        # TODO: add option to only use a sample size of the dataset
        if self.loss == -1:
            hist1 = self.model.fit(x=x_train, y=y_train, validation_split=0.0, batch_size=batch_size, epochs=epochs,verbose = 2)
            loss1 = hist1.history['loss'][-1]
            hist2 = self.model2.fit(x=x_train, y=y_train, validation_split=0.0, batch_size=batch_size, epochs=epochs,verbose = 2)
            loss2 = hist2.history['loss'][-1]

            self.loss = min(loss1, loss2)

            if loss1 <= loss2:
                hist = hist1
                self.accuracy = hist1.history['accuracy'][-1]

            else:
                hist = hist2
                self.trial += 1
                self.model = self.model2
                self.accuracy = hist2.history['accuracy'][-1]
        else:
            hist2 = self.model2.fit(x=x_train, y=y_train, validation_split=0.0 ,batch_size=batch_size, epochs=epochs,verbose = 2)
            loss2 = hist2.history['loss'][-1]
            if loss2 < self.loss:
                self.loss = loss2
                self.model = self.model2
                hist = hist2
                self.accuracy = hist2.history['accuracy'][-1]
                self.trial = 0
                self.accuracy = hist2.history['accuracy'][-1]
                print("!!!!!!!!!!!!!!!!!!!!!        THE NEIGHBOURING SOLUTION WAS SELECTION ABOVE THE PRESENT FOOOD SOURCE        !!!!!!!!!!!!!!!!!!!!!")
            else:
                self.trial += 1
        #print(self.accuracy)


    def calculateFitness(self):
        self.fitness = 1/(1 + self.loss)

    def model_fit_complete(self, x_train, y_train,x_test,y_test ,batch_size, epochs):
        hist = self.model.fit(x=x_train, y=y_train, validation_split=0.0, batch_size=batch_size, epochs=epochs,verbose = 2)
        results = self.model.evaluate(x = x_test,y=y_test,batch_size=batch_size,verbose = 2)
        print("Model metrics names are = ",self.model.metrics_names)
        return hist,results
    
    def model_delete(self):
        # This is used to free up memory during PSO training
        del self.model
        keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        self.model = None
