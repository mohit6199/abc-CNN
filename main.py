import keras
from abcCNN import abcCNN
import numpy as np
import time
import keras.backend
import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ######## Algorithm parameters ##################
    
    # dataset = "mnist"
    # dataset = "mnist-rotated-digits"
    # dataset = "mnist-rotated-with-background"
    dataset = "rectangles"
    # dataset = "rectangles-images"
    # dataset = "convex"
    # dataset = "fashion-mnist"
    # dataset = "mnist-random-background"
    # dataset = "mnist-background-images"

    limit = 1
    number_runs = 5
    number_iterations = 10
    population_size = 10

    batch_size_abc = 64
    batch_size_full_training = 64
    
    epochs_abc = 1
    epochs_full_training = 50
    
    max_conv_output_channels = 256
    max_fully_connected_neurons = 300

    min_layer = 3
    max_layer = 10

    # Probability of each layer type (should sum to 1)
    probability_convolution = 0.6
    probability_pooling = 0.3
    probability_fully_connected = 0.1

    max_conv_kernel_size = 7

    Cg = 0.5
    dropout = 0.5

    ########### Run the algorithm ######################
    results_path = "./results/" + dataset + "/"

    if not os.path.exists(results_path):
            os.makedirs(results_path)

    all_gBest_metrics = np.zeros((number_runs, 2))
    runs_time = []
    all_best_accuracy = []
    all_best_losses = []
    all_discarded_best_accuracy = []
    all_discarded_best_losses = []
    #for models that are trained for complete number of epochs
    all_complete_best_accuracy = []
    all_complete_disacrded_best_accuracy = []

    for i in range(number_runs):
        print("Run number: " + str(i))
        start_time = time.time()
        abc = abcCNN(dataset=dataset, n_iter=number_iterations, pop_size=population_size, \
            batch_size=batch_size_abc, epochs=epochs_abc, min_layer=min_layer, max_layer=max_layer, \
            conv_prob=probability_convolution, pool_prob=probability_pooling, \
            fc_prob=probability_fully_connected, max_conv_kernel=max_conv_kernel_size, \
            max_out_ch=max_conv_output_channels, max_fc_neurons=max_fully_connected_neurons, dropout_rate=dropout,limit = limit)

        print("Final losses = ", abc.loss)
        print("Final Accuracies = ",abc.accuracy)
        print("Final Trial = ", abc.trial)

        #Save the final loss and accuracy of the run
        np.save(results_path + "final_loss_" + str(i) + "_run.npy", abc.loss)
        np.save(results_path + "final_accuracy_" + str(i) + "_run.npy", abc.accuracy)

        minLossIndex = abc.loss.index(min(abc.loss))

        print("\n\n\nThe best model produced is as follows")
        print("the model = " , abc.population.particle[minLossIndex])
        print("Accuracy = ",abc.population.particle[minLossIndex].accuracy)
        print("Loss = ", abc.population.particle[minLossIndex].loss)


        #Trains the best model for more epochs and saves the metrics
        hist,eval = abc.model_fit_comp(min_loss_index=minLossIndex,epochs_full_training=epochs_full_training)
        #print("\n\n\nComplete training hist = ",hist)
        print("\n\n\nComplete training accuracy = ",hist.history["accuracy"][-1])
        print("Complete training loss = ", hist.history["loss"][-1])
        print("Complete training eval_loss = ", eval[0])
        print("Complete training eval_accuracy = ",eval[1])

        all_best_accuracy.append(hist.history["accuracy"][-1])
        all_best_losses.append((hist.history["loss"][-1]))

        #Saves the best Model in Yaml Form
        best_model_yaml = abc.population.particle[minLossIndex].model.to_yaml()
        with open(results_path + "best_model_" + str(i) + "_run.yaml", "w") as yaml_file:
            yaml_file.write(best_model_yaml)
        # Save best Best model weights to HDF5 file
        abc.population.particle[minLossIndex].model.save_weights(results_path + "best_model_weights_" + str(i) + "_run.h5")

        print("ALL LOSSES and all ACCURACIES")
        print(np.transpose(abc.allLosses))
        print("\n")
        print(np.transpose(abc.allAccuracies))

        print("\n\nBEST DISACRDED SOLUTION = ",abc.discardedBestSolution)
        print("Best Discarded soln accuracy = ",abc.discardedBestSolution.accuracy)



        disacrded_hist,discarded_eval = abc.model_fit_comp_disacrded(epochs_full_training)
        #print("\n\n\nComplete training hist of disacrded Solution = ", disacrded_hist)
        print("\n\n\nComplete training accuracy = ", disacrded_hist.history["accuracy"][-1])
        print("Complete training loss = ", disacrded_hist.history["loss"][-1])
        print("Complete training eval_loss of discarded solution = ", discarded_eval[0])
        print("Complete training eval_accuracy of discarded solution = ", discarded_eval[1])

        all_discarded_best_accuracy.append(disacrded_hist.history["accuracy"][-1])
        all_discarded_best_losses.append(disacrded_hist.history["loss"][-1])

        best_discarded_model_yaml = abc.discardedBestSolution.model.to_yaml()
        with open(results_path + "best_discarded_model_" + str(i) + "_run.yaml", "w") as yaml_file:
            yaml_file.write(best_discarded_model_yaml)
        # Save best gBest model weights to HDF5 file
        abc.discardedBestSolution.model.save_weights(results_path + "best_discarded_model_weights_" + str(i) + "_run.h5")



        #CALCULATES THE TIME TAKEN
        end_time = time.time()
        running_time = end_time - start_time
        runs_time.append(running_time)
        print("This run took: " + str(running_time) + " seconds.")


        for x in range(0,population_size):
            plt.plot(np.transpose(abc.allAccuracies)[x])
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.savefig(results_path + "accuracy_run_" + str(i) + ".png")
        plt.close()

    print("\n\nRUNS OVER \n")
    print("ALL the best Accuracies = ",all_best_accuracy)
    print("ALL the best Losses = ", all_best_losses)
    print("All best accuracies average = ", sum(all_best_accuracy)/number_runs)
    print("All best losses average = ", sum(all_best_losses) / number_runs)
    print("time taken for all runs = ",runs_time)




    '''    # Plot current gBest
        matplotlib.use('Agg')
        plt.plot(abc.gBest_acc)
        plt.xlabel("Iteration")
        plt.ylabel("gBest acc")
        plt.savefig(results_path + "gBest-iter-" + str(i) + ".png")
        plt.close()

        print('gBest architecture: ')
        print(abc.gBest)
    
        np.save(results_path + "gBest_inter_" + str(i) + "_acc_history.npy", abc.gBest_acc)

        np.save(results_path + "gBest_iter_" + str(i) + "_test_acc_history.npy", abc.gBest_test_acc)

        end_time = time.time()

        running_time = end_time - start_time

        runs_time.append(running_time)

        # Fully train the gBest model found
        n_parameters = abc.fit_gBest(batch_size=batch_size_full_training, epochs=epochs_full_training, dropout_rate=dropout)
        all_gbest_par.append(n_parameters)

        # Evaluate the fully trained gBest model
        gBest_metrics = abc.evaluate_gBest(batch_size=batch_size_full_training)

        if gBest_metrics[1] >= best_gBest_acc:
            best_gBest_acc = gBest_metrics[1]

            # Save best gBest model
            best_gBest_yaml = abc.gBest.model.to_yaml()

            with open(results_path + "best-gBest-model.yaml", "w") as yaml_file:
                yaml_file.write(best_gBest_yaml)
            
            # Save best gBest model weights to HDF5 file
            abc.gBest.model.save_weights(results_path + "best-gBest-weights.h5")

        all_gBest_metrics[i, 0] = gBest_metrics[0]
        all_gBest_metrics[i, 1] = gBest_metrics[1]

        print("This run took: " + str(running_time) + " seconds.")

         # Compute mean accuracy of all runs
        all_gBest_mean_metrics = np.mean(all_gBest_metrics, axis=0)

        np.save(results_path + "/time_to_run.npy", runs_time)

        # Save all gBest metrics
        np.save(results_path + "/all_gBest_metrics.npy", all_gBest_metrics)

        # Save results in a text file
        output_str = "All gBest number of parameters: " + str(all_gbest_par) + "\n"
        output_str = output_str + "All gBest test accuracies: " + str(all_gBest_metrics[:,1]) + "\n"
        output_str = output_str + "All running times: " + str(runs_time) + "\n"
        output_str = output_str + "Mean loss of all runs: " + str(all_gBest_mean_metrics[0]) + "\n"
        output_str = output_str + "Mean accuracy of all runs: " + str(all_gBest_mean_metrics[1]) + "\n"

        print(output_str)

        with open(results_path + "/final_results.txt", "w") as f:
            try:
                print(output_str, file=f)
            except SyntaxError:
                print >> f, output_str
'''
