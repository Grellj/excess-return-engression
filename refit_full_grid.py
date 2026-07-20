import sys
import ere_dataprep
from third_party.engression.engression import engression
import csv
import config
import numpy as np
import pandas as pd

# This script serves to run iterations of single model experiments or full grid searches for small search spaces and save their output, using a rolling refit
# number of epochs in each stage of training
num_epochs = config.num_epochs
frequency = config.frequency
train_len = len(ere_dataprep.X_train)
validation_len = len(ere_dataprep.X_validate)
num_runs = config.num_runs # number of repetitions for robustness check
for current_run in range (1, num_runs+1):
    # Creates csv file to save median Energy Loss for each of the model runs accross refit stages
    with open("results/output_info/hyperparameter_tuning/model_run_" + str(current_run) + "/output_info.csv", "w", newline= "") as output_info:
        output_writer = csv.writer(output_info)
        output_writer.writerow(["Epochs", "Batch Size", "LR", "Hidden Dim", "Noise Dim", "Number Layer", "Add BN", "Resblock", "Median Validation ES"])
    # Ensures that the model output is saved in the results folder as well as printed on the console during training
    class ERERunnerVisualizer:
        def __init__(self, output1, output2):
            self.output1 = output1
            self.output2 = output2
    
        def write(self, output):
            self.output1.write(output)
            self.output2.write(output)
    batch_sizes = config.batch_sizes
    learning_rates = config.learning_rates
    hidden_dim = config.hidden_dim
    noise_dim = config.noise_dim
    num_layer = config.num_layer
    add_bn = config.add_bn
    resblock = config.resblock
    console_output = sys.stdout
    for ne in num_epochs:
        for bs in batch_sizes:
            for lr in learning_rates:
                for hd in hidden_dim:
                    for nd in noise_dim:
                        for nl in num_layer:
                            for bn in add_bn:
                                for rb in resblock:
                                    file_output= open("results/output_info/hyperparameter_tuning/model_run_" + str(current_run) + "/model_run_overview_epochs"+str(ne)+"bs"+str(bs)+"lr"+str(lr)+"hd" + str(hd) + "nd" + str(nd) + "nl" + str(nl) +"bn"+str(bn)+"rb"+str(rb)+ ".txt", "w")
                                    ere_runner_visualizer = ERERunnerVisualizer(console_output, file_output)
                                    sys.stdout = ere_runner_visualizer
                                    #  Constructs and runs the first stage of the training experiment itself.
                                    i=0
                                    j=train_len
                                    ere_model = engression(ere_dataprep.X_train, ere_dataprep.Y_train, lr = lr, num_epochs= ne, batch_size=bs, hidden_dim = hd, noise_dim = nd, num_layer = nl, add_bn = bn, resblock = rb, verbose = False)
                                    # Evaluates model performance after first training using the first forecast frequency of the validation set
                                    current_e_value = ere_model.eval_loss(ere_dataprep.X_validate[i:i+frequency], ere_dataprep.Y_validate[i:i+frequency], loss_type="energy", sample_size=10)
                                    print(current_e_value)
                                    e_values = [current_e_value]
                                    i+=frequency
                                    j+=frequency
                                    # subsequent stages of training and evaluation
                                    while i<validation_len:
                                        ere_model = engression(ere_dataprep.X_train_extended[i:j], ere_dataprep.Y_train_extended[i:j], lr = lr, num_epochs= ne, batch_size=bs, hidden_dim = hd, noise_dim = nd, num_layer = nl, add_bn = bn, resblock = rb, verbose = False)
                                        if i+frequency <= validation_len:
                                            current_e_value = ere_model.eval_loss(ere_dataprep.X_validate[i:i+frequency], ere_dataprep.Y_validate[i:i+frequency], loss_type="energy", sample_size=10)
                                            i+=frequency
                                            j+=frequency
                                            print(current_e_value)
                                        else:
                                            current_e_value = ere_model.eval_loss(ere_dataprep.X_validate[i:validation_len], ere_dataprep.Y_validate[i:validation_len], loss_type="energy", sample_size=10)
                                            print(current_e_value)
                                            i+=frequency
                                        e_values.append(current_e_value)
                                    energy_median = np.median(e_values)
                                    output = [ne, bs, lr, hd, nd, nl, bn, rb, energy_median]
                                    with open("results/output_info/hyperparameter_tuning/model_run_" + str(current_run) + "/output_info.csv", "a", newline="") as output_info:
                                        output_writer = csv.writer(output_info)
                                        output_writer.writerow(output)
                                    sys.stdout = console_output
                                    file_output.close()

# The following aggregator code used to be part of a separate file, hence the structural overlap
# It serves to summarize the following measurements across all model runs for each hyperparameter combination respectively: 
# the median energy score achieved across different model runs
# the mean energy score 
# the highest occurring Energy Score
# the lowest occuring Energy Score
num_epochs = config.num_epochs
different_ne=len(num_epochs)
num_runs = config.num_runs
batch_sizes = config.batch_sizes
num_bs = len(batch_sizes)
learning_rates = config.learning_rates
num_lr = len(learning_rates)
hidden_dim = config.hidden_dim
num_hd = len(hidden_dim)
noise_dim = config.noise_dim
num_nd = len(noise_dim)
num_layer = config.num_layer
num_nl = len(num_layer)
add_bn = config.add_bn
num_bn = len(add_bn)
resblock = config.resblock
num_rb = len(resblock)
# df for energy score values, note that they are themselves aggregated accross previously computed median values
energy_df = pd.DataFrame(index= range(different_ne*num_bs*num_lr*num_hd*num_nd*num_nl*num_bn*num_rb), columns = range(num_runs))
for iteration in range(0, num_runs):
    energy_df[iteration] = pd.read_csv("results/output_info/hyperparameter_tuning/model_run_" + str(iteration+1) + "/output_info.csv").loc[:, "Median Validation ES"]
# assembling the required statistics
energy_medians = energy_df.median(axis=1)
energy_means = energy_df.mean(axis=1)
energy_maxima = energy_df.max(axis=1)
energy_minima = energy_df.min(axis=1)

i = 0
# the csv containing the final overview
with open("results/output_info/hyperparameter_tuning/refit_output_aggregation.csv", "w", newline="") as refit_output_aggregation:
    aggregation_writer = csv.writer(refit_output_aggregation)
    aggregation_writer.writerow(["Epochs", "Batch Size", "LR", "Hidden Dim", "Noise Dim", "Num Layer", "Add BN", "Resblock", "Median ES", "Mean ES", "Min ES", "Max ES"])
    # Note that a compatible loop implementation above guarantees the correct order of models in the csv files to be aggregated
    for ne in num_epochs:
        for bs in batch_sizes:
            for lr in learning_rates:
                for hd in hidden_dim:
                    for nd in noise_dim:
                        for nl in num_layer:
                            for bn in add_bn:
                                for rb in resblock:
                                    aggregation_writer.writerow([ne, bs, lr, hd, nd, nl, bn, rb, energy_medians.iloc[i], energy_means.iloc[i], energy_minima.iloc[i], energy_maxima.iloc[i]])
                                    i+=1

# creates LaTeX export
tex_df = pd.read_csv("results/output_info/hyperparameter_tuning/refit_output_aggregation.csv")
tex_str = tex_df.to_latex(index = False)
with open("results/output_info/hyperparameter_tuning/refit_output_aggregation.tex", "w") as output_tex:
    output_tex.write(tex_str)