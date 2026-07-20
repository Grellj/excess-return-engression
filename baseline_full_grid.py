import sys
import ere_dataprep
from third_party.engression.engression import engression
import csv
import config
import pandas as pd

# This script serves to run iterations of single model experiments or full grid searches for small search spaces and save their output
# number of epochs in each block of training, multiplied with the number of blocks it gives the total number of epochs
num_blocks = config.num_blocks
block_epochs = config.block_epochs
num_runs = config.num_runs
for current_run in range (1, num_runs+1):
    # Creates csv file to save best training block result for each of the model runs
    with open("results/output_info/hyperparameter_tuning/model_run_" + str(current_run) + "/output_info.csv", "w", newline= "") as output_info:
        output_writer = csv.writer(output_info)
        output_writer.writerow(["Batch Size", "LR", "Hidden Dim", "Noise Dim", "Number Layer", "Add BN", "Resblock", "Validation Energy Loss", "Achieved in Block"])
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
    for bs in batch_sizes:
        for lr in learning_rates:
            for hd in hidden_dim:
                for nd in noise_dim:
                    for nl in num_layer:
                        for bn in add_bn:
                            for rb in resblock:
                                file_output= open("results/output_info/hyperparameter_tuning/model_run_" + str(current_run) + "/model_run_overview_bs"+str(bs)+"lr"+str(lr)+"hd" + str(hd) + "nd" + str(nd) + "nl" + str(nl) +"bn"+str(bn)+"rb"+str(rb)+ ".txt", "w")
                                ere_runner_visualizer = ERERunnerVisualizer(console_output, file_output)
                                sys.stdout = ere_runner_visualizer
                                #  Constructs and runs the first block of the training experiment itself.
                                ere_model = engression(ere_dataprep.X_train, ere_dataprep.Y_train, lr = lr, num_epochs= block_epochs, batch_size=bs, hidden_dim = hd, noise_dim = nd, num_layer = nl, add_bn = bn, resblock = rb)
                                # Evaluates model performance after first training using the validation set
                                print("Energy loss evaluation")
                                current_e_values = ere_model.eval_loss(ere_dataprep.X_validate, ere_dataprep.Y_validate, loss_type="energy", verbose = True)
                                print(current_e_values)
                                current_block = 1
                                # subsequent blocks of training and evaluation
                                for b in range(2, (num_blocks+1)):
                                    ere_model.train(ere_dataprep.X_train, ere_dataprep.Y_train, num_epochs=block_epochs, batch_size=bs)
                                    print("Energy loss evaluation")
                                    new_e_values = ere_model.eval_loss(ere_dataprep.X_validate, ere_dataprep.Y_validate, loss_type="energy", verbose = True)
                                    print(new_e_values)
                                    if new_e_values[0] < current_e_values[0]:
                                        current_e_values = new_e_values
                                        current_block = b
                                best_block_output = [bs, lr, hd, nd, nl, bn, rb, current_e_values[0], current_block]
                                with open("results/output_info/hyperparameter_tuning/model_run_" + str(current_run) + "/output_info.csv", "a", newline="") as output_info:
                                    output_writer = csv.writer(output_info)
                                    output_writer.writerow(best_block_output)
                                sys.stdout = console_output
                                file_output.close()

# The following aggregator code used to be a separate file, hence the structural overlap
# It serves to summarize the following measurements across all model runs for each hyperparameter combination respectively: 
# the median optimum energy score achieved 
# the mean optimum energy score 
# the highest occuring Energy Score
# the lowest occuring Energy Score
# the maximum amount of training blocks required to reach any energy score optimum 
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
# df for energy score values
energy_df = pd.DataFrame(index= range(num_bs*num_lr*num_hd*num_nd*num_nl*num_bn*num_rb), columns = range(num_runs))
# df for training blocks in which es optima were achieved, used during optimization
number_blocks_df = pd.DataFrame(index= range(num_bs*num_lr*num_hd*num_nd*num_nl*num_bn*num_rb), columns = range(num_runs))
for iteration in range(0, num_runs):
    energy_df[iteration] = pd.read_csv("results/output_info/hyperparameter_tuning/model_run_" + str(iteration+1) + "/output_info.csv").loc[:, "Validation Energy Loss"]
    number_blocks_df[iteration] = pd.read_csv("results/output_info/hyperparameter_tuning/model_run_" + str(iteration+1) + "/output_info.csv").loc[:, "Achieved in Block"]
# assembling the required statistics
energy_medians = energy_df.median(axis=1)
energy_means = energy_df.mean(axis=1)
energy_maxima = energy_df.max(axis=1)
energy_minima = energy_df.min(axis=1)
block_maxima = number_blocks_df.max(axis=1)

i = 0
# the csv containing the final overview
with open("results/output_info/hyperparameter_tuning/baseline_output_aggregation.csv", "w", newline="") as baseline_output_aggregation:
    aggregation_writer = csv.writer(baseline_output_aggregation)
    aggregation_writer.writerow(["Batch Size", "LR", "Hidden Dim", "Noise Dim", "Num Layer", "Add BN", "Resblock", "Median ES", "Mean ES", "Min ES", "Max ES", "Max Blocks"])
    # Note that a compatible loop implementation above guarantees the correct order of models in the csv files to be aggregated
    for bs in batch_sizes:
        for lr in learning_rates:
            for hd in hidden_dim:
                for nd in noise_dim:
                    for nl in num_layer:
                        for bn in add_bn:
                            for rb in resblock:
                                aggregation_writer.writerow([bs, lr, hd, nd, nl, bn, rb, energy_medians.iloc[i], energy_means.iloc[i], energy_minima.iloc[i], energy_maxima.iloc[i], block_maxima.iloc[i]])
                                i+=1

# creates LaTeX export
tex_df = pd.read_csv("results/output_info/hyperparameter_tuning/baseline_output_aggregation.csv")
tex_str = tex_df.to_latex(index = False)
with open("results/output_info/hyperparameter_tuning/baseline_output_aggregation.tex", "w") as output_tex:
    output_tex.write(tex_str)