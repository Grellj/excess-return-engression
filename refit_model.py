import sys
import ere_dataprep
from third_party.engression.engression import engression
import csv
import results.output_info.config as config
import numpy as np

# This script serves to run iterations of single model experiments or full grid searches for small search spaces and save their output, using a rolling refit
# number of epochs in each stage of training
num_epochs = config.num_epochs
frequency = config.frequency
train_len = len(ere_dataprep.X_train)
validation_len = len(ere_dataprep.X_validate)
num_runs = config.num_runs # number of repetitions for robustness check
for current_run in range (1, num_runs+1):
    # Creates csv file to save median Energy Loss for each of the model runs accross refit stages
    with open("results/output_info/model_run_" + str(current_run) + "/output_info.csv", "w", newline= "") as output_info:
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
                                    file_output= open("results/output_info/model_run_" + str(current_run) + "/model_run_overview_epochs"+str(ne)+"bs"+str(bs)+"lr"+str(lr)+"hd" + str(hd) + "nd" + str(nd) + "nl" + str(nl) +"bn"+str(bn)+"rb"+str(rb)+ ".txt", "w")
                                    ere_runner_visualizer = ERERunnerVisualizer(console_output, file_output)
                                    sys.stdout = ere_runner_visualizer
                                    #  Constructs and runs the first stage of the training experiment itself.
                                    i=0
                                    j=train_len
                                    ere_model = engression(ere_dataprep.X_train, ere_dataprep.Y_train, lr = lr, num_epochs= ne, batch_size=bs, hidden_dim = hd, noise_dim = nd, num_layer = nl, add_bn = bn, resblock = rb)
                                    # Evaluates model performance after first training using the first forecast frequency of the validation set
                                    current_e_value = ere_model.eval_loss(ere_dataprep.X_validate[i:i+frequency], ere_dataprep.Y_validate[i:i+frequency], loss_type="energy", sample_size=10)
                                    print(current_e_value)
                                    e_values = [current_e_value]
                                    i+=frequency
                                    j+=frequency
                                    # subsequent stages of training and evaluation
                                    while i<validation_len:
                                        ere_model = engression(ere_dataprep.X_train_extended[i:j], ere_dataprep.Y_train_extended[i:j], lr = lr, num_epochs= ne, batch_size=bs, hidden_dim = hd, noise_dim = nd, num_layer = nl, add_bn = bn, resblock = rb)
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
                                    with open("results/output_info/model_run_" + str(current_run) + "/output_info.csv", "a", newline="") as output_info:
                                        output_writer = csv.writer(output_info)
                                        output_writer.writerow(output)
                                    sys.stdout = console_output
                                    file_output.close()
