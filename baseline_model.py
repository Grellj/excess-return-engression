import sys
import ere_dataprep
from third_party.engression.engression import engression
import csv
import results.output_info.config as config

# This script serves to run iterations of single model experiments or full grid searches for small search spaces and save their output
# number of epochs in each block of training, multiplied with the number of blocks it gives the total number of epochs
num_blocks = config.num_blocks
block_epochs = config.block_epochs
num_runs = config.num_runs
for current_run in range (1, num_runs+1):
    # Creates csv file to save best training block result for each of the model runs
    with open("results/output_info/model_run_" + str(current_run) + "/output_info.csv", "w", newline= "") as output_info:
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
                                file_output= open("results/output_info/model_run_" + str(current_run) + "/model_run_overview_bs"+str(bs)+"lr"+str(lr)+"hd" + str(hd) + "nd" + str(nd) + "nl" + str(nl) +"bn"+str(bn)+"rb"+str(rb)+ ".txt", "w")
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
                                with open("results/output_info/model_run_" + str(current_run) + "/output_info.csv", "a", newline="") as output_info:
                                    output_writer = csv.writer(output_info)
                                    output_writer.writerow(best_block_output)
                                sys.stdout = console_output
                                file_output.close()
