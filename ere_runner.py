import sys
import ere_dataprep
from third_party.engression.engression import engression
import csv

# This script serves to run model experiments and save their output

# Creates csv file to save best training block result for each of the model runs
with open("results/output_info.csv", "w", newline= "") as output_info:
    output_writer = csv.writer(output_info)
    output_writer.writerow(["Batch Size", "Learning Rate", "Validation Energy Loss", "Achieved in Block"])
# Ensures that the model output is saved in the results folder as well as printed on the console during training
class ERERunnerVisualizer:
    def __init__(self, output1, output2):
        self.output1 = output1
        self.output2 = output2

    def write(self, output):
        self.output1.write(output)
        self.output2.write(output)

batch_sizes = [32, 64, 128, 256, 512]
learning_rates = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
console_output = sys.stdout
for bs in batch_sizes:
    for l in learning_rates:
        file_output= open("results/model_run_overview_bs" + str(bs) + "lr" + str(l) + ".txt", "w")
        ere_runner_visualizer = ERERunnerVisualizer(console_output, file_output)
        sys.stdout = ere_runner_visualizer
        #  Constructs and runs the first 10-epoch block of the training experiment itself.
        ere_model = engression(ere_dataprep.X_train, ere_dataprep.Y_train, lr = l, num_epochs=10, batch_size=bs)
        # Evaluates model performance after first training using the validation set
        print("Energy loss evaluation")
        current_e_values = ere_model.eval_loss(ere_dataprep.X_validate, ere_dataprep.Y_validate, loss_type="energy", verbose = True)
        print(current_e_values)
        current_block = 1
        # subsequent 9 blocks of training and evaluation
        for b in range(2, 11):
            ere_model.train(ere_dataprep.X_train, ere_dataprep.Y_train, num_epochs=10, batch_size=bs)
            print("Energy loss evaluation")
            new_e_values = ere_model.eval_loss(ere_dataprep.X_validate, ere_dataprep.Y_validate, loss_type="energy", verbose = True)
            print(new_e_values)
            if new_e_values[0] < current_e_values[0]:
                current_e_values = new_e_values
                current_block = b
        print("L2 loss evaluation")
        print(ere_model.eval_loss(ere_dataprep.X_validate, ere_dataprep.Y_validate))
        best_block_output = [bs, l, current_e_values[0], current_block]
        with open("results/output_info.csv", "a", newline="") as output_info:
            output_writer = csv.writer(output_info)
            output_writer.writerow(best_block_output)
        sys.stdout = console_output
        file_output.close()