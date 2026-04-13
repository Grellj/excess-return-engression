import sys
import ere_dataprep
from third_party.engression.engression import engression

# Ensures that the model output is saved in the results folder as well as printed on the console during training
class ERERunnerVisualizer:
    def __init__(self, output1, output2):
        self.output1 = output1
        self.output2 = output2

    def write(self, output):
        self.output1.write(output)
        self.output2.write(output)

learning_rates = [0.001, 0.0001, 0.00001]
batch_sizes = [32, 64, 128, 256, 512]
console_output = sys.stdout
for bs in batch_sizes:
    for l in learning_rates:
        file_output= open("results/model_run_overview_bs" + str(bs) + "lr" + str(l) + ".txt", "w")
        ere_runner_visualizer = ERERunnerVisualizer(console_output, file_output)
        sys.stdout = ere_runner_visualizer
        #  Constructs and runs the first 100-epoch block of the training experiment itself.
        ere_model = engression(ere_dataprep.X_train, ere_dataprep.Y_train, lr = l, num_epochs=50, batch_size=bs)
        # Evaluates model performance after first training using the validation set
        print("Energy loss evaluation")
        print(ere_model.eval_loss(ere_dataprep.X_validate, ere_dataprep.Y_validate, loss_type="energy", verbose=True))
        # subsequent 9 blocks of training and evaluation
        for i in range(1, 10):
            ere_model.train(ere_dataprep.X_train, ere_dataprep.Y_train, num_epochs=50, batch_size=bs)
            print("Energy loss evaluation")
            print(ere_model.eval_loss(ere_dataprep.X_validate, ere_dataprep.Y_validate, loss_type="energy", verbose=True))
        print("L2 loss evaluation")
        print(ere_model.eval_loss(ere_dataprep.X_validate, ere_dataprep.Y_validate))
        sys.stdout = console_output
        file_output.close()