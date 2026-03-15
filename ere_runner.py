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

console_output = sys.stdout
file_output= open("results/model_run_overview.txt", "w")
ere_runner_visualizer = ERERunnerVisualizer(console_output, file_output)
sys.stdout = ere_runner_visualizer
#  Constructs and runs the training experiment itself.
ere_model = engression(ere_dataprep.X_tensor, ere_dataprep.Y_tensor, num_epochs=1000, print_every_nepoch=50)
file_output.close()