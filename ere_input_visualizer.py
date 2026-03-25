import ere_dataprep

# Serves to display essential information about the training data in the results folder.
with open("results/input_info.txt", "w") as input_info:
    input_info.write("the shape of the X-tensor used in training is: \n")
    input_info.write(str(ere_dataprep.X_tensor.shape))
    input_info.write("\nThe shape of the Y-tensor used in training is: \n")
    input_info.write(str(ere_dataprep.Y_tensor.shape))
    input_info.write("\nNote that different batch sizes are currently being implemented in ere_runner.py and thus not displayed here. \n")