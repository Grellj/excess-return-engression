import ere_dataprep

# Serves to display essential information about the training data in the results folder.
with open("results/input_info.txt", "w") as input_info:
    input_info.write("the shape of the X-tensor used in training is: \n")
    input_info.write(str(ere_dataprep.X_train.shape))
    input_info.write("\nThe shape of the Y-tensor used in training is: \n")
    input_info.write(str(ere_dataprep.Y_train.shape))
    input_info.write("\nThe shape of the X-tensor used for validation is: \n")
    input_info.write(str(ere_dataprep.X_validate.shape))
    input_info.write("\nThe shape of the Y-tensor used for validation is: \n")
    input_info.write(str(ere_dataprep.Y_validate.shape))
    input_info.write("\nThe shape of the X-tensor used for final testing is: \n")
    input_info.write(str(ere_dataprep.X_test.shape))
    input_info.write("\nThe shape of the Y-tensor used for final testing is: \n")
    input_info.write(str(ere_dataprep.Y_test.shape))
    input_info.write("\nNote that batch sizes 32, 64, 128, 256 and 512 are being implemented in ere_runner.py and thus not displayed here. \n")