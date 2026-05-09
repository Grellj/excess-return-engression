import ere_dataprep

# Serves to display essential information about the training data in the results folder.
with open("results/input_info/input_shapes.txt", "w") as input_shapes:
    input_shapes.write("the shape of the X-tensor used in training is: \n")
    input_shapes.write(str(ere_dataprep.X_train.shape))
    input_shapes.write("\nThe shape of the Y-tensor used in training is: \n")
    input_shapes.write(str(ere_dataprep.Y_train.shape))
    input_shapes.write("\nThe shape of the X-tensor used for validation is: \n")
    input_shapes.write(str(ere_dataprep.X_validate.shape))
    input_shapes.write("\nThe shape of the Y-tensor used for validation is: \n")
    input_shapes.write(str(ere_dataprep.Y_validate.shape))
    input_shapes.write("\nThe shape of the X-tensor used for final testing is: \n")
    input_shapes.write(str(ere_dataprep.X_test.shape))
    input_shapes.write("\nThe shape of the Y-tensor used for final testing is: \n")
    input_shapes.write(str(ere_dataprep.Y_test.shape))
    input_shapes.write("\nNote that different batch sizes are being implemented in main.py and thus not displayed here. \n")