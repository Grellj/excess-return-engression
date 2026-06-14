import csv
import pandas as pd
import config

# This script serves to summarize the following measurements across all model runs for each hyperparameter combination respectively: 
# the median optimum energy score achieved 
# the mean optimum energy score 
# the maximum amount of training blocks required to reach any energy score optimum 
batch_sizes = config.batch_sizes
num_bs = len(batch_sizes)
learning_rates = config.learning_rates
num_lr = len(learning_rates)
# df for energy score values
energy_df = pd.DataFrame(index= range(num_bs*num_lr), columns = range(15))
# df for training blocks in which es optima were achieved
number_blocks_df = pd.DataFrame(index= range(num_bs*num_lr), columns = range(15))
for iteration in range(0, 15):
    energy_df[iteration] = pd.read_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/model_run_" + str(iteration+1) + "/output_info.csv").iloc[:num_bs*num_lr, 2]
    number_blocks_df[iteration] = pd.read_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/model_run_" + str(iteration+1) + "/output_info.csv").iloc[:num_bs*num_lr, 3]
# assembling the required statistics
energy_medians = energy_df.median(axis=1)
energy_means = energy_df.mean(axis=1)
block_maxima = number_blocks_df.max(axis=1)

i = 0
# the csv containing the final overview
with open("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/main_output_aggregation.csv", "w", newline="") as main_output_aggregation:
    aggregation_writer = csv.writer(main_output_aggregation)
    aggregation_writer.writerow(["Batch Size", "Learning Rate", "Median ES", "Mean ES", "Maximum Required Blocks"])
    for bs in batch_sizes:
        for lr in learning_rates:
            aggregation_writer.writerow([bs, lr, energy_medians.iloc[i], energy_means.iloc[i], block_maxima.iloc[i]])
            i+=1

# creates LaTeX export
tex_df = pd.read_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/main_output_aggregation.csv")
tex_df["Learning Rate"] = tex_df["Learning Rate"].astype(str)
tex_str = tex_df.to_latex(index = False)
with open("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/main_output_aggregation.tex", "w") as output_tex:
    output_tex.write(tex_str)