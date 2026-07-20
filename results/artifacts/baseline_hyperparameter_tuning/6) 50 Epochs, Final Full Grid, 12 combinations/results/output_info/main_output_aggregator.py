import csv
import pandas as pd
import config

# This script serves to summarize the following measurements across all model runs for each hyperparameter combination respectively: 
# the median optimum energy score achieved 
# the mean optimum energy score 
# optionally the maximum amount of training blocks required to reach any energy score optimum 
hidden_dim = config.hidden_dim
noise_dim = config.noise_dim
num_layer = config.num_layer
# df for energy score values
energy_df = pd.DataFrame(index= range(12), columns = range(15))
# df for training blocks in which es optima were achieved, was used during optimization
# number_blocks_df = pd.DataFrame(index= range(12), columns = range(15))
for iteration in range(0, 15):
    energy_df[iteration] = pd.read_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/model_run_" + str(iteration+1) + "/output_info.csv").loc[:, "Validation Energy Loss"]
    # number_blocks_df[iteration] = pd.read_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/model_run_" + str(iteration+1) + "/output_info.csv").iloc[:12, 3]
# assembling the required statistics
energy_medians = energy_df.median(axis=1)
energy_means = energy_df.mean(axis=1)
energy_maxima = energy_df.max(axis=1)
energy_minima = energy_df.min(axis=1)
# block_maxima = number_blocks_df.max(axis=1)

i = 0
# the csv containing the final overview
with open("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/main_output_aggregation.csv", "w", newline="") as main_output_aggregation:
    aggregation_writer = csv.writer(main_output_aggregation)
    aggregation_writer.writerow(["Hidden Dim", "Noise Dim", "Num Layer", "Median ES", "Mean ES", "Min ES", "Max ES"])
    # Note that a compatible loop implementation in main guarantees the correct order of models in the csv files to be aggregated
    for hd in hidden_dim:
        for nd in noise_dim:
            for nl in num_layer:
                aggregation_writer.writerow([hd, nd, nl, energy_medians.iloc[i], energy_means.iloc[i], energy_minima.iloc[i], energy_maxima.iloc[i]])
                i+=1

# creates LaTeX export
tex_df = pd.read_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/main_output_aggregation.csv")
tex_str = tex_df.to_latex(index = False)
with open("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/main_output_aggregation.tex", "w") as output_tex:
    output_tex.write(tex_str)