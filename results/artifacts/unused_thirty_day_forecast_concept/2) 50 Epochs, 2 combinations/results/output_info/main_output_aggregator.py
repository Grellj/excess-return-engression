import csv
import pandas as pd
import config

# This script serves to summarize the following measurements across all model runs for each hyperparameter combination respectively: 
# the median optimum energy score achieved 
# the mean optimum energy score 
# the highest occuring Energy Score
# the lowest occuring Energy Score
# the maximum amount of training blocks required to reach any energy score optimum 
num_runs = config.num_runs
batch_sizes = config.batch_sizes
num_bs = len(batch_sizes)
learning_rates = config.learning_rates
num_lr = len(learning_rates)
hidden_dim = config.hidden_dim
num_hd = len(hidden_dim)
noise_dim = config.noise_dim
num_nd = len(noise_dim)
num_layer = config.num_layer
num_nl = len(num_layer)
add_bn = config.add_bn
num_bn = len(add_bn)
resblock = config.resblock
num_rb = len(resblock)
# df for energy score values
energy_df = pd.DataFrame(index= range(num_bs*num_lr*num_hd*num_nd*num_nl*num_bn*num_rb), columns = range(num_runs))
# df for training blocks in which es optima were achieved, used during optimization
number_blocks_df = pd.DataFrame(index= range(num_bs*num_lr*num_hd*num_nd*num_nl*num_bn*num_rb), columns = range(num_runs))
for iteration in range(0, num_runs):
    energy_df[iteration] = pd.read_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/model_run_" + str(iteration+1) + "/output_info.csv").loc[:, "Validation Energy Loss"]
    number_blocks_df[iteration] = pd.read_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/model_run_" + str(iteration+1) + "/output_info.csv").loc[:, "Achieved in Block"]
# assembling the required statistics
energy_medians = energy_df.median(axis=1)
energy_means = energy_df.mean(axis=1)
energy_maxima = energy_df.max(axis=1)
energy_minima = energy_df.min(axis=1)
block_maxima = number_blocks_df.max(axis=1)

i = 0
# the csv containing the final overview
with open("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/main_output_aggregation.csv", "w", newline="") as main_output_aggregation:
    aggregation_writer = csv.writer(main_output_aggregation)
    aggregation_writer.writerow(["Batch Size", "LR", "Hidden Dim", "Noise Dim", "Num Layer", "Add BN", "Resblock", "Median ES", "Mean ES", "Min ES", "Max ES", "Max Blocks"])
    # Note that a compatible loop implementation in main guarantees the correct order of models in the csv files to be aggregated
    for bs in batch_sizes:
        for lr in learning_rates:
            for hd in hidden_dim:
                for nd in noise_dim:
                    for nl in num_layer:
                        for bn in add_bn:
                            for rb in resblock:
                                aggregation_writer.writerow([bs, lr, hd, nd, nl, bn, rb, energy_medians.iloc[i], energy_means.iloc[i], energy_minima.iloc[i], energy_maxima.iloc[i], block_maxima.iloc[i]])
                                i+=1

# creates LaTeX export
tex_df = pd.read_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/main_output_aggregation.csv")
tex_str = tex_df.to_latex(index = False)
with open("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/main_output_aggregation.tex", "w") as output_tex:
    output_tex.write(tex_str)