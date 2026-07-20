import csv
import pandas as pd

# This script serves to filter the top 5 best optuna trials and display their properties

optuna_df = pd.read_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/ere_opt.csv")
optuna_df = optuna_df[["value","params_add_bn","params_hidden_dim","params_noise_dim","params_num_layer","params_resblock"]]
optuna_df = optuna_df.sort_values("value", ascending = True).head(5)
# the csv containing the final overview
optuna_df.to_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/optuna_output_aggregation.csv", index = False)
# creates LaTeX export
tex_str = optuna_df.to_latex(index = False)
with open("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/optuna_output_aggregation.tex", "w") as output_tex:
    output_tex.write(tex_str)