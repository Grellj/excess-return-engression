import sys
import ere_dataprep
from third_party.engression.engression import engression
import numpy as np
import csv
import optuna
import pandas as pd
import config

# This script serves to optimize the following hyperparameters: num_layer, hidden_dim, noise_dim, add_bn and resblock.

train_len = len(ere_dataprep.X_train)
validation_len = len(ere_dataprep.X_validate)
ne = config.num_epochs[0] # number of epochs
frequency = config.frequency
num_trials = config.num_opt_trials
batch_sizes = config.batch_sizes
learning_rates = config.learning_rates
num_layer=config.num_layer
hidden_dim=config.hidden_dim
noise_dim=config.noise_dim
add_bn = config.add_bn
resblock = config.resblock

def objective (trial):
    bs = trial.suggest_categorical("batch_sizes", batch_sizes)
    lr = trial.suggest_categorical("learning_rates", learning_rates)
    nl = trial.suggest_categorical("num_layer", num_layer)
    hd = trial.suggest_categorical("hidden_dim", hidden_dim)
    nd = trial.suggest_categorical("noise_dim", noise_dim)
    bn = trial.suggest_categorical("add_bn", add_bn)
    rb = trial.suggest_categorical("resblock", resblock)
    #  Constructs and runs the first stage of the training experiment itself.
    i=0
    j=train_len
    ere_model = engression(ere_dataprep.X_train, ere_dataprep.Y_train, lr = lr, num_epochs= ne, batch_size=bs, hidden_dim = hd, noise_dim = nd, num_layer = nl, add_bn = bn, resblock = rb, verbose = False)
    # Evaluates model performance after first training using the first forecast frequency of the validation set
    current_e_value = ere_model.eval_loss(ere_dataprep.X_validate[i:i+frequency], ere_dataprep.Y_validate[i:i+frequency], loss_type="energy", sample_size=10)
    print(current_e_value)
    e_values = [current_e_value]
    i+=frequency
    j+=frequency
    # subsequent stages of training and evaluation
    while i<validation_len:
        ere_model = engression(ere_dataprep.X_train_extended[i:j], ere_dataprep.Y_train_extended[i:j], lr = lr, num_epochs= ne, batch_size=bs, hidden_dim = hd, noise_dim = nd, num_layer = nl, add_bn = bn, resblock = rb, verbose = False)
        if i+frequency <= validation_len:
            current_e_value = ere_model.eval_loss(ere_dataprep.X_validate[i:i+frequency], ere_dataprep.Y_validate[i:i+frequency], loss_type="energy", sample_size=10)
            i+=frequency
            j+=frequency
            print(current_e_value)
        else:
            current_e_value = ere_model.eval_loss(ere_dataprep.X_validate[i:validation_len], ere_dataprep.Y_validate[i:validation_len], loss_type="energy", sample_size=10)
            print(current_e_value)
            i+=frequency
        e_values.append(current_e_value)
    energy_median = np.median(e_values)
    return energy_median
ere_study = optuna.create_study(direction = "minimize")
ere_study.optimize(objective, n_trials = num_trials)
ere_opt_df=ere_study.trials_dataframe()
ere_opt_df.to_csv("results/output_info/hyperparameter_tuning/refit_opt.csv")

# The following code serves to filter the best optuna trials and display their properties

optuna_df = pd.read_csv("results/output_info/hyperparameter_tuning/refit_opt.csv")
optuna_df = optuna_df[["value","params_add_bn","params_hidden_dim","params_noise_dim","params_num_layer","params_resblock"]]
optuna_df = optuna_df.sort_values("value", ascending = True).head(5)
# the csv containing the final overview
optuna_df.to_csv("results/output_info/hyperparameter_tuning/refit_opt_aggregation.csv", index = False)
# creates LaTeX export
tex_str = optuna_df.to_latex(index = False)
with open("results/output_info/hyperparameter_tuning/refit_opt_aggregation.tex", "w") as output_tex:
    output_tex.write(tex_str)