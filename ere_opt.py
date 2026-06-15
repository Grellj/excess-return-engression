import sys
import ere_dataprep
from third_party.engression.engression import engression
import csv
import optuna
import pandas as pd
import results.output_info.config

# This script serves to optimize the following hyperparameters: num_layer, hidden_dim, noise_dim, add_bn and resblock.

num_blocks = config.num_blocks
block_epochs = config.block_epochs
bs = config.batch_sizes[0]
l = config.learning_rates[0]
num_layer=config.num_layer
hidden_dim=config.hidden_dim
noise_dim=config.noise_dim
add_bn = config.add_bn
resblock = config.resblock

def objective (trial):
    nl = trial.suggest_categorical("num_layer", num_layer)
    hd = trial.suggest_categorical("hidden_dim", hidden_dim)
    nd = trial.suggest_categorical("noise_dim", noise_dim)
    bn = trial.suggest_categorical("add_bn", add_bn)
    rb = trial.suggest_categorical("resblock", resblock)
    #  Constructs and runs the first block of the training experiment itself.
    ere_model = engression(ere_dataprep.X_train, ere_dataprep.Y_train, lr=l, num_epochs=block_epochs, batch_size=bs, num_layer=nl, hidden_dim=hd, noise_dim=nd, add_bn=bn, resblock=rb, verbose=False)
    # Evaluates model performance after first training using the validation set
    current_e_value = ere_model.eval_loss(ere_dataprep.X_validate, ere_dataprep.Y_validate, loss_type="energy", verbose=False)
    current_block = 1
    # subsequent blocks of training and evaluation
    for b in range(2, (num_blocks + 1)):
        ere_model.train(ere_dataprep.X_train, ere_dataprep.Y_train, num_epochs=block_epochs, batch_size=bs, verbose=False)
        new_e_value = ere_model.eval_loss(ere_dataprep.X_validate, ere_dataprep.Y_validate, loss_type="energy", verbose=False)
        if new_e_value < current_e_value:
            current_e_value = new_e_value
            current_block = b
    return current_e_value
ere_study = optuna.create_study(direction = "minimize")
ere_study.optimize(objective, 128)
ere_opt_df=ere_study.trials_dataframe()
ere_opt_df.to_csv("results/output_info/ere_opt.csv")