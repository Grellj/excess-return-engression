import sys
import ere_dataprep
from third_party.engression.engression import engression
import csv
import pandas as pd
from pipeline.pipeline import horizon
import results.output_info.config as config
from third_party.evaluation.crps_sample import crps_sample
from third_party.evaluation.scoring_rules_supp import es_sample, vs_sample, dss_sample
# This scripts serves to conduct all experiments on the finished model
num_blocks = config.num_blocks # number training blocks
block_epochs = config.block_epochs # number epochs in one block
bs = config.batch_sizes[0] # batch size
lr = config.learning_rates[0] # learning rate
hd = config.hidden_dim[0] # hidden dim
nd = config.noise_dim[0] # noise dim
nl = config.num_layer[0] # number layers
bn = config.add_bn[0] # add batch normalization
rb = config.resblock[0] # resblock

# basic model run to generate samples
#  Constructs and runs the first block of the training experiment itself.
ere_model = engression(ere_dataprep.X_train, ere_dataprep.Y_train, lr=lr, num_epochs=block_epochs, batch_size=bs, hidden_dim=hd, noise_dim=nd, num_layer=nl, add_bn=bn, resblock=rb)
# Evaluates model performance after first training using the test set
best_e_value = ere_model.eval_loss(ere_dataprep.X_test, ere_dataprep.Y_test, loss_type="energy", verbose=False)
print(best_e_value)
best_samples = ere_model.sample(ere_dataprep.X_test, sample_size = 100, expand_dim = True)
best_block = 1
# subsequent blocks of training and evaluation
for b in range(2, (num_blocks + 1)):
    ere_model.train(ere_dataprep.X_train, ere_dataprep.Y_train, num_epochs=block_epochs, batch_size=bs)
    new_e_value = ere_model.eval_loss(ere_dataprep.X_test, ere_dataprep.Y_test, loss_type="energy", verbose=False)
    new_samples = ere_model.sample(ere_dataprep.X_test, sample_size = 100, expand_dim = True)
    print(new_e_value)
    if new_e_value < best_e_value:
        best_e_value = new_e_value
        best_samples = new_samples
        best_block = b
last_samples = new_samples
last_e_value = new_e_value
best_block_output = [bs, lr, hd, nd, nl, bn, rb, best_e_value, best_block]
Y_test = ere_dataprep.Y_test.numpy()
best_samples=best_samples.numpy()
last_samples = last_samples.numpy()

# Asset indices, alphabetic order of stocks is provided by the original input builder.
AAPL, BA, CAT, GE, JNJ, JPM, MRK, MSFT, PFE, XOM = range(0,10)

# provides the realized return for single assets to calculate the CRPS
def true(asset, test_data):
    true_vals = []
    for i in range(0,horizon):
        single_day = test_data[:,asset+i*10]
        true_vals.append(single_day)
    return true_vals
# provides sample values for single assets to calculate CRPS
def pred(asset, samples):
    pred_vals = []
    for i in range(0, horizon):
        single_day_sample = samples[:,asset+i*10,:]
        pred_vals.append(single_day_sample)
    return pred_vals
def crps_horizon(asset, test_data, samples):
    crps_vals = []
    asset_true=true(asset, test_data)
    asset_pred=pred(asset, samples)
    for i in range(0, horizon):
        single_day_crps = crps_sample(asset_true[i], asset_pred[i])
        crps_vals.append(single_day_crps)
    return crps_vals

ere_model_es = es_sample(Y_test, best_samples)
print("ES: ", ere_model_es)
ere_model_vs = vs_sample(Y_test, best_samples)
print("VS: ", ere_model_vs)
ere_model_dss = dss_sample(Y_test, best_samples)
print("DSS: ", ere_model_dss)