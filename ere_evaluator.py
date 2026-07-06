import sys
import ere_dataprep
from third_party.engression.engression import engression
import csv
import pandas as pd
from pipeline.pipeline import horizon
import results.output_info.config as config
from third_party.evaluation.crps_sample import crps_sample
from third_party.evaluation.scoring_rules_supp import es_sample, vs_sample, dss_sample
import numpy as np

# This scripts serves to conduct all experiments on the finished model
# hyperparameters which are independent of baseline and refit model
num_runs = config.num_runs
bs = config.batch_sizes[0] # batch size
lr = config.learning_rates[0] # learning rate
hd = config.hidden_dim[0] # hidden dim
nd = config.noise_dim[0] # noise dim
bn = config.add_bn[0] # add batch normalization
rb = config.resblock[0] # resblock
frequency = config.frequency
# Asset indices, alphabetic order of stocks is provided by the original input builder.
AAPL, BA, CAT, GE, JNJ, JPM, MRK, MSFT, PFE, XOM = range(0,10)
all_assets = [AAPL, BA, CAT, GE, JNJ, JPM, MRK, MSFT, PFE, XOM]

# Optional if the horizon logic in pipeline.py is to be used:
# provides the realized return for single assets to calculate the CRPS
# def true(asset, test_data):
     # true_vals = []
    # for i in range(0,horizon):
        # single_day = test_data[:,asset+i*10]
        # true_vals.append(single_day)
    # return true_vals
# provides sample values for single assets to calculate CRPS
# def pred(asset, samples):
    # pred_vals = []
    # for i in range(0, horizon):
        # single_day_sample = samples[:,asset+i*10,:]
        # pred_vals.append(single_day_sample)
    # return pred_vals
# def crps_horizon(asset, test_data, samples):
    # crps_vals = []
    # asset_true=true(asset, test_data)
    # asset_pred=pred(asset, samples)
    # for i in range(0, horizon):
        # single_day_crps = crps_sample(asset_true[i], asset_pred[i])
        # crps_vals.append(single_day_crps)
    # return crps_vals

# Returns collection of companies' mean crps 
def crps_list(true_vals, pred_vals, asset_list):
    crps_vals = []
    for a in asset_list:
        asset_true = true_vals[:,a]
        asset_pred = pred_vals[:,a,:]
        asset_crps = crps_sample(asset_true, asset_pred)
        asset_crps = np.mean(asset_crps)
        crps_vals.append(asset_crps)
    return crps_vals

# Baseline Model- Evaluation
baseline_el_vals = []
baseline_es_vals = []
baseline_vs_vals = []
baseline_dss_vals = []
baseline_crps_vals = []
for n in range(0, num_runs):
    #  Constructs and runs the training experiment itself.
    baseline = engression(ere_dataprep.X_train_extended, ere_dataprep.Y_train_extended, lr=lr, num_epochs=50, batch_size=bs, hidden_dim=hd, noise_dim=nd, num_layer=10, add_bn=bn, resblock=rb)
    current_el = baseline.eval_loss(ere_dataprep.X_test, ere_dataprep.Y_test, loss_type="energy", sample_size=200)
    baseline_el_vals.append(current_el)
    baseline_samples = baseline.sample(ere_dataprep.X_test, sample_size = 200, expand_dim = True)
    Y_test = ere_dataprep.Y_test.numpy()
    baseline_samples=baseline_samples.numpy()
    baseline_ES = es_sample(Y_test, baseline_samples)
    baseline_es_vals.append(baseline_ES)
    baseline_vs = vs_sample(Y_test, baseline_samples)
    baseline_vs_vals.append(baseline_vs)
    baseline_dss = dss_sample(Y_test, baseline_samples)
    baseline_dss_vals.append(baseline_dss)
    baseline_crps = crps_list(Y_test, baseline_samples, all_assets)
    baseline_crps_vals.append(baseline_crps)
baseline_el = np.median(baseline_el_vals)
baseline_es = np.median(baseline_es_vals)
baseline_vs = np.median(baseline_vs_vals)
baseline_dss = np.median(baseline_dss_vals)
baseline_crps=np.median(baseline_crps_vals, axis = 0)

# Refit Model Evaluation
refit_el_vals = []
refit_es_vals = []
refit_vs_vals = []
refit_dss_vals = []
refit_crps_vals = []
for n in range(0, num_runs):
    refit_samples = []
    temp_el_vals = []
    #  Constructs and runs the refit experiment.
    i=0
    j=len(ere_dataprep.X_train_extended)
    test_len = len(ere_dataprep.X_test)
    while i<test_len:
        refit = engression(ere_dataprep.X_engressable[i:j], ere_dataprep.Y_engressable[i:j], lr = lr, num_epochs= 100, batch_size=bs, hidden_dim = hd, noise_dim = nd, num_layer = 16, add_bn = bn, resblock = rb, verbose = False)
        if i+frequency <= test_len:
            current_el = refit.eval_loss(ere_dataprep.X_test[i:i+frequency], ere_dataprep.Y_test[i:i+frequency], loss_type="energy", sample_size=200)
            refit_samples.append(refit.sample(ere_dataprep.X_test[i:i + frequency], sample_size=200, expand_dim = True).numpy())
            i+=frequency
            j+=frequency
        else:
            current_el = refit.eval_loss(ere_dataprep.X_test[i:test_len], ere_dataprep.Y_test[i:test_len], loss_type="energy", sample_size=200)
            refit_samples.append(refit.sample(ere_dataprep.X_test[i:test_len], sample_size=200, expand_dim=True).numpy())
            i+=frequency
        temp_el_vals.append(current_el)
    el_median = np.median(temp_el_vals)
    refit_el_vals.append(el_median)
    Y_test = ere_dataprep.Y_test.numpy()
    refit_samples = np.concatenate(refit_samples, axis = 0)
    refit_ES = es_sample(Y_test, refit_samples)
    refit_es_vals.append(refit_ES)
    refit_vs = vs_sample(Y_test, refit_samples)
    refit_vs_vals.append(refit_vs)
    refit_dss = dss_sample(Y_test, refit_samples)
    refit_dss_vals.append(refit_dss)
    refit_crps = crps_list(Y_test, refit_samples, all_assets)
    refit_crps_vals.append(refit_crps)
refit_el = np.median(refit_el_vals)
refit_es = np.median(refit_es_vals)
refit_vs = np.median(refit_vs_vals)
refit_dss = np.median(refit_dss_vals)
refit_crps = np.median(refit_crps_vals, axis = 0)

# Ablation refit Model Evaluation
abl_refit_el_vals = []
abl_refit_es_vals = []
abl_refit_vs_vals = []
abl_refit_dss_vals = []
abl_refit_crps_vals = []
for n in range(0, num_runs):
    abl_refit_samples = []
    temp_el_vals = []
    #  Constructs and runs the refit experiment.
    i=0
    j=len(ere_dataprep.X_abl_train_extended)
    test_len = len(ere_dataprep.X_abl_test)
    while i<test_len:
        refit = engression(ere_dataprep.X_abl_engressable[i:j], ere_dataprep.Y_engressable[i:j], lr = lr, num_epochs= 100, batch_size=bs, hidden_dim = hd, noise_dim = nd, num_layer = 16, add_bn = bn, resblock = rb, verbose = False)
        if i+frequency <= test_len:
            current_el = refit.eval_loss(ere_dataprep.X_abl_test[i:i+frequency], ere_dataprep.Y_test[i:i+frequency], loss_type="energy", sample_size=200)
            abl_refit_samples.append(refit.sample(ere_dataprep.X_abl_test[i:i + frequency], sample_size=200, expand_dim = True).numpy())
            i+=frequency
            j+=frequency
        else:
            current_el = refit.eval_loss(ere_dataprep.X_abl_test[i:test_len], ere_dataprep.Y_test[i:test_len], loss_type="energy", sample_size=200)
            abl_refit_samples.append(refit.sample(ere_dataprep.X_abl_test[i:test_len], sample_size=200, expand_dim=True).numpy())
            i+=frequency
        temp_el_vals.append(current_el)
    el_median = np.median(temp_el_vals)
    abl_refit_el_vals.append(el_median)
    Y_test = ere_dataprep.Y_test.numpy()
    abl_refit_samples = np.concatenate(abl_refit_samples, axis = 0)
    abl_refit_ES = es_sample(Y_test, abl_refit_samples)
    abl_refit_es_vals.append(abl_refit_ES)
    abl_refit_vs = vs_sample(Y_test, abl_refit_samples)
    abl_refit_vs_vals.append(abl_refit_vs)
    abl_refit_dss = dss_sample(Y_test, abl_refit_samples)
    abl_refit_dss_vals.append(abl_refit_dss)
    abl_refit_crps = crps_list(Y_test, abl_refit_samples, all_assets)
    abl_refit_crps_vals.append(abl_refit_crps)
abl_refit_el = np.median(abl_refit_el_vals)
abl_refit_es = np.median(abl_refit_es_vals)
abl_refit_vs = np.median(abl_refit_vs_vals)
abl_refit_dss = np.median(abl_refit_dss_vals)
abl_refit_crps = np.median(abl_refit_crps_vals, axis = 0)
