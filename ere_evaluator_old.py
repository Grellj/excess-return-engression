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

# This script serves to conduct all experiments on the finished model
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

# mean score evaluation of models, aggregating the median of means across 5 repetitions for each model
# Baseline Model- Evaluation
baseline_el_vals = [] # Engression-specific Energy Score/Energy Loss which was also used for hyperparameter tuning
baseline_es_vals = [] # mean ES as used for scoring
baseline_vs_vals = [] # Variogram Scores
baseline_dss_vals = [] # Dawid Sebastiani Scores 
baseline_crps_vals = [] # Continuous Ranked Probability Scores, in this case a list of lists 
# Repetitions to achieve robust results
for n in range(0, num_runs):
    #  Constructs and runs the training experiment itself.
    baseline = engression(ere_dataprep.X_train_extended, ere_dataprep.Y_train_extended, lr=lr, num_epochs=50, batch_size=bs, hidden_dim=hd, noise_dim=nd, num_layer=10, add_bn=bn, resblock=rb)
    current_el = baseline.eval_loss(ere_dataprep.X_test, ere_dataprep.Y_test, loss_type="energy", sample_size=200)
    baseline_el_vals.append(current_el)
    baseline_samples = baseline.sample(ere_dataprep.X_test, sample_size = 200, expand_dim = True)
    Y_test = ere_dataprep.Y_test.numpy()
    baseline_samples=baseline_samples.numpy()
    baseline_es = es_sample(Y_test, baseline_samples)
    baseline_es_vals.append(baseline_es)
    baseline_vs = vs_sample(Y_test, baseline_samples)
    baseline_vs_vals.append(baseline_vs)
    baseline_dss = dss_sample(Y_test, baseline_samples)
    baseline_dss_vals.append(baseline_dss)
    baseline_crps = crps_list(Y_test, baseline_samples, all_assets)
    baseline_crps_vals.append(baseline_crps)
# aggregates performance across all model runs
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
    refit_es = es_sample(Y_test, refit_samples)
    refit_es_vals.append(refit_es)
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
ablation_el_vals = []
ablation_es_vals = []
ablation_vs_vals = []
ablation_dss_vals = []
ablation_crps_vals = []
for n in range(0, num_runs):
    ablation_samples = []
    temp_el_vals = []
    #  Constructs and runs the refit experiment.
    i=0
    j=len(ere_dataprep.X_abl_train_extended)
    test_len = len(ere_dataprep.X_abl_test)
    while i<test_len:
        refit = engression(ere_dataprep.X_abl_engressable[i:j], ere_dataprep.Y_engressable[i:j], lr = lr, num_epochs= 100, batch_size=bs, hidden_dim = hd, noise_dim = nd, num_layer = 16, add_bn = bn, resblock = rb, verbose = False)
        if i+frequency <= test_len:
            current_el = refit.eval_loss(ere_dataprep.X_abl_test[i:i+frequency], ere_dataprep.Y_test[i:i+frequency], loss_type="energy", sample_size=200)
            ablation_samples.append(refit.sample(ere_dataprep.X_abl_test[i:i + frequency], sample_size=200, expand_dim = True).numpy())
            i+=frequency
            j+=frequency
        else:
            current_el = refit.eval_loss(ere_dataprep.X_abl_test[i:test_len], ere_dataprep.Y_test[i:test_len], loss_type="energy", sample_size=200)
            ablation_samples.append(refit.sample(ere_dataprep.X_abl_test[i:test_len], sample_size=200, expand_dim=True).numpy())
            i+=frequency
        temp_el_vals.append(current_el)
    el_median = np.median(temp_el_vals)
    ablation_el_vals.append(el_median)
    Y_test = ere_dataprep.Y_test.numpy()
    ablation_samples = np.concatenate(ablation_samples, axis = 0)
    ablation_es = es_sample(Y_test, ablation_samples)
    ablation_es_vals.append(ablation_es)
    ablation_vs = vs_sample(Y_test, ablation_samples)
    ablation_vs_vals.append(ablation_vs)
    ablation_dss = dss_sample(Y_test, ablation_samples)
    ablation_dss_vals.append(ablation_dss)
    ablation_crps = crps_list(Y_test, ablation_samples, all_assets)
    ablation_crps_vals.append(ablation_crps)
ablation_el = np.median(ablation_el_vals)
ablation_es = np.median(ablation_es_vals)
ablation_vs = np.median(ablation_vs_vals)
ablation_dss = np.median(ablation_dss_vals)
ablation_crps = np.median(ablation_crps_vals, axis = 0)

# Model runs without repetition for daily ES, VS and DSS behavior
# Baseline Model Daily Evaluation
#  Constructs and runs the training experiment itself.
baseline = engression(ere_dataprep.X_train_extended, ere_dataprep.Y_train_extended, lr=lr, num_epochs=50, batch_size=bs, hidden_dim=hd, noise_dim=nd, num_layer=10, add_bn=bn, resblock=rb)
baseline_daily_samples = baseline.sample(ere_dataprep.X_test, sample_size =1000, expand_dim = True)
Y_test = ere_dataprep.Y_test.numpy()
baseline_daily_samples=baseline_daily_samples.numpy()
# Aggregation of ES behavior
baseline_daily_es_vals = es_sample(Y_test, baseline_daily_samples, return_single_scores=True)
baseline_daily_es = baseline_daily_es_vals[1] # removes pre-calculated mean 
baseline_daily_es_min = np.min(baseline_daily_es)
baseline_daily_es_1_percentile = np.percentile(baseline_daily_es, 1)
baseline_daily_es_median = np.median(baseline_daily_es)
baseline_daily_es_mean = np.mean(baseline_daily_es)
baseline_daily_es_std = np.std(baseline_daily_es)
baseline_daily_es_99_percentile = np.percentile(baseline_daily_es, 99)
baseline_daily_es_max = np.max(baseline_daily_es)
baseline_daily_es_range = baseline_daily_es_max - baseline_daily_es_min
# Aggregation of VS behavior
baseline_daily_vs_vals = vs_sample(Y_test, baseline_daily_samples, return_single_scores=True)
baseline_daily_vs = baseline_daily_vs_vals[1]
baseline_daily_vs_min = np.min(baseline_daily_vs)
baseline_daily_vs_1_percentile = np.percentile(baseline_daily_vs, 1)
baseline_daily_vs_median = np.median(baseline_daily_vs)
baseline_daily_vs_mean = np.mean(baseline_daily_vs)
baseline_daily_vs_std = np.std(baseline_daily_vs)
baseline_daily_vs_99_percentile = np.percentile(baseline_daily_vs, 99)
baseline_daily_vs_max = np.max(baseline_daily_vs)
baseline_daily_vs_range = baseline_daily_vs_max - baseline_daily_vs_min
# Aggregation of DSS behavior
baseline_daily_dss_vals = dss_sample(Y_test, baseline_daily_samples, return_single_scores=True)
baseline_daily_dss = baseline_daily_dss_vals[1]
baseline_daily_dss_min = np.min(baseline_daily_dss)
baseline_daily_dss_1_percentile = np.percentile(baseline_daily_dss, 1)
baseline_daily_dss_median = np.median(baseline_daily_dss)
baseline_daily_dss_mean = np.mean(baseline_daily_dss)
baseline_daily_dss_std = np.std(baseline_daily_dss)
baseline_daily_dss_99_percentile = np.percentile(baseline_daily_dss, 99)
baseline_daily_dss_max = np.max(baseline_daily_dss)
baseline_daily_dss_range = baseline_daily_dss_max - baseline_daily_dss_min

# Refit Model Daily Evaluation
refit_daily_samples = []
#  Constructs and runs the refit experiment.
i=0
j=len(ere_dataprep.X_train_extended)
test_len = len(ere_dataprep.X_test)
while i<test_len:
    refit = engression(ere_dataprep.X_engressable[i:j], ere_dataprep.Y_engressable[i:j], lr = lr, num_epochs= 100, batch_size=bs, hidden_dim = hd, noise_dim = nd, num_layer = 16, add_bn = bn, resblock = rb, verbose = False)
    if i+frequency <= test_len:
        refit_daily_samples.append(refit.sample(ere_dataprep.X_test[i:i + frequency], sample_size=1000, expand_dim = True).numpy())
        i+=frequency
        j+=frequency
    else:
        refit_daily_samples.append(refit.sample(ere_dataprep.X_test[i:test_len], sample_size=1000, expand_dim=True).numpy())
        i+=frequency
Y_test = ere_dataprep.Y_test.numpy()
refit_daily_samples = np.concatenate(refit_daily_samples, axis = 0)
# ES behavior
refit_daily_es_vals = es_sample(Y_test, refit_daily_samples, return_single_scores=True)
refit_daily_es = refit_daily_es_vals[1]
refit_daily_es_min = np.min(refit_daily_es)
refit_daily_es_1_percentile = np.percentile(refit_daily_es, 1)
refit_daily_es_median = np.median(refit_daily_es)
refit_daily_es_mean = np.mean(refit_daily_es)
refit_daily_es_std = np.std(refit_daily_es)
refit_daily_es_99_percentile = np.percentile(refit_daily_es, 99)
refit_daily_es_max = np.max(refit_daily_es)
refit_daily_es_range = refit_daily_es_max - refit_daily_es_min
# VS behavior
refit_daily_vs_vals = vs_sample(Y_test, refit_daily_samples, return_single_scores=True)
refit_daily_vs = refit_daily_vs_vals[1]
refit_daily_vs_min = np.min(refit_daily_vs)
refit_daily_vs_1_percentile = np.percentile(refit_daily_vs, 1)
refit_daily_vs_median = np.median(refit_daily_vs)
refit_daily_vs_mean = np.mean(refit_daily_vs)
refit_daily_vs_std = np.std(refit_daily_vs)
refit_daily_vs_99_percentile = np.percentile(refit_daily_vs, 99)
refit_daily_vs_max = np.max(refit_daily_vs)
refit_daily_vs_range = refit_daily_vs_max - refit_daily_vs_min
# DSS behavior
refit_daily_dss_vals = dss_sample(Y_test, refit_daily_samples, return_single_scores=True)
refit_daily_dss = refit_daily_dss_vals[1]
refit_daily_dss_min = np.min(refit_daily_dss)
refit_daily_dss_1_percentile = np.percentile(refit_daily_dss, 1)
refit_daily_dss_median = np.median(refit_daily_dss)
refit_daily_dss_mean = np.mean(refit_daily_dss)
refit_daily_dss_std = np.std(refit_daily_dss)
refit_daily_dss_99_percentile = np.percentile(refit_daily_dss, 99)
refit_daily_dss_max = np.max(refit_daily_dss)
refit_daily_dss_range = refit_daily_dss_max - refit_daily_dss_min

# Ablation refit Model Daily Evaluation
ablation_daily_samples = []
#  Constructs and runs the refit experiment.
i=0
j=len(ere_dataprep.X_abl_train_extended)
test_len = len(ere_dataprep.X_abl_test)
while i<test_len:
    refit = engression(ere_dataprep.X_abl_engressable[i:j], ere_dataprep.Y_engressable[i:j], lr = lr, num_epochs= 100, batch_size=bs, hidden_dim = hd, noise_dim = nd, num_layer = 16, add_bn = bn, resblock = rb, verbose = False)
    if i+frequency <= test_len:
        ablation_daily_samples.append(refit.sample(ere_dataprep.X_abl_test[i:i + frequency], sample_size=1000, expand_dim = True).numpy())
        i+=frequency
        j+=frequency
    else:
        ablation_daily_samples.append(refit.sample(ere_dataprep.X_abl_test[i:test_len], sample_size=1000, expand_dim=True).numpy())
        i+=frequency
Y_test = ere_dataprep.Y_test.numpy()
ablation_daily_samples = np.concatenate(ablation_daily_samples, axis = 0)
# ES behavior
ablation_daily_es_vals = es_sample(Y_test, ablation_daily_samples, return_single_scores=True)
ablation_daily_es = ablation_daily_es_vals[1]
ablation_daily_es_min = np.min(ablation_daily_es)
ablation_daily_es_1_percentile = np.percentile(ablation_daily_es, 1)
ablation_daily_es_median = np.median(ablation_daily_es)
ablation_daily_es_mean = np.mean(ablation_daily_es)
ablation_daily_es_std = np.std(ablation_daily_es)
ablation_daily_es_99_percentile = np.percentile(ablation_daily_es, 99)
ablation_daily_es_max = np.max(ablation_daily_es)
ablation_daily_es_range = ablation_daily_es_max - ablation_daily_es_min
# VS behavior
ablation_daily_vs_vals = vs_sample(Y_test, ablation_daily_samples, return_single_scores=True)
ablation_daily_vs = ablation_daily_vs_vals[1]
ablation_daily_vs_min = np.min(ablation_daily_vs)
ablation_daily_vs_1_percentile = np.percentile(ablation_daily_vs, 1)
ablation_daily_vs_median = np.median(ablation_daily_vs)
ablation_daily_vs_mean = np.mean(ablation_daily_vs)
ablation_daily_vs_std = np.std(ablation_daily_vs)
ablation_daily_vs_99_percentile = np.percentile(ablation_daily_vs, 99)
ablation_daily_vs_max = np.max(ablation_daily_vs)
ablation_daily_vs_range = ablation_daily_vs_max - ablation_daily_vs_min
# DSS behavior
ablation_daily_dss_vals = dss_sample(Y_test, ablation_daily_samples, return_single_scores=True)
ablation_daily_dss = ablation_daily_dss_vals[1]
ablation_daily_dss_min = np.min(ablation_daily_dss)
ablation_daily_dss_1_percentile = np.percentile(ablation_daily_dss, 1)
ablation_daily_dss_median = np.median(ablation_daily_dss)
ablation_daily_dss_mean = np.mean(ablation_daily_dss)
ablation_daily_dss_std = np.std(ablation_daily_dss)
ablation_daily_dss_99_percentile = np.percentile(ablation_daily_dss, 99)
ablation_daily_dss_max = np.max(ablation_daily_dss)
ablation_daily_dss_range = ablation_daily_dss_max - ablation_daily_dss_min
# Creates and saves the table comparing the two measurements of the Energy Score for each Engression model
el_es_comparison = pd.DataFrame(data=[[baseline_el, baseline_es],[refit_el, refit_es],[ablation_el, ablation_es]], columns = ["EL", "ES"], index=["Baseline Model", "Refit Model", "Ablation Refit Model"])
el_es_comparison.to_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/el_es_comparison.csv")
el_es_comparison.to_latex("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/el_es_comparison.tex")

# displays the comparison of all mean scores for each of the models
mean_performance_comparison = pd.DataFrame(data = [[baseline_es, baseline_vs, baseline_dss], [refit_es, refit_vs, refit_dss], [ablation_es, ablation_vs, ablation_dss]], columns = ["ES", "VS", "DSS"], index = ["Baseline Model", "Refit Model", "Ablation Refit Model"])
mean_performance_comparison.to_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/mean_performance_comparison.csv")
mean_performance_comparison.to_latex("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/mean_performance_comparison.tex")

# displays comparison of mean crps for all models and all assets
mean_crps_comparison = pd.DataFrame(data = {"Baseline Model": baseline_crps, "Refit Model": refit_crps, "Ablation Refit Model": ablation_crps}, index = ["AAPL", "BA", "CAT", "GE", "JNJ", "JPM", "MRK", "MSFT", "PFE", "XOM"])
mean_crps_comparison.to_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/mean_crps_comparison.csv")
mean_crps_comparison.to_latex("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/mean_crps_comparison.tex")

# Saves specific ES values on which the aggregation of behavior was conducted
daily_es_vals = pd.DataFrame(data = {"Baseline Model ES": baseline_daily_es, "Refit Model ES": refit_daily_es, "Ablation Refit Model ES": ablation_daily_es})
daily_es_vals.to_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/daily_es_vals.csv")

# displays comparison of aggregated ES behavior across models
daily_es_aggregation = pd.DataFrame(data = {"Baseline Model ES": [baseline_daily_es_min, baseline_daily_es_1_percentile, baseline_daily_es_median, baseline_daily_es_mean, baseline_daily_es_std, baseline_daily_es_99_percentile, baseline_daily_es_max, baseline_daily_es_range], "Refit Model ES": [refit_daily_es_min, refit_daily_es_1_percentile, refit_daily_es_median, refit_daily_es_mean, refit_daily_es_std, refit_daily_es_99_percentile, refit_daily_es_max, refit_daily_es_range], "Ablation Refit Model ES": [ablation_daily_es_min, ablation_daily_es_1_percentile, ablation_daily_es_median, ablation_daily_es_mean, ablation_daily_es_std, ablation_daily_es_99_percentile, ablation_daily_es_max, ablation_daily_es_range]}, index = ["Min", "1%-Percentile", "Median", "Mean", "STD", "99%-Percentile", "Max", "Range"])
daily_es_aggregation.to_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/daily_es_aggregation.csv")
daily_es_aggregation.to_latex("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/daily_es_aggregation.tex")

# Saves specific VS values across which daily behavior was aggregated
daily_vs_vals = pd.DataFrame(data = {"Baseline Model VS": baseline_daily_vs, "Refit Model VS": refit_daily_vs, "Ablation Refit Model VS": ablation_daily_vs})
daily_vs_vals.to_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/daily_vs_vals.csv")
# Displays VS daily behavior
daily_vs_aggregation = pd.DataFrame(data = {"Baseline Model VS": [baseline_daily_vs_min, baseline_daily_vs_1_percentile, baseline_daily_vs_median, baseline_daily_vs_mean, baseline_daily_vs_std, baseline_daily_vs_99_percentile, baseline_daily_vs_max, baseline_daily_vs_range], "Refit Model VS": [refit_daily_vs_min, refit_daily_vs_1_percentile, refit_daily_vs_median, refit_daily_vs_mean, refit_daily_vs_std, refit_daily_vs_99_percentile, refit_daily_vs_max, refit_daily_vs_range], "Ablation Refit Model VS": [ablation_daily_vs_min, ablation_daily_vs_1_percentile, ablation_daily_vs_median, ablation_daily_vs_mean, ablation_daily_vs_std, ablation_daily_vs_99_percentile, ablation_daily_vs_max, ablation_daily_vs_range]}, index = ["Min", "1%-Percentile", "Median", "Mean", "STD", "99%-Percentile", "Max", "Range"])
daily_vs_aggregation.to_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/daily_vs_aggregation.csv")
daily_vs_aggregation.to_latex("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/daily_vs_aggregation.tex")

# Saves specific DSS values across which daily behavior was aggregated
daily_dss_vals = pd.DataFrame(data = {"Baseline Model DSS": baseline_daily_dss, "Refit Model DSS": refit_daily_dss, "Ablation Refit Model DSS": ablation_daily_dss})
daily_dss_vals.to_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/daily_dss_vals.csv")
# Displays aggregated daily DSS behavior
daily_dss_aggregation = pd.DataFrame(data = {"Baseline Model DSS": [baseline_daily_dss_min, baseline_daily_dss_1_percentile, baseline_daily_dss_median, baseline_daily_dss_mean, baseline_daily_dss_std, baseline_daily_dss_99_percentile, baseline_daily_dss_max, baseline_daily_dss_range], "Refit Model DSS": [refit_daily_dss_min, refit_daily_dss_1_percentile, refit_daily_dss_median, refit_daily_dss_mean, refit_daily_dss_std, refit_daily_dss_99_percentile, refit_daily_dss_max, refit_daily_dss_range], "Ablation Refit Model DSS": [ablation_daily_dss_min, ablation_daily_dss_1_percentile, ablation_daily_dss_median, ablation_daily_dss_mean, ablation_daily_dss_std, ablation_daily_dss_99_percentile, ablation_daily_dss_max, ablation_daily_dss_range]}, index = ["Min", "1%-Percentile", "Median", "Mean", "STD", "99%-Percentile", "Max", "Range"])
daily_dss_aggregation.to_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/daily_dss_aggregation.csv")
daily_dss_aggregation.to_latex("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/daily_dss_aggregation.tex")

# Displays aggregated comparison of ES values across repeated model runs
robustness_check_es = pd.DataFrame(data = [[np.min(baseline_es_vals), np.median(baseline_es_vals), np.max(baseline_es_vals)], [np.min(refit_es_vals), np.median(refit_es_vals), np.max(refit_es_vals)], [np.min(ablation_es_vals), np.median(ablation_es_vals), np.max(ablation_es_vals)]], columns = ["Min", "Median", "Max"], index = ["Baseline Model ES", "Refit Model ES", "Ablation Refit Model ES"])
robustness_check_es.to_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/robustness_check_es.csv")
robustness_check_es.to_latex("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/robustness_check_es.tex")

# Displays aggregated comparison of VS values across repeated model runs
robustness_check_vs = pd.DataFrame(data = [[np.min(baseline_vs_vals), np.median(baseline_vs_vals), np.max(baseline_vs_vals)], [np.min(refit_vs_vals), np.median(refit_vs_vals), np.max(refit_vs_vals)], [np.min(ablation_vs_vals), np.median(ablation_vs_vals), np.max(ablation_vs_vals)]], columns = ["Min", "Median", "Max"], index = ["Baseline Model VS", "Refit Model VS", "Ablation Refit Model VS"])
robustness_check_vs.to_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/robustness_check_vs.csv")
robustness_check_vs.to_latex("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/robustness_check_vs.tex")

# Displays aggregated comparison of DSS values across repeated model runs
robustness_check_dss = pd.DataFrame(data = [[np.min(baseline_dss_vals), np.median(baseline_dss_vals), np.max(baseline_dss_vals)], [np.min(refit_dss_vals), np.median(refit_dss_vals), np.max(refit_dss_vals)], [np.min(ablation_dss_vals), np.median(ablation_dss_vals), np.max(ablation_dss_vals)]], columns = ["Min", "Median", "Max"], index = ["Baseline Model DSS", "Refit Model DSS", "Ablation Refit Model DSS"])
robustness_check_dss.to_csv("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/robustness_check_dss.csv")
robustness_check_dss.to_latex("C:/Users/johan/Documents/Uni/Semester 15 und 16/Bachelorarbeit/excess-return-engression/results/output_info/final_evaluation/robustness_check_dss.tex")





# Optional code in case the horizon logic in pipeline.py is to be used:
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
