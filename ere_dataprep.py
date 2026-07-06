from pipeline import X_unifier, X_abl_unifier, Y_unifier
from third_party.cgm_method.input_builder import CGMInputBuilder
from third_party.cgm_method.configs import CGMDataConfig, CGMFitConfig
from third_party.data.data_handling import DataHandler
import torch

engression_data_config = CGMDataConfig()
engression_fit_config = CGMFitConfig()
engression_data_handler = DataHandler(engression_data_config.split_point)
engression_dict = engression_data_handler.get_data(
    exclude_pandemic= engression_data_config.exclude_pandemic,
    filter_duplicates= engression_data_config.filter_features
)
engression_builder = CGMInputBuilder(
    window_size = engression_fit_config.train_window_size
)
# Note: train_set is a naming convention inherited from CGM code, here it refers to the entire pre-split dataset
engression_df = engression_dict["train_set"]
X_past, X_std, X_all, X_weekday, X_ablation, Y = engression_builder.fit_prepare(engression_df)
X_engressable = X_unifier (X_past, X_weekday, X_std, X_all)
X_abl_engressable = X_abl_unifier(X_ablation)
Y_engressable = Y_unifier(Y)
# The tensor length to be used as the basis for splits, which can be assumed to be identical for X and Y
l = len(X_engressable)
X_train = torch.from_numpy(X_engressable[0:int(0.8*l)])
X_abl_train = torch.from_numpy(X_abl_engressable[0:int(0.8*l)])
Y_train = torch.from_numpy(Y_engressable[0:int(0.8*l)])
X_train_extended = torch.from_numpy(X_engressable[0:int(0.9*l)])
X_abl_train_extended = torch.from_numpy(X_abl_engressable[0:int(0.9*l)])
Y_train_extended = torch.from_numpy(Y_engressable[0:int(0.9*l)])
X_validate = torch.from_numpy(X_engressable[int(0.8*l):int(0.9*l)])
X_abl_validate = torch.from_numpy(X_abl_engressable[int(0.8*l):int(0.9*l)])
Y_validate = torch.from_numpy(Y_engressable[int(0.8*l):int(0.9*l)])
X_test = torch.from_numpy(X_engressable[int(0.9*l):l])
X_abl_test = torch.from_numpy(X_abl_engressable[int(0.9*l):l])
Y_test = torch.from_numpy(Y_engressable[int(0.9*l):l])
