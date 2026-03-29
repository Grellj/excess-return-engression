from pipeline import X_unifier, Y_unifier
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

engression_train_df = engression_dict["train_set"]
X_past, X_std, X_all, X_weekday, Y = engression_builder.fit_prepare(engression_train_df)
X_engressable = X_unifier (X_past, X_weekday, X_std, X_all)
Y_engressable = Y_unifier(Y)
# The tensor length to be used as he basis for splits, wich can be assumed to be identical for X and Y
l = len(X_engressable)
X_train = torch.from_numpy(X_engressable[0:int(0.7*l)])
Y_train = torch.from_numpy(Y_engressable[0:int(0.7*l)])
X_validate = torch.from_numpy(X_engressable[int(0.7*l):int(0.85*l)])
Y_validate = torch.from_numpy(Y_engressable[int(0.7*l):int(0.85*l)])
X_test = torch.from_numpy(X_engressable[int(0.85*l):l])
Y_test = torch.from_numpy(Y_engressable[int(0.85*l):l])
