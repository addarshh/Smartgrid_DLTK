#!/usr/bin/env python
# coding: utf-8

from smartpriority.RunSmartPriority import run_online_prediction
from postprocessing.RunPostprocessing import run_postprocessing
from BasePath import base_path
from utils.utils import SingleConfig

yaml_file = base_path + 'data/yaml/fc_predict.yml'
singleConfig = SingleConfig(yml_file_loc=yaml_file,basepath=base_path)

smart_config = singleConfig.smart_config
postproc_config = singleConfig.postproc_config


# run classification model
online_pred_obj = run_online_prediction(config=smart_config)

# save output to either index or csv
master_df_folder = base_path + "data/fc/predict_output/prediction"
output = run_postprocessing(postproc_config, train=False, master_df_folder=master_df_folder)

print("Finished.")
