#!/usr/bin/env python
# coding: utf-8

from smartpriority.RunSmartPriority import run_online_prediction
from postprocessing.RunPostprocessing import run_postprocessing
from BasePath import base_path
from utils.utils import SingleConfig

yaml_file = base_path + 'data/yaml/gc_predict.yml'
singleConfig = SingleConfig(yml_file_loc=yaml_file,basepath=base_path)

smart_config = singleConfig.smart_config
postproc_config = singleConfig.postproc_config

# params
train = False
# preproc_config.master_single_file = True


# run classification model
online_pred_obj = run_online_prediction(config=smart_config)

master_df_folder = base_path + "data/gc/predict_output/prediction"

# save output to either index or csv
output = run_postprocessing(postproc_config, train, master_df_folder)
