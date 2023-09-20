#!/usr/bin/env python
# coding: utf-8

from preprocessing.RunPreprocessing import create_master_data
from feature_generation.RunFeatureGeneration import create_feature_data
from hyperticketing.RunHyperTicketing import run_online_filtering
from smartpriority.RunSmartPriority import run_online_prediction
from postprocessing.RunPostprocessing import run_postprocessing
from BasePath import base_path
from utils.utils import SingleConfig

yaml_file = base_path + 'data/yaml/gc_predict.yml'
singleConfig = SingleConfig(yml_file_loc=yaml_file,basepath=base_path)

preproc_config = singleConfig.preproc_config
hyper_config = singleConfig.hyper_config
smart_config = singleConfig.smart_config
postproc_config = singleConfig.postproc_config

# params
train = False
preproc_config.master_single_file = True


# create master data from data sources
master_df_folder = create_master_data(config=preproc_config, train=train)

# assign swarm labels to events
online_filter_obj = run_online_filtering(config=hyper_config)

# create feature data from master data
feature_df = create_feature_data(config=preproc_config, train=train)

# run classification model
online_pred_obj = run_online_prediction(config=smart_config)

# save output to either index or csv
output = run_postprocessing(postproc_config, train, master_df_folder)
