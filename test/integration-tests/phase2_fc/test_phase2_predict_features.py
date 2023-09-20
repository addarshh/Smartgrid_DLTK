#!/usr/bin/env python
# coding: utf-8

from feature_generation.RunFeatureGeneration import create_feature_data
from BasePath import base_path
from utils.utils import SingleConfig

yaml_file = base_path + 'data/yaml/fc_predict.yml'
singleConfig = SingleConfig(yml_file_loc=yaml_file,basepath=base_path)

preproc_config = singleConfig.preproc_config


# params
train = False
preproc_config.master_single_file = True


# create feature data from master data
feature_df = create_feature_data(config=preproc_config, train=train)

