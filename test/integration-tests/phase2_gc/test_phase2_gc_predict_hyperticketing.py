#!/usr/bin/env python
# coding: utf-8

from hyperticketing.RunHyperTicketing import run_online_filtering
from BasePath import base_path
from utils.utils import SingleConfig

yaml_file = base_path + 'data/yaml/gc_predict.yml'
singleConfig = SingleConfig(yml_file_loc=yaml_file,basepath=base_path)

preproc_config = singleConfig.preproc_config
hyper_config = singleConfig.hyper_config

# params
train = False
preproc_config.master_single_file = True

# assign swarm labels to events
online_filter_obj = run_online_filtering(config=hyper_config)

