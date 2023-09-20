# !/usr/bin/env python
# coding: utf-8

from __future__ import division
from hyperticketing.OnlineFiltering import OnlineFiltering
from hyperticketing.TrainPSO import PSO
from utils.utils import get_latest_folder

def run_pso(config):
    """
    : Run Hyperticketing Particle Swarm Optimization training
    """

    # Read master df
    config.train['filepaths']["master_df_name"] = config.train['filepaths']["master_df_folder_dir"] + config.train['filepaths']["master_df_name"]
    print('Getting Master DF from ', config.train['filepaths']["master_df_name"])

    # Run PSO
    pso_filter = PSO(config)
    output_folder_path = pso_filter.save_files(path=config.train['filepaths']["saved_model_path"])

    return output_folder_path


def run_online_filtering(config):
    """
    : Run Hyperticketing inference
    """

    print('Reading model from ', config.online["filepaths"]["saved_model_folder_dir"])
    online_filtering_obj = OnlineFiltering(config.online["filepaths"]["saved_model_folder_dir"])

    # Run online filtering
    try:
        config.online["filepaths"]["existing_swarm"] = get_latest_folder("latest", config.online["filepaths"]["save_swarm_dir"]) + "/swarm_label.csv"
    except:
        pass
    print('Reading existing swarm from ', config.online["filepaths"]["existing_swarm"])
    online_filtering_obj.filtering_even_faster(config.online["filepaths"]["existing_swarm"],
                                               config.online["filepaths"]["master_df_name"])
    online_filtering_obj.save_output_files(path=config.online["filepaths"]["save_swarm_dir"])

    return online_filtering_obj
