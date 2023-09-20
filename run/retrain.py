#!/usr/bin/env python
# coding: utf-8

from preprocessing.RunPreprocessing import create_master_data
from feature_generation.RunFeatureGeneration import create_feature_data
from hyperticketing.RunHyperTicketing import run_pso
from smartpriority.RunSmartPriority import train_smart_priority_model
from postprocessing.RunPostprocessing import run_postprocessing
from BasePath import base_path
from utils.utils import SingleConfig
import sg_splunk
from datetime import datetime, timedelta
import os


def pull_previous_master_df(out_dir, device_type):
    """
    : Pull syslog data for ion3k and global core
    """

    yml = base_path + "data/yaml/input_data_config.yml"
    sg_splunk.set_config(yml)

    # MasterDF
    sg_splunk.set_credentials("ah-1064594-001.sdi.corp")
    sg_splunk.pull_data(data_type="master_df",
                        output_dir=out_dir,
                        timing=False,
                        device_type=device_type,
                        name="master_df")

    return


def run_train(device_type):
    """
    : Run prediction steps
    """

    if device_type == "ion3k":
        yaml_file = base_path + 'data/yaml/fc_train.yml'
        now = datetime.now().strftime('%Y-%m-%d_%H%M%S' + "/")
        master_df_folder = base_path + "data/fc/train_output/retrain/" + now
        os.makedirs(master_df_folder, exist_ok=True)
        pull_previous_master_df(master_df_folder, 'ion3k')

    if device_type == "global_core":
        yaml_file = base_path + 'data/yaml/gc_train.yml'
        now = datetime.now().strftime('%Y-%m-%d_%H%M%S' + "/")
        master_df_folder = base_path + "data/gc/train_output/retrain/" + now
        os.makedirs(master_df_folder, exist_ok=True)
        pull_previous_master_df(master_df_folder, 'global_core')


    singleConfig = SingleConfig(yml_file_loc=yaml_file, basepath=base_path)

    preproc_config = singleConfig.preproc_config
    hyper_config = singleConfig.hyper_config
    smart_config = singleConfig.smart_config
    postproc_config = singleConfig.postproc_config


    # Hyper Ticketing
    hyper_config.train['filepaths']["master_df_folder_dir"] = master_df_folder
    hyper_config.train['filepaths']["saved_model_path"] = master_df_folder
    swarm_label_folder = run_pso(config=hyper_config)

    # Feature Generation
    preproc_config.feature['filepaths']["train_pso_data_dir"] = swarm_label_folder
    preproc_config.feature['filepaths']["train_master_data_dir"] = master_df_folder
    create_feature_data(config=preproc_config, train=True)

    # Smart Priority Training
    smart_config.train['filepaths']["master_df_dir"] = master_df_folder
    smart_config.train['filepaths']["saved_swarm_folder_dir"] = swarm_label_folder
    train_smart_priority_model(config=smart_config)

    # Postprocessing
    run_postprocessing(postproc_config, train=True, master_df_folder=master_df_folder)

    return


if __name__ == "__main__":
    run_train(device_type="ion3k")
    run_train(device_type="global_core")