#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from feature_generation.EventFeatures import EventFeatures
from feature_generation.FlipLabel import flip_labels
from utils.utils import get_latest_folder


def create_feature_data(config, train=True):
    """
    : Read master_df and create features. Returns feature_df.
    """
    conf = config.feature

    # Find master_df csv
    if train:
        master_data_dir = get_latest_folder(conf['filepaths']["train_master_data_dir"], conf['filepaths']["data_dir"])
    else:
        master_data_dir = conf['filepaths']["data_dir"] + '/prediction'
    master_df = pd.read_csv(master_data_dir + '/' + conf['filepaths']["master_df_file"])
    master_df['first_occurrence'] = pd.to_datetime(master_df['first_occurrence'])

    # Flip label if any event in its group has a label of "useful"
    if train:
        master_df = flip_labels(master_df, conf)

    # Initialize event features object
    feature_data_obj = EventFeatures(master_df, conf, train)

    # Save feature_df as a csv
    feature_data_obj.feature_df.to_csv(master_data_dir + '/' + conf['filepaths']["feature_df_file"], header=True, index=False)

    return
