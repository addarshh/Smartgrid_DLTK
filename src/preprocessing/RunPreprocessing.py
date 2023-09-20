#!/usr/bin/env python
# coding: utf-8


import os
import datetime
from preprocessing.RawData import RawData
from preprocessing.MergeData import MergeData


def create_master_data(config, train=True):
    """
    : Read input files and return master data frame
    """

    # read all input data files
    raw_data_obj = RawData(working_dir=config.master['filepaths']["raw_data_dir"])

    try:
        flaps_dashboard_file = config.master['filepaths']['flaps_dashboard_file']
    except:
        flaps_dashboard_file=None

    syslog_file = config.master['filepaths']["syslog_data_files"]
    normalizer_yml = config.master['filepaths']["normalizer_yml"]
    datatype = config.master['parameters']["type"]
    raw_data_obj.get_normalized_events(syslog_file, normalizer_yml, flaps_dashboard_file, datatype)
    raw_data_obj.get_device_lookup(config.master['filepaths']["device_lookup_files"])

    if train:
        raw_data_obj.get_remedy_incidents(remedy_inc_file=config.master['filepaths']["remedy_inc_file"])
        raw_data_obj.get_remedy_label(config.master['filepaths']["remedy_labels_file"])

    # join all data sources
    merge_data_obj = MergeData(raw_data_obj, config)
    master_df = merge_data_obj.master_df.copy()

    # Saving master df
    if train:
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S' + "/")
        output_folder_path = config.master['filepaths']["save_data_dir"] + '/' + now
        os.makedirs(output_folder_path)
    else:
        output_folder_path = config.master['filepaths']["save_data_dir"] + '/prediction/'

    master_df = master_df.drop_duplicates("event_id")
    master_df.to_csv(output_folder_path + config.master['filepaths']["master_df_file"], header=True, index=False)
    print('master data for training saved in: {}'.format(output_folder_path))

    return output_folder_path
