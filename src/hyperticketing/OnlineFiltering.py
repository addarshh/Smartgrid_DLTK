# !/usr/bin/env python
# coding: utf-8

from __future__ import division
import os
import json
import pandas as pd
import numpy as np
import datetime
from BasePath import base_path
from hyperticketing.EventGroupEval import spatial_temp_filter


def process_master_for_swarming(master_df_dir, device_type='all'):
    """
    : Load master_df
    """
    master_df = pd.read_csv(master_df_dir)
    if device_type != 'all':
        master_df = master_df.loc[master_df['ITEM'] == device_type].copy().reset_index()
    return master_df


def old_new_group_label_mapping(group_df, existing_swarm):
    """
    : Get new/old label mapping.
    : When predicting online, the group labels need to be consistent from run to run
    """
    new_group_label_key = sorted(group_df.loc[group_df['existing_label'].isna(), 'group_label'].unique())
    new_group_label_val = np.array(range(0, len(new_group_label_key))) + max(existing_swarm['group_label']) + 1
    new_group_label_dict = {new_group_label_key[i]: new_group_label_val[i] for i in range(len(new_group_label_key))}
    old_group_label_key = group_df.loc[group_df['existing_label'].notna()][['group_label', 'existing_label']].groupby(
        by=['existing_label', 'group_label']).head(1)['group_label'].values
    old_group_label_val = group_df.loc[group_df['existing_label'].notna()][['group_label', 'existing_label']].groupby(
        by=['existing_label', 'group_label']).head(1)['existing_label'].values
    old_group_label_dict = {old_group_label_key[i]: old_group_label_val[i] for i in range(len(old_group_label_key))}
    group_label_dict = {**new_group_label_dict, **old_group_label_dict}
    return group_label_dict


class OnlineFiltering:
    """
    : Object that runs the filtering algorithm prediction using the stored pso parameters
    """

    def __init__(self, work_dir):
        config_file = work_dir + 'pso_filters_run/pso_filter.json'
        with open(config_file) as f:
            pso_filter = json.load(f)
        self.window = pso_filter['window']
        self.spatial_filter = pso_filter['spatial_filter']
        self.beta = pso_filter['beta']
        self.new_event_swarm = None
        return

    def filtering_even_faster(self, existing_swarm_dir, new_master_df_dir):
        """
        : Load swarm parameters and run filtering algorithm
        """

        # read existing swarm
        existing_swarm = pd.read_csv(existing_swarm_dir)
        existing_swarm['ip_group'] = existing_swarm['ip_group'].astype('O')

        # read new events
        original_new_event = process_master_for_swarming(new_master_df_dir)

        merged_event = original_new_event.append(existing_swarm)[
            original_new_event.columns.to_list()].drop_duplicates(subset=['event_id'])

        # get new event data with spatial filters
        group_df = spatial_temp_filter(merged_event, self.window, self.spatial_filter)

        # merge with existing label
        group_df = group_df.merge(
            existing_swarm[['event_id', 'group_label']].rename(columns={'group_label': 'existing_label'}),
            how='left', on='event_id')

        # get new/old label mapping
        group_label_dict = old_new_group_label_mapping(group_df, existing_swarm)

        # apply mapping
        group_df['new_group_label'] = group_df['group_label'].apply(lambda x: group_label_dict[x])

        # merge with original
        self.new_event_swarm = original_new_event[['event_id']]\
            .merge(group_df,  how='left', on='event_id')\
            .drop(columns=['group_label', 'existing_label'])\
            .rename(columns={'new_group_label': 'group_label'})
        return

    def save_output_files(self, path):
        """
        : Saves pso result to files
        """

        # TODO: don't change working directory
        # make folder for output
        output_folder_name = 'pso_online_filtering_run_' + datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        output_folder_path = path + '/' + output_folder_name
        os.makedirs(output_folder_path)
        os.chdir(output_folder_path)
        print('All model files will be saved in ', os.getcwd())

        filter_dict = {
            'temporal_filter': self.window,
            'spatial_filter': self.spatial_filter,
        }
        config_file = 'pso_online_filtering.json'
        f = open(config_file, 'w')
        f.write(json.dumps(filter_dict))
        f.close()

        # save the new swarm labels
        self.new_event_swarm.to_csv('swarm_label.csv', header=True, index=False)
        os.chdir(base_path)
        return