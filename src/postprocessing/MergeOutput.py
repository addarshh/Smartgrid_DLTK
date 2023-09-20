#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from datetime import timedelta
import datetime
import requests
import json
import os


class MergeOutput:
    """
    : Object to read latest prediction results and merge into dashboard
    : TODO: Fix pep8 flaws
    """

    def __init__(self, config):

        # get master file and output dir
        if config.post_proc['parameters']["from_training"]:
            if config.post_proc['filepaths']["training_folder_dir"] == 'latest':
                config.post_proc['filepaths']["training_folder_dir"] = config.post_proc['filepaths']["training_data_dir"] + pd.Series(
                    [i for i in os.listdir(config.post_proc['filepaths']["training_data_dir"]) if '20' in i]).max() + '/'
                config.post_proc['filepaths']["classification_pipeline_output_folder_dir"] = config.post_proc['filepaths']["training_folder_dir"]
                config.post_proc['filepaths']["classification_pipeline_output_folder_dir"] = config.post_proc['filepaths']["classification_pipeline_output_folder_dir"] + pd.Series(
                    [i for i in os.listdir(config.post_proc['filepaths']["classification_pipeline_output_folder_dir"]) if
                     (('20' in i) and ('model' in i))]).max() + '/'

            self.output_file_dir = config.post_proc['filepaths']["classification_pipeline_output_folder_dir"] + config.post_proc['filepaths']["classification_model_output_name"]
            self.master_file_dir = config.post_proc['filepaths']["training_folder_dir"] + config.post_proc['filepaths']["master_file_name"]

        else:
            self.dashboard_file_dir = config.post_proc['filepaths']["dashboard_folder_dir"] + config.post_proc['filepaths']["dashboard_output_name"]
            config.post_proc['filepaths']["online_classification_folder_dir"] = config.post_proc['filepaths']["online_classification_folder_dir"] + pd.Series(
                [i for i in os.listdir(config.post_proc['filepaths']["online_classification_folder_dir"]) if '20' in i]).max() + '/'
            self.output_file_dir = config.post_proc['filepaths']["online_classification_folder_dir"] + config.post_proc['filepaths']["online_classification_output_name"]
            self.master_file_dir = config.post_proc['filepaths']["prediction_folder_dir"] + config.post_proc['filepaths']["master_file_name"]

        self.final_output = None
        self.pred_output = None
        self.existing_dashboard = None
        return

    def merge_data(self, config):
        """
        : Merge prediction output with the existing dashboard
        """

        if config.post_proc['parameters']["from_training"]:
            print('Create dashboard from training data: ', self.output_file_dir)
            self.final_output = pd.merge(pd.read_csv(self.output_file_dir)[config.post_proc['fields']["output_col"]],
                                         pd.read_csv(self.master_file_dir)[config.post_proc['fields']["master_col"]], how='inner',
                                         on='event_id')

            # add training flag
            self.final_output['training'] = 1

            # add user_label and reason to the file for consistency
            self.final_output['user_label'] = np.nan
            self.final_output['reason'] = np.nan

        else:
            print('Merge output from: ', self.output_file_dir)
            print('With dashboard from: ', self.dashboard_file_dir)

            self.pred_output = pd.merge(pd.read_csv(self.output_file_dir)[config.post_proc['fields']["output_col"]],
                                        pd.read_csv(self.master_file_dir)[config.post_proc['fields']["master_col"]], how='inner',
                                        on='event_id')

            # TODO: Adding first occurrence to master df. Would be better to do this when master_df is created
            self.pred_output['first_occurrence'] = pd.to_datetime(self.pred_output['timestamp']).apply(
                lambda x: int(x.timestamp()))
            self.pred_output['first_occurrence'] = pd.to_datetime(self.pred_output['first_occurrence'], unit='s')
            columns = self.pred_output.columns.to_list() + ['Num_Useful_pred_per_group', 'user_label', 'reason']

            flag_trunc = 0
            try:
                self.existing_dashboard = pd.read_csv(self.dashboard_file_dir)
            except:
                self.existing_dashboard = pd.DataFrame(columns=columns)
                flag_trunc = 1

            # Convert first_occurrence to timestamp
            self.existing_dashboard['first_occurrence'] = pd.to_datetime(self.existing_dashboard['first_occurrence'])
            self.existing_dashboard = self.existing_dashboard[columns]

            # truncate pred_output to dashboard last timestamp - 2hrs and append to existing
            # do not need to append the events to existing, send it to the indexer with a curr timestamp
            if flag_trunc == 0:
                truncate_timestamp = self.existing_dashboard['first_occurrence'].dt.tz_localize(None).max() - timedelta(hours=2)
            else:
                truncate_timestamp = self.pred_output['first_occurrence'].min()

            self.final_output = pd.concat([self.existing_dashboard.assign(new_pred=0).drop(
                columns=['Num_Useful_pred_per_group', 'user_label', 'reason']),
                                           self.pred_output.loc[
                                               self.pred_output['first_occurrence'] >= truncate_timestamp].assign(
                                               new_pred=1)]).drop(columns=['Group_Useful_pred'])

            # remove duplicates (keep latest)
            self.final_output = self.final_output.sort_values(by=['new_pred'], ascending=False).drop_duplicates(
                subset=['event_id'], keep='first').drop(columns=['new_pred'])

            # recalculate group prediction
            group_useful = self.final_output[['group_label', 'Useful_pred']].groupby(by='group_label').sum().rename(
                columns={'Useful_pred': 'Num_Useful_pred_per_group'}).reset_index()
            group_useful['Group_Useful_pred'] = np.where(group_useful['Num_Useful_pred_per_group'] > 0, 1, 0)
            self.final_output = pd.merge(self.final_output, group_useful, how='inner', on=['group_label'])

            # add col to indicate this is from online
            self.final_output['training'] = 0

            # add user_label and reason
            self.final_output = pd.merge(self.final_output,
                                         self.existing_dashboard[['event_id', 'user_label', 'reason']], how='left',
                                         on='event_id')

        self.final_output['curr_time'] = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        return

    def save_output(self, config, master_df, train):
        """
        : Save final output to a csv with the dashboard data
        """
        self.final_output = pd.merge(master_df,self.final_output, how = 'inner', on = 'event_id', suffixes=('', '_drop'))
        self.final_output.drop([col for col in self.final_output.columns if 'drop' in col], axis=1, inplace=True)
        # self.final_output = self.final_output.loc[:,~self.final_output.columns.duplicated()]
        if train:
            self.final_output.to_csv(config.post_proc['filepaths']["dashboard_folder_dir"] + 'training_output.csv', header=True, index=False)
        else:
            if config.post_proc['parameters']["rewrite"]:
                self.dashboard_file_dir = config.post_proc['filepaths']["dashboard_folder_dir"] + config.post_proc['filepaths']["dashboard_output_name"]
                self.final_output.to_csv(self.dashboard_file_dir, header=True, index=False)
        return
