#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import pickle
import json
import datetime
import copy
from BasePath import base_path

class OnlinePrediction:
    """
    : Run the online smart priority prediction
    """

    def __init__(self, saved_model_folder, feature_file_dir, swarm_file_dir):

        self.feature_file_dir = feature_file_dir
        self.swarm_file_dir = swarm_file_dir
        self.saved_model_dir = saved_model_folder
        self.scaler_model_file = saved_model_folder + 'scaler_model_file.pkl'
        self.dim_red_model_file = saved_model_folder + 'dim_red_model_file.pkl'
        self.cluster_model_file = saved_model_folder + 'cluster_model_file.pkl'
        self.classification_model_file = saved_model_folder + 'classification_model_file.pkl'
        self.config_file = saved_model_folder + 'config_file.json'
        with open(self.config_file) as f:
            self.config_dict = json.load(f)
        self.drop_out_vars = ['USEFUL?', 'IS_REMEDY', 'IS_LOR', 'ACTIONABLE?', 'cluster_label', 'label']
        self.drop_column = ['IS_REMEDY', 'USEFUL?', 'event_id', 'cluster_label',
                            'IS_LOR', 'ACTIONABLE?', 'first_occurrence']

        self.feature_df = None
        self.feature_clustering = None
        self.red_dim = None
        self.merged_results = None
        self.classification_df = None
        self.model_fit = None
        self.y_test_prob_full = None
        self.y_test_pred_full = None
        self.X_test_full = None
        return

    def feature_selection(self, time_cut_off = None):
        """
        : Gets dataframe with the correct features
        """
        self.feature_df = pd.read_csv(self.feature_file_dir)
        if time_cut_off!=None:
            self.feature_df.loc[:,'timestamp'] = pd.to_datetime(self.feature_df['first_occurrence'],utc=True)
            self.feature_df = self.feature_df[self.feature_df['timestamp']>time_cut_off.replace(tzinfo = datetime.timezone.utc)]
            self.feature_df = self.feature_df.drop(columns='timestamp').reset_index(drop=True)
        self.feature_df['USEFUL?'] = 0
        self.feature_df['ACTIONABLE?'] = 0
        self.feature_df['IS_REMEDY'] = 0
        self.feature_clustering = self.feature_df[self.config_dict['feature_col']].copy()
        return

    def preprocess(self):
        """
        : Standardize the features using standard scaler
        """
        scaler = pickle.load(open(self.scaler_model_file, 'rb'))
        self.feature_clustering = pd.DataFrame(scaler.transform(self.feature_clustering))
        self.feature_clustering.columns = self.config_dict['feature_col']
        return

    def post_process_clusters(self):
        """
        : Post processing clusters. Wait is this needed?
        """

        # add the cluster label and regular label to the feature df
        self.feature_df['cluster_label'] = 1

        # format final df
        classification_df = copy.deepcopy(self.feature_clustering)
        classification_df['cluster_label'] = 1
        classification_df['label'] = classification_df[self.config_dict['predict_var']]
        self.classification_df = classification_df.drop(self.drop_column, axis=1, errors='ignore')

        print(f'Total: {len(self.classification_df)}')
        return

    def load_classification_model(self):
        """
        : Load model from pickle file
        """
        self.model_fit = pickle.load(open(self.classification_model_file, 'rb'))
        return

    def prediction(self):
        """
        : Generate model prediction for all records
        """
        xx_class = self.classification_df.drop(columns=['label'])
        y_test_prob = self.model_fit.predict_proba(xx_class)[:, 1]
        y_test_pred = (y_test_prob >= self.config_dict['best_threshold']).astype('int')

        # prediction across all data
        df_action_cluster = self.feature_df.drop(self.drop_column, axis=1, errors='ignore')

        self.y_test_prob_full = y_test_prob
        self.y_test_pred_full = y_test_pred
        self.X_test_full = df_action_cluster
        return

    def make_output_file(self):
        """
        : Make output dataframe
        """
        test_df = self.feature_df.loc[self.X_test_full.index]
        test_df['Useful_pred'] = self.y_test_pred_full
        test_df['Useful_prob'] = self.y_test_prob_full
        test_df['test'] = 1

        swarm_df = pd.read_csv(self.swarm_file_dir)
        merged_results = test_df.merge(swarm_df[['group_label', 'event_id']], how='left', on=['event_id'])
        grouped_by_swarm = merged_results[['Useful_pred', 'group_label']].groupby(
            ['group_label']).sum().reset_index()
        grouped_by_swarm.loc[grouped_by_swarm['Useful_pred'] > 0, 'Useful_pred'] = 1
        grouped_by_swarm = grouped_by_swarm.rename(columns={'Useful_pred': 'Group_Useful_pred'})
        self.merged_results = pd.merge(merged_results, grouped_by_swarm, how='left', on=['group_label'])
        return

    def save_prediction_files(self, path):
        """
        : Save output to csv file
        """
        # TODO: Don't change directory
        # make folder for output
        output_folder_name = 'online_prediction_run_' + datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        output_folder_path = path + '/' + output_folder_name
        os.makedirs(output_folder_path)
        os.chdir(output_folder_path)
        print('All online prediction files will be saved in ', os.getcwd())

        # save config
        self.config_dict['saved_model_dir'] = self.saved_model_dir
        config_file = 'config_file.json'
        f = open(config_file, 'w')
        f.write(json.dumps(self.config_dict))
        f.close()

        # save output
        self.merged_results.to_csv('online_prediction.csv', header=True, index=False)
        os.chdir(base_path)
        return
