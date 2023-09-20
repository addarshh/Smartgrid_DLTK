#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

# utils to save model
import os
import pickle
import json
import copy

# preprocessing features
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample

# clustering models
from sklearn.cluster import KMeans

# gb and linear model
from sklearn.ensemble import RandomForestClassifier


# model pipeline
class ModelPipeline:
    def __init__(self, feature_file_dir, config, fit_model=True):
        """
        : Set parameters for training
        """

        # set parameters
        self.drop_column = ['IS_REMEDY', 'USEFUL?', 'IS_LOR', 'ACTIONABLE?', 'first_occurrence']

        self.feature_df = pd.read_csv(feature_file_dir)

        drop_cols = ["event_id", "timestamp", "severity", "first_occurrence"]
        self.feature_col = [f for f in self.feature_df.columns if f not in drop_cols]
        if not config.train['parameters']["post_feature"]:
            self.feature_col = [feat for feat in self.feature_col if 'post' not in feat]

        self.useful_labels = self.feature_df['USEFUL?'].values
        self.remedy_labels = self.feature_df['IS_REMEDY'].values
        self.actionable_labels = self.feature_df['ACTIONABLE?'].values
        self.config = config

        self.config_dict = {
            'nmin': config.train['parameters']['preprocess_nmin'],
            'nmax': config.train['parameters']['preprocess_nmax'],
            'feature_col': self.feature_col,
            'preprocess_method': config.train['parameters']['preprocess_method'],
            'predict_var': config.train['parameters']['predict_var'],
            'classification_model_param_dict': config.train['parameters']['classification_model_param_dict'],
            'best_threshold': None
        }

        self.feature_data = None
        self.scaler = None
        self.classification_df = None
        self.top_cluster_nums = None
        self.model_fit = None
        self.model_data = None

        self.run(fit_model)
        return

    def run(self, fit_model):
        """
        : Run model training pipeline
        """
        self.preprocess()
        self.fit_classification_model(fit_model)
        return

    def preprocess(self):
        """
        : Standardize the features using standard scaler (need to use min max for non negativity for NMF)
        """

        method = self.config_dict['preprocess_method']
        nmin = self.config_dict['nmin']
        nmax = self.config_dict['nmax']

        if method == 'minmax':
            print('using MinMaxScaler for standardizing features')
            nrange = (nmin, nmax)
            self.scaler = MinMaxScaler(nrange)
        else:
            print('using StandardScaler (default) for standardizing features')
            self.scaler = StandardScaler()

        # scale on all data
        self.feature_data = self.feature_df[self.feature_col].copy()
        self.feature_data = pd.DataFrame(self.scaler.fit_transform(self.feature_data))
        self.feature_data.columns = self.feature_col
        self.feature_data = self.add_labels_back(self.feature_data)
        self.feature_data['event_id'] = self.feature_df['event_id']
        predict_var = self.config_dict['predict_var']
        self.feature_data['label'] = self.feature_data[predict_var]
        return

    def fit_classification_model(self, fit_model):
        """
        : Fit Model
        """

        model_param_dict = self.config_dict['classification_model_param_dict']

        self.feature_data = self.merge_times(self.feature_data)

        # get test and training data
        print(self.feature_data['label'].value_counts())

        if fit_model:
            self.balanced_sample(self.feature_data)

            print(f'Fitting Random Forest Model.....')
            model = RandomForestClassifier(random_state=0,
                                           n_estimators=model_param_dict['n_estimators'],
                                           max_depth=model_param_dict['max_depth'],
                                           class_weight='balanced')
            self.model_fit = model.fit(self.X_train, self.y_train)
        else:
            df = self.feature_data.copy()
            self.feature_data = pd.concat([df, df, df])
            self.balanced_sample(self.feature_data)
        return

    def add_labels_back(self, df):
        """
        : Add labels back to dataframe
        """
        df['USEFUL?'] = self.useful_labels
        df['IS_REMEDY'] = self.remedy_labels
        df['ACTIONABLE?'] = self.actionable_labels
        return df

    def merge_times(self, df):
        """
        : Add epoch times back to dataframe
        """
        times_df = self.feature_df
        times_df["epoch_time"] = pd.to_datetime(times_df["first_occurrence"]).apply(lambda x: int(x.timestamp()))
        df = pd.merge(df, times_df[["event_id", "epoch_time"]], on="event_id", how="inner")
        return df

    def balanced_sample(self, df):
        """
        : Creates a balanced subsample of dataset
        """

        # Split train and test
        df = df.sort_values(by="epoch_time")
        train = df.iloc[0:int(2*len(df)/3), :]
        test = df.iloc[int(2*len(df)/3):, :]

        self.X_test = test.drop(columns=["epoch_time", "label"])
        self.y_test = test['label']
        self.test_event_ids = test["event_id"]

        # Get ratio for balanced sampling
        train = train.drop(self.drop_column, axis=1, errors='ignore')
        a = train['label'].value_counts()[0]
        b = train['label'].value_counts()[1]
        ratio = np.sqrt(a / b)

        # mix_sampling
        self.X_train = self.mix_sample(train, ratio, ratio)
        self.y_train = self.X_train['label']
        self.X_train = self.X_train.drop(columns=["event_id", "epoch_time", "label"])

        return

    @staticmethod
    def mix_sample(df_im, ratio_up, ratio_down):
        """
        : Randomized resampling
        """
        df_minority = df_im.loc[df_im['label'] == 1]
        df_majority = df_im.loc[df_im['label'] == 0]
        df_minority_upsample = resample(df_minority,
                                        replace=True,
                                        n_samples=int(len(df_minority) * ratio_up),
                                        random_state=42)
        df_majority_downsample = resample(df_majority,
                                          replace=True,
                                          n_samples=int(len(df_majority) / ratio_down),
                                          random_state=42)
        df = pd.concat([df_minority_upsample, df_majority_downsample])
        return df

    def save_model_files(self, model_eval, threshold, path):
        """
        : Save output of model training
        """

        # make folder for output
        output_folder_name = 'model_pipeline_run'
        output_folder_path = path + '/' + output_folder_name
        os.makedirs(output_folder_path)
        print('All model files will be saved in ', output_folder_path)

        # Save weights
        scaler_model_file = output_folder_path + '/scaler_model_file.pkl'
        pickle.dump(self.scaler, open(scaler_model_file, 'wb'))

        classification_model_file = output_folder_path + '/classification_model_file.pkl'
        pickle.dump(self.model_fit, open(classification_model_file, 'wb'))

        self.config_dict['best_threshold'] = threshold

        config_file = output_folder_path + '/config_file.json'
        with open(config_file, 'w') as f:
            f.write(json.dumps(self.config_dict))
            f.close()

        # Save output
        merged_results = model_eval.create_output_df()
        merged_results.to_csv(output_folder_path + '/model_output.csv')
        return
