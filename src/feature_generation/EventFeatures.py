#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import pickle
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.model_selection import train_test_split
from feature_generation.SequentialFeatures import rolling_num_features_on_time, \
                                                            pad_single_event, padding_time_sequence,\
                                                            interface_features_from_message


class EventFeatures:
    """
    : Object that generates all the features used by the smart priority model
    """

    def __init__(self, master_df, conf, train=True):

        self.conf = conf
        self.master_df = master_df.copy()

        self.encoded_master_df = None
        self.feature_df = None

        # encoding categorical features
        catboost_encoder_file = conf['filepaths']["train_master_data_dir"] + '/' + conf['filepaths']["catboost_encoder_file"]
        self.catboost_encoding(train=train, catboost_encoder_file=catboost_encoder_file)

        # make features
        timezone_lookup_file = conf['filepaths']["raw_data_dir"] + conf['filepaths']["timezone_lookup_file"]
        self.make_features(timezone_lookup_file)

        # add useful, actionable, is_remedy labels when training
        if train:
            self.add_labels()
        return

    def catboost_encoding(self, train, catboost_encoder_file):
        """
        : Create catboost encoding of the categorical features
        """

        if train:
            # train catboost encoder using 5% of events (and drop these events)
            print('Training catboost encoder')
            x_keep, x_cbe, y_keep, y_cbe = train_test_split(self.master_df[self.conf['fields']["cat_features"] + ['event_id']],
                                                            self.master_df[self.conf['parameters']["ts_target"]],
                                                            test_size=0.05, random_state=42)
            cbe_encoder = CatBoostEncoder()
            cbe_encoder.fit_transform(x_cbe[self.conf['fields']["cat_features"]], y_cbe)

            pickle.dump(cbe_encoder, open(catboost_encoder_file, 'wb'))

        else:
            print('Loading trained catboost encoder \n')
            cbe_encoder = pickle.load(open(catboost_encoder_file, 'rb'))
            x_keep = self.master_df[self.conf['fields']["cat_features"] + ['event_id']].copy()

        # encode the other data
        keep_cbe = cbe_encoder.transform(x_keep[self.conf['fields']["cat_features"]])
        keep_cbe.columns = [col + '_encoded' for col in keep_cbe.columns]

        # merge encoded columns to the original
        keep_cbe = keep_cbe.join(x_keep[['event_id']])
        self.encoded_master_df = pd.merge(keep_cbe, self.master_df, how='left', on=['event_id'])

        # update numerical feature list
        self.conf['fields']["num_feature"] = self.conf['fields']["num_feature"] + keep_cbe.columns.to_list()[:-1]
        return

    def backward_rolling_num_features(self):
        """
        : Create backward rolling numerical features
        """

        print('Making backward rolling numerical features... This takes some time')

        backward_rolling_num_df = None
        backward_rolling_windows = self.conf['parameters']["backward_rolling_windows"]
        for i in range(len(backward_rolling_windows)):
            window_str = str(backward_rolling_windows[i]) + 's'
            #print(window_str)
            backward_df = rolling_num_features_on_time(self.encoded_master_df,
                                                       self.conf['fields']["num_feature"],
                                                       window_size=backward_rolling_windows[i],
                                                       forward=False)

            if i == 0:
                backward_rolling_num_df = backward_df
            else:
                backward_rolling_num_df = pd.merge(backward_rolling_num_df,
                                                   backward_df,
                                                   how='inner',
                                                   on=['event_id'])
        return backward_rolling_num_df

    def forward_rolling_num_features(self):
        """
        : Create forward rolling numerical features
        """

        print('Making forward rolling numerical features... This takes a little while')

        forward_rolling_num_df = None
        forward_rolling_windows = self.conf['parameters']["forward_rolling_windows"]
        for i in range(len(forward_rolling_windows)):
            window_str = str(forward_rolling_windows[i]) + 's'
            #print(window_str)
            forward_df = rolling_num_features_on_time(self.encoded_master_df,
                                                      self.conf['fields']["num_feature"],
                                                      window_size=forward_rolling_windows[i],
                                                      forward=True)

            if i == 0:
                forward_rolling_num_df = forward_df
            else:
                forward_rolling_num_df = pd.merge(forward_rolling_num_df, 
                                                  forward_df,
                                                  how='inner', on=['event_id'])
        return forward_rolling_num_df

    def local_time_features(self, timezone_lookup_file):
        """
        : Create local time features
        """

        print('Making local time features')
        timezone_lookup = pd.read_csv(timezone_lookup_file)

        # Get local time
        timezone_lookup = timezone_lookup[['State', 'Timezone']].groupby('State').head(1)
        self.encoded_master_df['state'] = self.encoded_master_df['location'].apply(lambda x: x[:2])
        self.encoded_master_df = pd.merge(self.encoded_master_df, timezone_lookup, how='left', left_on='state',
                                          right_on='State').drop('State', 1)
        self.encoded_master_df['Timezone'] = pd.to_timedelta(self.encoded_master_df['Timezone'], unit='H')

        # Update missing Timezones with longitude-calculated
        if 'longitude' in self.encoded_master_df.columns:
            self.encoded_master_df['longitude'] = pd.to_numeric(self.encoded_master_df['longitude'], errors='coerce').fillna(-6)
            self.encoded_master_df['TimezoneCalc'] = (self.encoded_master_df['longitude']/15.0).astype('int').astype('float')
            self.encoded_master_df['TimezoneCalc'] = pd.to_timedelta(self.encoded_master_df['TimezoneCalc'], unit='H')
            self.encoded_master_df.Timezone.fillna(self.encoded_master_df.TimezoneCalc, inplace=True)
            del self.encoded_master_df['TimezoneCalc']

        self.encoded_master_df['local_first_occurrence'] = pd.to_datetime(
            self.encoded_master_df['first_occurrence']) + self.encoded_master_df['Timezone']

        # add dow and hour of day
        local_time_feature_df = self.encoded_master_df[['event_id', 'local_first_occurrence']].copy()
        local_time_feature_df['local_dow'] = local_time_feature_df['local_first_occurrence'].dt.weekday
        local_time_feature_df['local_hod'] = local_time_feature_df['local_first_occurrence'].dt.hour
        local_time_feature_df['local_business_day'] = np.where(local_time_feature_df['local_dow'] < 5, 1, 0)
        local_time_feature_df['local_business_hour'] = np.where(
            (local_time_feature_df['local_business_day'] > 0) &
            (abs(local_time_feature_df['local_hod'] - 12) <= 4), 1, 0)
        return local_time_feature_df

    def make_features(self, timezone_lookup_file):
        """
        : Function that merges all the features into one dataframe
        """
        print('Generating feature_df')

        feature_df = self.master_df[['event_id', 'first_occurrence']].copy().drop_duplicates("event_id")

        # Severity of the event and numeric features
        feature_df = pd.merge(feature_df, self.encoded_master_df[['event_id'] + self.conf['fields']["num_feature"]],
                                   how='inner', on=['event_id'])

        # Padded sequence feature
        print('Generating padded time sequence feature... This takes a while')
        padded_sequence_df = padding_time_sequence(self.encoded_master_df,
                                                   forward_window=self.conf['parameters']["forward_sequence_window"],
                                                   backward_window=self.conf['parameters']["backward_sequence_window"])
        feature_df = pd.merge(feature_df, padded_sequence_df, how='inner', on=['event_id'])
        
        # Rolling aggregation features
        backward_rolling_num_df = self.backward_rolling_num_features()
        forward_rolling_num_df = self.forward_rolling_num_features()
        feature_df = pd.merge(feature_df, backward_rolling_num_df, how='inner', on=['event_id'])
        feature_df = pd.merge(feature_df, forward_rolling_num_df, how='inner', on=['event_id'])

        # Local time features
        local_time_feature_df = self.local_time_features(timezone_lookup_file)
        feature_df = pd.merge(feature_df, local_time_feature_df.drop('local_first_occurrence', 1),
                                   how='inner', on=['event_id'])

        # Padded single event feature
        print('Generating padded single event feature')
        padded_single_event_df = pad_single_event(self.encoded_master_df,
                                                  backward_window=self.conf['parameters']["backward_single_event_window"],
                                                  forward_window=self.conf['parameters']["forward_single_event_window"])
        feature_df = pd.merge(feature_df, padded_single_event_df, how='inner', on=['event_id'])

        # Interface feature
        print('Generating interface feature')
        interface_df = interface_features_from_message(self.encoded_master_df)
        feature_df = pd.merge(feature_df, interface_df, how='inner', on=['event_id'])

        # fill na for kurtosis features --> -10
        for col in feature_df.columns.values:
            if 'kurt' in col:
                feature_df[col].fillna('-10', inplace=True)

        # Fill na for other columns (var, slope) --> 0
        self.feature_df = feature_df.fillna(0).reset_index(drop=True)
        return

    def add_labels(self):
        """
        : Add the USEFUL, IS_REMEDY, and ACTIONABLE labels to the feature dataset
        """

        self.feature_df = pd.merge(self.feature_df,
                                   self.encoded_master_df[['event_id', 'USEFUL?', 'IS_REMEDY', 'ACTIONABLE?']],
                                   how='inner', on=['event_id'])

        # fill na for labels
        self.feature_df['IS_REMEDY'] = self.feature_df['IS_REMEDY'].fillna(0)
        self.feature_df['ACTIONABLE?'] = self.feature_df['ACTIONABLE?'].fillna(0)
        self.feature_df['USEFUL?'] = self.feature_df['USEFUL?'].fillna(0)
        return
