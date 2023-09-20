#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from preprocessing.Associations import get_association_labels


class MergeData:

    def __init__(self, raw_data_obj, config):

        self.raw_data_obj = raw_data_obj
        self.master_df = None
        self.remedy_with_label_df = None
        self.user_label = None
        self.normalized_col = config.master['fields']["normalized_fields"]
        self.master_col = config.master['fields']["master_df_fields"]
        self.config = config

        assert self.raw_data_obj.normalized_event_df is not None
        assert self.raw_data_obj.raw_device_df is not None
        self.merge_events_with_device_lookup()

        if (self.raw_data_obj.raw_remedy_df is not None) and (self.raw_data_obj.raw_remedy_label_df is not None):

            # Merge labels
            self.merge_remedy_incidents_with_labels()

            # merge syslog events with remedy
            self.merge_events_with_remedy()

            # merge remedy associations
            self.get_associations(config.master['filepaths']["raw_data_dir"] + config.master['filepaths']["remedy_associations_file"])

            # overwrite existing label with user label if configured
            if config.master['parameters']["user_label"]:
                self.merge_user_label(config.master['filepaths']["user_label_dir"])
        return

    def merge_events_with_device_lookup(self):
        """
        : Merges the normalized_event_df (syslog_data_ion3k) with the raw_device_df (fc_devicesLocation_new)
        :
        : Note: this will filter out any device not in the device lookup file
        """

        print('Merging device data with events')
        self.master_df = pd.merge(self.raw_data_obj.normalized_event_df,
                                  self.raw_data_obj.raw_device_df,
                                  how='inner',
                                  left_on=['device_name'],
                                  right_on=['host'])

        # if any of those cols does not exist
        for col in self.normalized_col:
            if col not in self.master_df.columns.to_list():
                self.master_df[col] = np.nan

        self.master_df = self.master_df[self.normalized_col].reset_index(drop=True)

        # get city code
        self.master_df['city_code'] = self.master_df['location'].apply(lambda x: x[:3])

        # Add IP group
        self.master_df['ip_group'] = self.master_df['ip_address'].apply(lambda x: '.'.join(x.split('.')[:2]))

        # Create first occurrence column with epoch time
        self.master_df['first_occurrence'] = pd.to_datetime(self.master_df['timestamp'])
        self.master_df['epoch_time'] = pd.to_datetime(self.master_df['timestamp']).apply(lambda x: int(x.timestamp()))
        return

    def merge_remedy_incidents_with_labels(self):
        """
        : Assigns labels for the remedy incidents (USEFUL? = Y/N)
        """
        print('Assigning labels to events')

        # fill na with 1
        label_cols = ['STATUS_TXT', 'STATUS_REASON_TXT', 'RESOLUTION_CATEGORY',
                      'RESOLUTION_CATEGORY_TIER_2', 'RESOLUTION_CATEGORY_TIER_3']
        remedy_label_df = self.raw_data_obj.raw_remedy_label_df.copy()
        remedy_label_df['USEFUL?'] = remedy_label_df['USEFUL?'].fillna(1)
        remedy_label_df = remedy_label_df.set_index(label_cols)['USEFUL?']

        # merge with raw_remedy_df, drop na, na corresponds to pending/open remedy tickets
        self.remedy_with_label_df = pd.merge(self.raw_data_obj.raw_remedy_df,
                                             remedy_label_df,
                                             how='left',
                                             on=label_cols).dropna(subset=['USEFUL?'])

        # adding ACTIONABLE label
        self.remedy_with_label_df['ACTIONABLE?'] = np.where(
            (self.remedy_with_label_df['RESOLUTION_CATEGORY'] != 'NO ACTION TAKEN') &
            (self.remedy_with_label_df['RESOLUTION_CATEGORY'].notna()), 1, 0)

        # adding LOR label
        self.remedy_with_label_df['IS_LOR'] = np.where(
            (self.remedy_with_label_df['CATEGORIZATION_TIER_3'] == 'NETWORK CONNECTIVITY \\ LOSS OF REDUNDANCY'), 1, 0)

        # adding IS_REMEDY label
        self.remedy_with_label_df = self.remedy_with_label_df.assign(IS_REMEDY=1)
        return

    def merge_events_with_remedy(self):
        """
        : Merge remedy incident data with normalized event data
        : Join event data with the syslog data
        :  1. Inner join on the host / HPD_CI
        :  2. Drop anything that is "in progress"
        :  3. Filter for events that are within the time range
        """

        master_df = self.associate_remedy_with_syslog(self.master_df,
                                                      self.remedy_with_label_df,
                                                      self.config.master["parameters"]["remedy_window"])

        # drop events that are in progress
        master_df = master_df[~master_df['STATUS_TXT'].isin(['Pending', 'Assigned', 'In Progress'])]
        for col in ['USEFUL?', 'ACTIONABLE?', 'IS_REMEDY', 'IS_LOR']:
            master_df[col] = master_df[col].fillna(0)
        self.master_df = master_df[self.master_col]

        self.master_df = self.master_df.drop_duplicates(subset=['event_id'])

        print("Total events: {}".format(len(self.master_df)))
        self.master_df = self.filter_chronic_devices(self.master_df)

        return

    def merge_user_label(self, user_label_dir):
        """
        : This merges manually generated labels with the remedy incidents
        """
        try:
            self.user_label = pd.read_csv(user_label_dir).drop_duplicates()
            self.user_label["user_label"] = self.user_label["user_label"].replace("Non Actionable", 0)
            self.user_label["user_label"] = self.user_label["user_label"].replace("Actionable", 1)
            # drop the rows without user label

            self.events_label = self.user_label[self.user_label['event_id'].notna()].drop_duplicates()
            self.events_dashboard = self.user_label[self.user_label['event_id'].isna()].drop_duplicates(
                subset=['group_label'], keep='first')

            self.user_label = self.events_label[['event_id', "group_label"]].merge(
                self.events_dashboard[['group_label', 'user_label']],
                how='inner',
                on=['group_label'])

            self.user_label = self.user_label.dropna(subset=['event_id'])

            if len(self.user_label) > 0:
                columns_list_master_df = self.master_df.columns
                self.user_label = self.user_label[['event_id', "user_label"]].merge(
                    self.master_df.drop('USEFUL?', axis=1),
                    how='inner',
                    on=['event_id'])
                print("Number of user label associated from feedback data: ", len(self.user_label))
                # append to the existing file
                self.master_df = pd.merge(self.master_df, self.user_label[['event_id', 'user_label']], how='left',
                                          on=['event_id'])

                # fill na for user_label
                self.master_df["user_label"] = self.master_df["user_label"].fillna(self.master_df['USEFUL?'])

                self.master_df = self.master_df.drop('USEFUL?', 1).rename(columns={"user_label": 'USEFUL?'})
                self.master_df = self.master_df[columns_list_master_df]

        except:
            print("User label not associated")
            pass

        return

    def get_associations(self, associations_file):
        """
        : Merge remedy associations with master_df
        """
        self.master_df = get_association_labels(associations_file, self.master_df)
        return

    @staticmethod
    def associate_remedy_with_syslog(syslog_df, remedy_df, time_window):
        """
        : Merge syslog events with the remedy tickets based on timestamp
        : Syslog events without a remedy ticket are labeled as NOT USEFUL
        """

        df = syslog_df.copy()

        df = pd.merge(df,
                      remedy_df,
                      how='inner',
                      left_on='host',
                      right_on='HPD_CI')

        df['SUBMIT_DATE'] = pd.to_datetime(df['SUBMIT_DATE']).dt.tz_localize('UTC')
        df['submit_et'] = df['SUBMIT_DATE'].apply(lambda x: int(x.timestamp()))
        df['submit_et_start'] = df['submit_et'] - time_window
        df['Within_Timeframe'] = (df['submit_et_start'] < df['epoch_time']) & (df['submit_et'] > df['epoch_time'])
        df = df[df['Within_Timeframe']]
        df = pd.merge(syslog_df,
                      df[list(remedy_df.columns) + ['event_id', 'submit_et']],
                      how='left',
                      left_on='event_id',
                      right_on='event_id')
        return df

    @staticmethod
    def filter_chronic_devices(master_df):
        """
        : Filter out devices that have large ratio of syslog events to remedy tickets
        """

        df = master_df[["device_name", "event_id"]].groupby(["device_name"]).count().reset_index()
        df2 = master_df[["device_name", "IS_REMEDY"]].groupby(["device_name"]).sum().reset_index()
        df = pd.merge(df, df2, on="device_name")
        df["actions"] = df["event_id"]/(df["IS_REMEDY"] + 1)
        df["filter"] = df["actions"] < 500

        df = pd.merge(df[["device_name", "filter"]], master_df, on="device_name")
        df = df[df["filter"]]
        df = df.drop(columns=["filter"])
        return df