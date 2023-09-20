#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from preprocessing.SyslogNormalizer import SyslogNormalizer
import numpy as np
from BasePath import base_path


# RawData class
class RawData:

    def __init__(self, working_dir):
        self.working_dir = working_dir

        self.raw_remedy_df = None
        self.raw_remedy_label_df = None
        self.raw_device_df = None
        self.normalized_event_df = None
        return

    def get_remedy_incidents(self, remedy_inc_file):
        """
        : Read Remedy Incident file (self.raw_remedy_df)
        """
        print('Reading Remedy Incident Data from {}'.format(self.working_dir + remedy_inc_file))
        self.raw_remedy_df = pd.read_csv(self.working_dir + remedy_inc_file, engine='python')

        # remove duplicate INCs
        self.raw_remedy_df = self.raw_remedy_df.sort_values(by=['LAST_MODIFIED_DATE'], ascending=False).drop_duplicates(
            subset=['INCIDENT_NUMBER'])
        return

    def get_remedy_label(self, remedy_labels_file):
        """
        : Read Remedy Labels file (self.raw_remedy_label_df)
        """
        print('Reading Remedy Labels from {}'.format(self.working_dir + remedy_labels_file))
        self.raw_remedy_label_df = pd.read_csv(self.working_dir + remedy_labels_file)

        # drop rows with identical combination of columns
        self.raw_remedy_label_df = self.raw_remedy_label_df.sort_values(by='USEFUL?', ascending=False).drop_duplicates(
            subset=['RESOLUTION_CATEGORY',
                    'RESOLUTION_CATEGORY_TIER_2',
                    'RESOLUTION_CATEGORY_TIER_3',
                    'STATUS_TXT',
                    'STATUS_REASON_TXT'])

        # if label is blank: 'N' for NO ACTION TAKEN incidents, 'Y' for all others
        self.raw_remedy_label_df.loc[(self.raw_remedy_label_df['USEFUL?'].isna()) & (
                    self.raw_remedy_label_df['RESOLUTION_CATEGORY'] == 'NO ACTION TAKEN'), 'USEFUL?'] = 'N'
        self.raw_remedy_label_df['USEFUL?'] = self.raw_remedy_label_df['USEFUL?'].fillna('Y')

        # convert useful labels from Y/N to 1/0
        self.raw_remedy_label_df['USEFUL?'] = self.raw_remedy_label_df['USEFUL?'].apply(lambda x: 1 if x == 'Y' else 0)
        return

    def get_normalized_events(self, syslog_data_file, normalizer_yaml, flaps_dashboard_file, datatype):
        """
        : Read syslog data (self.normalized_event_df)
        """
        print("Reading Syslog Data")
        normalizer = SyslogNormalizer(normalizer_yaml)
        normalized_event_df = pd.DataFrame()

        # Read syslog data from single file
        filename = syslog_data_file
        event_df = pd.read_csv(self.working_dir + filename, engine='python', error_bad_lines=False)
        df = None
        if datatype == "ion3k":
            df = normalizer.normalize_ion3k(event_df)
        if datatype == "global_core":
            event_df = event_df.loc[event_df["messageLevel"].isin(['alert', 'crit', 'err'])]
            event_df = self.add_global_core_interface_flaps(event_df, flaps_dashboard_file)
            df = normalizer.normalize_global_core(event_df)
        normalized_event_df = normalized_event_df.append(df)

        # Sort and filter duplicate records
        self.normalized_event_df = normalized_event_df.sort_values(by=['device_name', 'timestamp'])\
                                                      .drop_duplicates(subset=['timestamp',
                                                                               'device_name',
                                                                               'event_policy_name',
                                                                               'severity',
                                                                               'date_hour',
                                                                               'date_mday',
                                                                               'date_month'], keep='first')\
                                                      .reset_index(drop=True)

        return

    def add_global_core_interface_flaps(self, global_core_syslog, flaps_dashboard_file):

        # global_core_syslog = global_core_syslog
        global_core_syslog.loc[:,'host'] = global_core_syslog['host'].apply(lambda x: x.split('.')[0].upper())

        dashboard_log = pd.read_csv(self.working_dir + flaps_dashboard_file)

        global_core_syslog.loc[:,'timestamp'] = pd.to_datetime(global_core_syslog['_time']).dt.tz_localize(None)
        global_core_syslog.loc[:,'timestamp'] = global_core_syslog['timestamp'].values.astype(np.int64) // 10 ** 9
        global_core_syslog.loc[:,'timestamp'] = pd.to_datetime(global_core_syslog['timestamp'], unit='s')

        dashboard_log.loc[:,'timestamp'] = pd.to_datetime(dashboard_log['_time']).dt.tz_localize(None)
        dashboard_log.loc[:,'timestamp'] = dashboard_log['timestamp'].values.astype(np.int64) // 10 ** 9
        dashboard_log.loc[:,'timestamp'] = pd.to_datetime(dashboard_log['timestamp'], unit='s')

        global_core_syslog.sort_values(by=['timestamp', 'host'], inplace=True)
        dashboard_log.sort_values(by=['timestamp', 'host'], inplace=True)

        merged = pd.merge_asof(dashboard_log[['timestamp', 'host', 'flap_count']],
                               global_core_syslog,
                               by='host', on='timestamp', tolerance=pd.Timedelta('60min'))

        merged_time_index = merged.set_index('timestamp').copy()
        sum_flap_count = merged_time_index.groupby('host')['flap_count'].rolling('60min').sum().reset_index()

        merged = pd.merge(merged.drop(columns='flap_count'),sum_flap_count,on=['timestamp','host'],how='left')

        merge_cols = list(global_core_syslog.columns)

        global_core_syslog = pd.merge(global_core_syslog,
                                      merged,
                                      on=merge_cols,
                                      how='left')

        duplicate_check_cols = list(global_core_syslog.columns)
        duplicate_check_cols = duplicate_check_cols.remove('flap_count')
        global_core_syslog = global_core_syslog.drop_duplicates(subset=duplicate_check_cols)
        global_core_syslog['flap_count'] = global_core_syslog['flap_count'].fillna(0)


        return global_core_syslog.drop(columns=['timestamp'])

    def get_device_lookup(self, device_lookup_files):
        """
        : Read device lookup data  (self.raw_device_df)
        """
        print('Reading device data')
        df = pd.DataFrame()
        filename = device_lookup_files
        df = df.append(pd.read_csv(self.working_dir + filename))
        self.raw_device_df = df.drop_duplicates(subset=['host'])\
                               .dropna(subset=['location']).dropna(subset=['ip_address'])
        return
