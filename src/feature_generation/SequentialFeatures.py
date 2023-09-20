#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from scipy.stats import linregress


def rolling_num_features_on_time(master_df, num_features, window_size, forward=True):
    """
    : Function creates forward or backward rolling numeric features
    """

    temp = master_df[['event_id', 'first_occurrence', "device_name", 'event_policy_name_encoded', 'severity_encoded']].copy()
    temp.index = pd.to_datetime(temp['first_occurrence'])
    temp = temp.sort_index()

    cols = ['event_policy_name_encoded', 'severity_encoded']
    prefix = 'pre_'
    if forward:
        prefix = 'post_'
    window_descr = prefix + str(window_size) + 's' + '_'
    a = str(window_size) + 's'
    stats = ['_mode', '_mean', '_var', '_kurt', '_slope']
    cols_new = [[window_descr + i + j for i in cols] for j in stats]

    if forward:
        t = temp[::-1].groupby("device_name")[cols]
        temp[cols_new[0]] = t.transform(lambda s: s.rolling(a).agg(lambda x: pd.Series.mode(x)[0]))[::-1]
        temp[cols_new[1]] = t.transform(lambda s: s.rolling(a).mean())[::-1]
        temp[cols_new[2]] = t.transform(lambda s: s.rolling(a).var())[::-1]
        temp[cols_new[3]] = t.transform(lambda s: s.rolling(a).kurt())[::-1]
        temp[cols_new[4]] = t.transform(lambda s: s.rolling(a).agg(lambda x: linregress(range(len(x)), x)[0]))[::-1]
    else:
        t = temp.groupby("device_name")[cols]
        temp[cols_new[0]] = t.transform(lambda s: s.rolling(a).agg(lambda x: pd.Series.mode(x)[0]))
        temp[cols_new[1]] = t.transform(lambda s: s.rolling(a).mean())
        temp[cols_new[2]] = t.transform(lambda s: s.rolling(a).var())
        temp[cols_new[3]] = t.transform(lambda s: s.rolling(a).kurt())
        temp[cols_new[4]] = t.transform(lambda s: s.rolling(a).agg(lambda x: linregress(range(len(x)), x)[0]))

    rolling_df = temp.drop(columns=['device_name', 'first_occurrence'] + cols).reset_index(drop=True)
    return rolling_df


def pad_single_event(master_df, backward_window, forward_window):
    """
    : pad long time interval with no events with healthy flag
    : just 1 event before and after
    : calculate the time delta of the event (within 75 min)
    """

    temp = master_df[['event_id', 'first_occurrence', "device_name",
                      'event_policy_name', 'event_policy_name_encoded']].copy()

    # get timestamp as index
    temp.index = pd.to_datetime(temp['first_occurrence'])
    temp = temp.sort_index()
    temp['timestamp'] = pd.to_datetime(temp['first_occurrence'])

    temp['time_pre'] = temp.groupby("device_name")['timestamp'].shift(1)
    temp['pre_time_diff'] = temp['timestamp'] - temp['time_pre']
    temp['pre_time_diff'] = temp['pre_time_diff'].fillna(pd.Timedelta(seconds=backward_window))
    temp['pre_duration'] = temp['pre_time_diff'].astype('timedelta64[s]').astype(np.int)
    temp['pre_event'] = temp.groupby("device_name")['event_policy_name_encoded'].shift(1)
    temp.loc[temp['pre_time_diff'] > np.timedelta64(backward_window, 's'), 'pre_event'] = 0
    temp.loc[temp['pre_time_diff'] > np.timedelta64(backward_window, 's'), 'pre_duration'] = backward_window

    temp['time_post'] = temp.groupby("device_name")['timestamp'].shift(-1)
    temp['post_time_diff'] = temp['time_post'] - temp['timestamp']
    temp['post_time_diff'] = temp['post_time_diff'].fillna(pd.Timedelta(seconds=forward_window))
    temp['post_duration'] = temp['post_time_diff'].astype('timedelta64[s]').astype(np.int)
    temp['post_event'] = temp.groupby("device_name")['event_policy_name_encoded'].shift(-1)
    temp.loc[temp['post_time_diff'] > np.timedelta64(forward_window, 's'), 'post_event'] = 0
    temp.loc[temp['post_time_diff'] > np.timedelta64(forward_window, 's'), 'pre_duration'] = forward_window

    data = temp[['event_id', 'post_event', 'post_duration', 'pre_event', 'pre_duration']]

    return data


def padding_time_sequence(master_df, forward_window, backward_window):
    """
    : Get multiple event(sum and mean of the event_policy_name encoded) within time period before and after
    : -- if no events, padded with 0
    """

    temp = master_df[['event_id','first_occurrence', "device_name",
                      'event_policy_name_encoded']].copy()

    # get timestamp as index
    temp.index = pd.to_datetime(temp['first_occurrence']).dt.tz_localize(None)
    temp = temp.sort_index()

    event_ids = [0] * len(master_df)
    forward_num = int(forward_window / 600 + 2)
    backward_num = int(backward_window / 600 + 2)
    if forward_num > 2:
        seq_padded = np.zeros(len(master_df), backward_num * 2 + forward_num * 2)
        columns = ['pre_event_mean_' + str(i) for i in range(backward_num)] + \
                  ['post_event_mean_' + str(i) for i in range(forward_num)] + \
                  ['pre_event_sum_' + str(i) for i in range(backward_num)] + \
                  ['post_event_sum_' + str(i) for i in range(forward_num)]
    else:
        seq_padded = np.zeros((len(master_df), backward_num * 2))
        columns = ['pre_event_mean_' + str(i) for i in range(backward_num)] + \
                  ['pre_event_sum_' + str(i) for i in range(backward_num)]

    temp_idx = 0
    for name, grp in temp.groupby(by=["device_name"]):

        # make new time stamps for truncate
        first_occurrence = grp.index.values
        roll_forward = first_occurrence + np.timedelta64(forward_window, 's')
        roll_backward = first_occurrence - np.timedelta64(backward_window, 's')
        backward_num = int(backward_window / 600 + 2)

        for idx in range(len(grp)):
            event_before_mean = np.array([0] * backward_num)
            event_before_sum = np.array([0] * backward_num)

            single_event_grp_before = grp.truncate(before=roll_backward[idx], after=grp.index[idx])
            if len(single_event_grp_before) > 1:
                event_before_df = single_event_grp_before.resample('10T').mean().fillna(0)
                event_before_mean[backward_num - len(event_before_df):] = event_before_df['event_policy_name_encoded']

                event_before_df = single_event_grp_before.resample('10T').sum().fillna(0)
                event_before_sum[backward_num - len(event_before_df):] = event_before_df['event_policy_name_encoded']

            if forward_num > 2:
                single_event_grp_after = grp.truncate(after=roll_forward[idx], before=grp.index[idx])
                event_after_df = single_event_grp_after.resample('10T').mean().fillna(0)
                event_after_mean = [0] * forward_num
                event_after_mean[0:len(event_after_df):] = event_after_df['event_policy_name_encoded']

                event_after_df = single_event_grp_after.resample('10T').sum().fillna(0)
                event_after_sum = [0] * forward_num
                event_after_sum[0:len(event_after_df):] = event_after_df['event_policy_name_encoded']
                pad_df = pd.DataFrame(event_before_mean + event_after_mean + event_before_sum + event_after_sum).T

            else:
                pad_df = np.concatenate((event_before_mean, event_before_sum), axis=0)

            # concat all of those
            event_ids[temp_idx] = grp.iloc[idx]['event_id']
            seq_padded[temp_idx, :] = pad_df
            temp_idx += 1
            #if temp_idx % 1000 == 0:
            #    print(temp_idx)

    seq_padded = pd.DataFrame(seq_padded)
    seq_padded.columns = columns
    seq_padded['event_id'] = event_ids

    return seq_padded


def interface_features_from_message(master_df):
    """
    : Add binary interface feature
    : TODO: Parametrize with yaml
    """

    master_df['interface_downtime'] = 0
    master_df.loc[master_df['event_policy_name'] == 'DEVICEHW_INTERFACE_DOWN', 'interface_downtime'] = 1

    master_df['interface_flaptime'] = 0
    master_df.loc[master_df['event_policy_name'] == 'DEVICEHW_INTERFACE_DOWN - flaps', 'interface_flaptime'] = 1

    master_df['interface_msg'] = 0
    master_df.loc[master_df['event_policy_name'] == 'DEVICEHW_INTERFACE_HALFDUPLEX', 'interface_msg'] = 1

    # one hot encoding for the interface column
    interface_df = master_df[['event_id', 'interface_downtime', 'interface_flaptime', 'interface_msg']]
    return interface_df
