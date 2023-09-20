# !/usr/bin/env python
# coding: utf-8

from __future__ import division
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score


def filter_by_time(df, window):
    """
    : Helper function to get swarm from events based on time and location
    """
    test = df.sort_values(by=['epoch_time']).copy()
    test['duration'] = test['epoch_time'].diff(periods=1)
    test['duration_within_window'] = np.where(test['duration'] <= window, 1, 0)
    test['duration_within_window_shift'] = test['duration_within_window'].diff(periods=1)
    test['group_start'] = np.where(test['duration_within_window_shift'] == 1, 1, 0)
    test['group_end'] = np.where(test['duration_within_window_shift'] == -1, 1, 0)
    test['group_start'] = test.group_start.shift(-1).fillna(0)
    test['group_end'] = test.group_end.shift(-1).fillna(0)
    test['group'] = test['duration_within_window'] + test['group_start']
    test['group_label'] = test['group_start'].cumsum()
    test['group_label'] = test['group_label'] * test['group']
    test['group_label'] = np.where(test['group_label'] == 0, np.nan, test['group_label'])
    try:
        start = int(test['group_label'].max() + 1)
    except Exception:
        start = 0
    fill_na = [i for i in range(start, int(sum(test['group_label'].isna()) + start))]
    na_df = test.loc[test['group_label'].isna()]
    na_df['group_label'] = pd.Series(fill_na).values
    test.loc[test['group_label'].isna(), 'group_label'] = na_df['group_label']
    if test['group_label'].min() != 0:
        test['group_label'] = test['group_label'] - 1
    test['num_group'] = test['group_label'].nunique()
    return test


def spatial_temp_filter(master, rolling_window, spatial_filter):
    """
    : Perform spatial and temporal grouping
    """
    col = ['event_id', 'epoch_time', 'city_code', 'location', 'ip_group', 'device_name', 'event_policy_name']
    group_df = master[col].groupby(by=[spatial_filter]).apply(filter_by_time, window=rolling_window).reset_index(
        drop=True)
    num_group = group_df[[spatial_filter, 'num_group']].groupby(by=spatial_filter).head(1)
    num_group['num_group_sum'] = num_group['num_group'].cumsum()
    num_group['num_group_shift'] = num_group['num_group_sum'].shift(1).fillna(0)
    group_df = pd.merge(group_df, num_group, how='inner', on=[spatial_filter, 'num_group'])
    group_df['group_label'] = group_df['group_label'] + group_df['num_group_shift']
    return group_df


class EventGroupEval:
    """
    : For the purpose of finding the best filtering conditions
    :  this holds the cost function to optimize the PSO
    """

    def __init__(self, association_event_df, eval_beta, rolling_window, spatial_filter):
        self.eval_beta = eval_beta
        self.rolling_window = rolling_window
        self.spatial_filter = spatial_filter
        self.association_event_df = association_event_df

        self.max_span = None
        self.eval_metric_jaccard = None

        self.group_df = spatial_temp_filter(self.association_event_df,
                                            self.rolling_window,
                                            self.spatial_filter)
        return

    def swarm_eval(self, association_event_df):
        """
        : Evaluate the score of the swarm versus the associated event
        """

        # max span in a swarm
        self.max_span = self.group_df[['epoch_time', 'group_label']].groupby(by=['group_label']).agg(
            lambda x: np.max(x) - np.min(x))['epoch_time'].max()

        # compare to association table
        a = self.group_df[['event_id', 'group_label']]
        b = association_event_df[association_event_df["IS_REMEDY"] == 1]
        b = b[['event_id', 'association_label']]
        compare_association = pd.merge(a, b, how='inner', on=['event_id'])
        per_swarm = self.get_mean_jaccard_score(compare_association, 'group_label', 'association_label')
        per_association = self.get_mean_jaccard_score(compare_association, 'association_label', 'group_label')

        # Calculate eval_metric based on the weight
        a = self.eval_beta * per_swarm + per_association
        b = (1 + self.eval_beta) * per_swarm * per_association
        self.eval_metric_jaccard = a / b
        return

    @staticmethod
    def get_mean_jaccard_score(compare_association, label_a, label_b):
        """
        : Loops through comparison to get jaccard score for label_a
        """
        jaccard_score_per_label_a = []
        for label in compare_association[label_a].unique():
            group1 = (compare_association[label_a] == label).astype(int)
            label1 = list(compare_association.loc[compare_association[label_a] == label, label_b].unique())
            group2 = (compare_association[label_b].isin(label1)).astype(int)
            jaccard_score_per_label_a.append(jaccard_score(group2, group1))
        return np.mean(np.array(jaccard_score_per_label_a))


def cost_function(particle, beta, percent_penalty, association_event_df):
    """
    : Cost function: Overall Jaccard score
    """
    event_group_eval_obj = EventGroupEval(association_event_df,
                                          eval_beta=beta,
                                          rolling_window=particle.window,
                                          spatial_filter=particle.spatial_filter)
    event_group_eval_obj.swarm_eval(association_event_df)
    total = event_group_eval_obj.eval_metric_jaccard
    penalty = (event_group_eval_obj.max_span / (60 * 60 * 24 * 7)) * percent_penalty
    return total + penalty
