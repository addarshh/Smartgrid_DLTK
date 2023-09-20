# !/usr/bin/env python
# coding: utf-8

import pandas as pd
from utils.utils import get_latest_folder


def flip_labels(master_df, conf):
    """
    : This object flips the labels of events to USEFUL=Y if any event in the swarm is labeled USEFUL=Y
    """
    pso_data_dir = get_latest_folder(conf['filepaths']["train_pso_data_dir"], conf['filepaths']["pso_dir"])
    swarm_label = pd.read_csv(pso_data_dir + '/' + conf['filepaths']["swarm_file"])
    master_df = turn_association_label(master_df)
    master_df = turn_swarm_label(master_df, swarm_label)
    return master_df


def turn_association_label(master_df):
    """
    : Flip the labels from remedy-associated events
    """

    # Relevant columns from the master
    association_master = master_df[['event_id', 'USEFUL?', 'INCIDENT_NUMBER', 'association_label']].copy()
    association_master = association_master[association_master["INCIDENT_NUMBER"].str.contains("INC").fillna(False)]
    master = master_df[['event_id', 'USEFUL?', 'INCIDENT_NUMBER']].copy()

    # Identify association groups with USEFUL label
    association_group_useful = association_master[['USEFUL?', 'association_label']].groupby('association_label')\
                                                                                   .max().reset_index()

    # Merge back to master
    association_master = association_master.drop(columns=['USEFUL?'])\
                                           .merge(association_group_useful, how='left', on=['association_label'])
    master_no_association = master[~master['event_id'].isin(association_master['event_id'])]
    master_flip = master_no_association.append(association_master.drop('association_label', 1))
    master_df = master_df.drop(columns=['USEFUL?'])\
                         .merge(master_flip[['event_id', 'USEFUL?']], how='inner', on=['event_id'])
    return master_df


def turn_swarm_label(master_df, swarm_label):
    """
    : Flip labels of events in the same swarm
    """

    # Relevant columns from the master, and add swarm label
    master = master_df[['event_id', 'USEFUL?', 'INCIDENT_NUMBER']].copy()
    swarm_master = master.merge(swarm_label[['event_id', 'group_label']], how='inner', on='event_id')

    # Identify swarms with a USEFUL label
    swarm_group_useful = swarm_master[['USEFUL?', 'group_label']].groupby('group_label').max().reset_index()

    # Merge back to master
    swarm_master = swarm_master.drop(columns=['USEFUL?']).merge(swarm_group_useful, how='left', on=['group_label'])
    #master_no_swarm = master[~master['event_id'].isin(swarm_master['event_id'])]
    #master_swarm_flip = master_no_swarm.append(swarm_master.drop('group_label', 1))
    master_df = master_df.drop(columns=['USEFUL?'])\
                         .merge(swarm_master[['event_id', 'USEFUL?']], how='inner', on=['event_id'])
    return master_df
