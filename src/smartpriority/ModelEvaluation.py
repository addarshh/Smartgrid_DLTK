#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


# evaluate model
class ModelEvaluation:

    def __init__(self, mp, master_file_dir, swarm_file_dir, config):
        """
        : Evaluate model performance
        """
        self.mp = mp

        self.master_df = pd.read_csv(master_file_dir)
        self.swarm_df = pd.read_csv(swarm_file_dir)

        self.calculate_classification_metrics(mp, threshold=config.train['parameters']["classification_threshold"])
        return

    def calculate_classification_metrics(self, mp, threshold):
        """
        : Binary classification metrics
        """

        # Calculate metrics for top clusters
        xtest = mp.X_test.drop(columns=["event_id"])
        xtest = xtest.drop(mp.drop_column, axis=1, errors='ignore')
        self.y_test_prob = mp.model_fit.predict_proba(xtest)[:, 1]
        self.y_test_pred = (self.y_test_prob >= threshold).astype('int')

        self.y_train_prob = mp.model_fit.predict_proba(mp.X_train)[:, 1]
        self.y_train_pred = (self.y_train_prob >= threshold).astype('int')

        # Grouped by swarm
        test_df = mp.X_test
        test_df['Useful_pred'] = self.y_test_pred
        merged_test_results_by_swarm = test_df.merge(self.swarm_df[['group_label', 'event_id']],
                                                     how='left', on=['event_id'])
        grouped_test_by_swarm = merged_test_results_by_swarm[['Useful_pred', 'USEFUL?', 'group_label']].groupby(
            ['group_label']).mean().reset_index()
        grouped_test_by_swarm['USEFUL?'] = np.where(grouped_test_by_swarm['USEFUL?'] > 0, 1, 0)
        grouped_test_by_swarm['Useful_pred'] = np.where(grouped_test_by_swarm['Useful_pred'] > 0, 1, 0)

        #print("Results for Individual Events (Train Set)")
        #self.print_confusion_matrix(self.y_train_pred, mp.y_train)

        #print("Results for Individual Events (Test Set)")
        #self.print_confusion_matrix(self.y_test_pred, mp.y_test)

        print('Results Grouped by Swarm')
        self.print_confusion_matrix(grouped_test_by_swarm['Useful_pred'], grouped_test_by_swarm["USEFUL?"])
        return

    def create_output_df(self):
        """
        : Creates model output dataframe
        """
        mp = self.mp

        test_df = mp.X_test
        test_df['Useful_pred'] = self.y_test_pred
        test_df['Useful_prob'] = self.y_test_prob
        test_df['test'] = 1

        results = pd.merge(test_df, self.swarm_df[['group_label', 'event_id']],
                                      how='left', on=['event_id'])
        grouped_by_swarm = results[['Useful_pred', 'USEFUL?', 'group_label']].groupby(
            ['group_label']).sum().reset_index()
        grouped_by_swarm['USEFUL?'] = np.where(grouped_by_swarm['USEFUL?'] > 0, 1, 0)
        grouped_by_swarm['Useful_pred'] = np.where(grouped_by_swarm['Useful_pred'] > 0, 1, 0)
        grouped_by_swarm = grouped_by_swarm.rename(columns={'USEFUL?': 'GROUP_USEFUL?',
                                                            'Useful_pred': 'Group_Useful_pred'})
        
        results = pd.merge(results, grouped_by_swarm, how='left', on=['group_label'])
        results = pd.merge(results, self.master_df.drop(columns="USEFUL?"), on=["event_id"], how="left")

        # event confusion
        results['confusion'] = 'TP'
        results.loc[(results['USEFUL?'] == 0) & (results['Useful_pred'] == 1), 'confusion'] = 'FP'
        results.loc[(results['USEFUL?'] == 0) & (results['Useful_pred'] == 0), 'confusion'] = 'TN'
        results.loc[(results['USEFUL?'] == 1) & (results['Useful_pred'] == 0), 'confusion'] = 'FN'

        # group confusion
        results['group_confusion'] = 'TP'
        results.loc[(results['GROUP_USEFUL?'] == 0) & (results['Group_Useful_pred'] == 1), 'group_confusion'] = 'FP'
        results.loc[(results['GROUP_USEFUL?'] == 0) & (results['Group_Useful_pred'] == 0), 'group_confusion'] = 'TN'
        results.loc[(results['GROUP_USEFUL?'] == 1) & (results['Group_Useful_pred'] == 0), 'group_confusion'] = 'FN'
        return results

    @staticmethod
    def get_feature_importance(mp):
        """
        : Returns feature importance
        """
        feat_imp_df = pd.DataFrame([mp.X_train.columns, mp.model_fit.feature_importances_]).T
        feat_imp_df.columns = ['Column', 'Feature Importance']
        print(feat_imp_df.sort_values(by='Feature Importance', ascending=False)[:10])
        return feat_imp_df

    @staticmethod
    def print_confusion_matrix(prediction, actual):
        """
        : Print confusion matrix and metrics
        """
        try:
            print('\n' + classification_report(actual, prediction) + '\n    AUC ',
                  roc_auc_score(actual, prediction))
        except:
            pass
        print(confusion_matrix(actual, prediction))
        return