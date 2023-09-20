#!/usr/bin/env python
# coding: utf-8


from smartpriority.ModelEvaluation import ModelEvaluation
from smartpriority.ModelPipeline import ModelPipeline
from smartpriority.OnlinePrediction import OnlinePrediction
from utils.utils import get_latest_folder


def train_smart_priority_model(config):
    """
    : Train smart priority model
    """

    # Find file paths for master_df.csv and swarm_labels.csv
    feature_file_dir = config.train["filepaths"]["master_df_dir"] + '/feature_df.csv'
    master_file_dir = config.train['filepaths']["master_df_dir"] + '/master_df.csv'
    swarm_file_dir = config.train["filepaths"]["saved_swarm_folder_dir"] + '/swarm_label.csv'
    print('Getting master and feature data from ', config.train['filepaths']["master_df_dir"])
    print('Getting swarm labels from ', swarm_file_dir)

    # start pipeline
    mp = ModelPipeline(feature_file_dir, config)

    # Evaluate model and calculate kpi
    model_eval = ModelEvaluation(mp, master_file_dir, swarm_file_dir, config)

    # Uncomment to generate feature importance csv
    #feature_importance_df = model_eval.get_feature_importance(mp)
    #feature_importance_df.to_csv(config.train['filepaths']["master_df_dir"] + "/feature_importance.csv")

    # Save model weights and output
    mp.save_model_files(model_eval,
                        threshold=config.train['parameters']["classification_threshold"],
                        path=config.train['filepaths']["master_df_dir"])
    return mp, model_eval


def run_online_prediction(config, time_cut_off=None):
    """
    : Run online prediction
    """

    # Find file paths for master_df.csv and swarm_labels.csv
    saved_model_folder = config.online["filepaths"]["saved_model_folder_dir"]
    swarm_file_dir = get_latest_folder(config.online["filepaths"]["saved_swarm_folder_dir"],
                                       config.online["filepaths"]["saved_swarm_dir"]) + '/swarm_label.csv'
    print('getting swarm labels from ', swarm_file_dir)
    print('getting saved model files from ', saved_model_folder)

    online_prediction_obj = OnlinePrediction(saved_model_folder=saved_model_folder,
                                             feature_file_dir=config.online["filepaths"]["feature_file_dir"],
                                             swarm_file_dir=swarm_file_dir)
    online_prediction_obj.feature_selection(time_cut_off=time_cut_off)
    online_prediction_obj.preprocess()
    online_prediction_obj.post_process_clusters()
    online_prediction_obj.load_classification_model()
    online_prediction_obj.prediction()
    online_prediction_obj.make_output_file()
    online_prediction_obj.save_prediction_files(path=config.online["filepaths"]["save_output_dir"])
    return online_prediction_obj


def run_model_evaluation(config, saved_model_folder):
    """
    :
    """

    # Find file paths for master_df.csv and swarm_labels.csv
    feature_file_dir = config.train["filepaths"]["master_df_dir"] + '/feature_df.csv'
    master_file_dir = config.train['filepaths']["master_df_dir"] + '/master_df.csv'
    swarm_file_dir = config.train["filepaths"]["saved_swarm_folder_dir"] + '/swarm_label.csv'
    print('Getting master and feature data from ', config.train['filepaths']["master_df_dir"])
    print('Getting swarm labels from ', swarm_file_dir)

    import pickle
    mp = ModelPipeline(feature_file_dir, config, fit_model=False)
    mp.model_fit = pickle.load(open(saved_model_folder + 'classification_model_file.pkl', 'rb'))

    model_eval = ModelEvaluation(mp, master_file_dir, swarm_file_dir, config)

    return