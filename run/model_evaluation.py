#!/usr/bin/env python
# coding: utf-8
import os

from preprocessing.RunPreprocessing import create_master_data
from feature_generation.RunFeatureGeneration import create_feature_data
from hyperticketing.RunHyperTicketing import run_pso
from smartpriority.RunSmartPriority import run_model_evaluation
from preprocessing.RunPreprocessing import create_master_data
from feature_generation.RunFeatureGeneration import create_feature_data
from hyperticketing.RunHyperTicketing import run_online_filtering
from smartpriority.RunSmartPriority import run_online_prediction
from postprocessing.RunPostprocessing import run_postprocessing
from BasePath import base_path
from utils.utils import SingleConfig
import sg_splunk
from datetime import datetime, timedelta


def pull_latest_data_for_evaluation(days=30):
    """
    : Pull syslog data for ion3k and global core
    """

    yml = base_path + "data/yaml/input_data_config.yml"
    sg_splunk.set_config(yml)


    now = datetime.utcnow()
    before = now - timedelta(minutes=days*24 * 60)
    end_date = now.strftime("%m/%d/%Y:%H:%M:%S")
    start_date = before.strftime("%m/%d/%Y:%H:%M:%S")
    concatenate = True

    for device_type in ("ion3k", "global_core"):
        if device_type == "ion3k":
            out_dir = base_path + "data/fc/train_input/evaluation"
        if device_type == "global_core":
            out_dir = base_path + "data/gc/train_input/evaluation"

        os.makedirs(out_dir, exist_ok=True)

        sg_splunk.set_credentials("ecslsplunk-ti")
        sg_splunk.pull_data(data_type="syslog_data_" + device_type,
                        output_dir=out_dir,
                        device_type=device_type,
                        end_date=end_date,
                        start_date=start_date,
                        name="syslog_data_" + device_type,
                        concatenate=concatenate)

        sg_splunk.set_credentials("reli0")
        sg_splunk.pull_data(data_type="cmdb_info",
                            device_type=device_type,
                            output_dir=out_dir,
                            timing=False,
                            concatenate=concatenate,
                            url='url2',
                            token=True)

        sg_splunk.pull_data(data_type="remedy_incident_data",
                            output_dir=out_dir,
                            device_type=device_type,
                            end_date=end_date,
                            start_date=start_date,
                            concatenate=concatenate,
                            url='url2',
                            token=True)

        if device_type == "global_core":
            sg_splunk.set_credentials("ecslsplunk-ti")
            sg_splunk.pull_dashboard(output_dir=out_dir,
                                     end_date=end_date,
                                     start_date=start_date,
                                     concatenate=concatenate)

        sg_splunk.set_credentials("ah-1064594-001.sdi.corp")
        sg_splunk.pull_data(data_type="remedy_associations",
                            output_dir=out_dir,
                            end_date=end_date,
                            start_date=start_date,
                            concatenate=concatenate)

        data_type_text = "feedback_data_" + device_type
        sg_splunk.pull_data(data_type=data_type_text,
                            output_dir=out_dir,
                            end_date=end_date,
                            start_date=start_date,
                            concatenate=concatenate)

    return


def run_train(device_type):
    """
    : Run prediction steps
    """

    if device_type == "ion3k":
        yaml_file = base_path + 'data/yaml/fc_train.yml'
        yaml_file2 = base_path + 'data/yaml/fc_predict.yml'

    if device_type == "global_core":
        yaml_file = base_path + 'data/yaml/gc_train.yml'
        yaml_file2 = base_path + 'data/yaml/gc_predict.yml'

    os.makedirs(base_path + "data/fc/train_output/evaluation", exist_ok=True)
    os.makedirs(base_path + "data/gc/train_output/evaluation", exist_ok=True)


    singleConfig = SingleConfig(yml_file_loc=yaml_file, basepath=base_path)

    preproc_config = singleConfig.preproc_config
    smart_config = singleConfig.smart_config
    postproc_config = singleConfig.postproc_config

    singleConfig2 = SingleConfig(yml_file_loc=yaml_file2, basepath=base_path)
    hyper_config = singleConfig2.hyper_config

    p = preproc_config.master
    p['filepaths']['raw_data_dir'] = p['filepaths']['raw_data_dir'].replace("phase2", "evaluation")
    p['filepaths']['save_data_dir'] = p['filepaths']['save_data_dir'] + '/evaluation'
    p['filepaths']['user_label_dir'] = p['filepaths']['user_label_dir'].replace("phase2", "evaluation")

    # Preprocessing
    master_df_folder = create_master_data(config=preproc_config, train=True)

    # Hyper Ticketing
    hyper_config.online['filepaths']["master_df_name"] = master_df_folder + '/master_df.csv'
    hyper_config.online['filepaths']["save_swarm_dir"] = master_df_folder + '/pso_filters_run'
    run_online_filtering(config=hyper_config)

    # Feature Generation
    from utils.utils import get_latest_folder
    a = get_latest_folder('latest', master_df_folder + "/pso_filters_run")
    preproc_config.feature['filepaths']["train_pso_data_dir"] = a
    preproc_config.feature['filepaths']["train_master_data_dir"] = master_df_folder
    create_feature_data(config=preproc_config, train=True)

    # Smart Priority Training
    smart_config.train['filepaths']["master_df_dir"] = master_df_folder
    smart_config.train['filepaths']["saved_swarm_folder_dir"] = a
    saved_model_folder = postproc_config.post_proc['filepaths']['prediction_folder_dir'] + 'latest/model_pipeline_run/'
    run_model_evaluation(config=smart_config, saved_model_folder=saved_model_folder)

    return


if __name__ == "__main__":
    #pull_latest_data_for_evaluation(days=5)
    #run_train(device_type="ion3k")
    run_train(device_type="global_core")