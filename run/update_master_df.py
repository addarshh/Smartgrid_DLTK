#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from preprocessing.RunPreprocessing import create_master_data
from feature_generation.RunFeatureGeneration import create_feature_data
from hyperticketing.RunHyperTicketing import run_pso
from smartpriority.RunSmartPriority import train_smart_priority_model
from postprocessing.RunPostprocessing import run_postprocessing
from BasePath import base_path
from utils.utils import SingleConfig
import sg_splunk
from datetime import datetime, timedelta



def pull_latest_data_for_training(days=30, start_time=None):
    """
    : Pull syslog data for ion3k and global core
    """

    yml = base_path + "data/yaml/input_data_config.yml"
    sg_splunk.set_config(yml)


    now = datetime.utcnow()
    end_date = now.strftime("%m/%d/%Y:%H:%M:%S")

    if start_time is None:
        before = now - timedelta(minutes=days*24 * 60)
        start_date = before.strftime("%m/%d/%Y:%H:%M:%S")
    else:
        start_date = start_time.strftime("%m/%d/%Y:%H:%M:%S")

    concatenate = True

    for device_type in ("ion3k", "global_core"):
        if device_type == "ion3k":
            out_dir = base_path + "data/fc/train_input/phase2"
        if device_type == "global_core":
            out_dir = base_path + "data/gc/train_input/phase2"

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
            sg_splunk.pull_flapping_dashboard(output_dir=out_dir,
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



def pull_previous_master_df(out_dir, device_type):
    """
    : Pull syslog data for ion3k and global core
    """

    yml = base_path + "data/yaml/input_data_config.yml"
    sg_splunk.set_config(yml)

    # MasterDF
    sg_splunk.set_credentials("ah-1064594-001.sdi.corp")
    sg_splunk.pull_data(data_type="master_df",
                        output_dir=out_dir,
                        timing=False,
                        device_type=device_type,
                        name="master_df_previous")
    return



def merge_master_dfs(master_df_folder):
    """
    :
    """

    df1 = pd.read_csv(master_df_folder + 'master_df.csv')
    df2 = pd.read_csv(master_df_folder + 'master_df_previous.csv')
    cols = list(df1.columns)
    df2 = df2[cols]
    df3 = pd.concat([df1, df2])
    df4 = df3.drop_duplicates(['event_id'])
    df4.to_csv(master_df_folder + "master_df.csv", index=False)

    return


def update_master_df_splunk(master_df_folder):
    """
    : Update master_df in splunk index
    """
    import sg_splunk
    from BasePath import base_path
    from utils.splunk_utils import pull_latest_training_date_from_splunk, save_master_df_to_splunk

    yml = base_path + "data/yaml/input_data_config.yml"
    sg_splunk.set_config(yml)
    sg_splunk.set_credentials("ah-1064594-001.sdi.corp")
    last_date = pull_latest_training_date_from_splunk(sg_splunk._inst.splk_u, sg_splunk._inst.splk_p,
                                                      device_type='ion3k')

    master_df = pd.read_csv(master_df_folder + "master_df.csv")
    save_master_df_to_splunk(master_df, last_date)
    return


def run_train(device_type):
    """
    : Run prediction steps
    """

    if device_type == "ion3k":
        yaml_file = base_path + 'data/yaml/fc_train.yml'

    if device_type == "global_core":
        yaml_file = base_path + 'data/yaml/gc_train.yml'


    singleConfig = SingleConfig(yml_file_loc=yaml_file, basepath=base_path)

    preproc_config = singleConfig.preproc_config

    # Preprocessing
    master_df_folder = create_master_data(config=preproc_config, train=True)

    pull_previous_master_df(master_df_folder, device_type)
    merge_master_dfs(master_df_folder)

    # Push updated master_df to splunk
    update_master_df_splunk(master_df_folder)

    return


if __name__ == "__main__":
    pull_latest_data_for_training(days=15)

    run_train(device_type="ion3k")
    run_train(device_type="global_core")