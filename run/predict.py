#!/usr/bin/env python
# coding: utf-8

from preprocessing.RunPreprocessing import create_master_data
from feature_generation.RunFeatureGeneration import create_feature_data
from hyperticketing.RunHyperTicketing import run_online_filtering
from smartpriority.RunSmartPriority import run_online_prediction
from postprocessing.RunPostprocessing import run_postprocessing
from BasePath import base_path
from utils.utils import SingleConfig
import sg_splunk
from datetime import datetime, timedelta


def pull_latest_data_for_prediction(last_time, hours=4):
    """
    : Pull syslog data for ion3k and global core
    """

    yml = base_path + "data/yaml/input_data_config.yml"
    sg_splunk.set_config(yml)

    sg_splunk.set_credentials("ecslsplunk-ti")

    now = datetime.utcnow()
    before = last_time - timedelta(minutes=hours * 60)
    end_date = now.strftime("%m/%d/%Y:%H:%M:%S")
    start_date = before.strftime("%m/%d/%Y:%H:%M:%S")

    for device_type in ("ion3k", "global_core"):
        if device_type == "ion3k":
            out_dir = base_path + "data/fc/predict_input/phase2"
        if device_type == "global_core":
            out_dir = base_path + "data/gc/predict_input/phase2"
            sg_splunk.set_credentials("ecslsplunk-ti")
            sg_splunk.pull_dashboard(output_dir=out_dir,
                                     end_date=end_date,
                                     start_date=start_date,
                                     concatenate=True)

        sg_splunk.set_credentials("ecslsplunk-ti")
        sg_splunk.pull_data(data_type="syslog_data_" + device_type,
                        output_dir=out_dir,
                        device_type=device_type,
                        end_date=end_date,
                        start_date=start_date,
                        name="syslog_data_" + device_type)

        sg_splunk.set_credentials("reli0")
        sg_splunk.pull_data(data_type="cmdb_info",
                            device_type=device_type,
                            output_dir=out_dir,
                            timing=False,
                            url='url2',
                            token=True)
    return


def run_prediction(device_type, time_cut_off):
    """
    : Run prediction steps
    """

    if device_type == "ion3k":
        yaml_file = base_path + 'data/yaml/fc_predict.yml'

    if device_type == "global_core":
        yaml_file = base_path + 'data/yaml/gc_predict.yml'

    singleConfig = SingleConfig(yml_file_loc=yaml_file, basepath=base_path)

    preproc_config = singleConfig.preproc_config
    hyper_config = singleConfig.hyper_config
    smart_config = singleConfig.smart_config
    postproc_config = singleConfig.postproc_config

    # params
    train = False
    preproc_config.master_single_file = True

    # create master data from data sources
    master_df = create_master_data(config=preproc_config, train=train)

    # assign swarm labels to events
    run_online_filtering(config=hyper_config)

    # create feature data from master data
    create_feature_data(config=preproc_config, train=train)

    # run classification model
    run_online_prediction(config=smart_config, time_cut_off=time_cut_off)

    postproc_config.post_proc['parameters']["streaming_to_index"] = True
    postproc_config.post_proc['parameters']["test_number"] = 3

    # save output to either index or csv
    run_postprocessing(postproc_config, train, master_df)

    return

def get_last_prediction_time():
    """
    :
    """
    import sg_splunk
    from BasePath import base_path
    from utils.splunk_utils import pull_latest_prediction_date_from_splunk

    yml = base_path + "data/yaml/input_data_config.yml"
    sg_splunk.set_config(yml)
    sg_splunk.set_credentials("ah-1064594-001.sdi.corp")
    ion3k = pull_latest_prediction_date_from_splunk(sg_splunk._inst.splk_u,
                                                      sg_splunk._inst.splk_p,
                                                      device_type='ion3k')

    global_core = pull_latest_prediction_date_from_splunk(sg_splunk._inst.splk_u,
                                                      sg_splunk._inst.splk_p,
                                                      device_type='global_core')

    ion3k = datetime.strptime(ion3k.values[0,0][:10] + " " + ion3k.values[0,0][11:19], "%Y-%m-%d %H:%M:%S")
    global_core = datetime.strptime(global_core.values[0,0][:10] + " " + global_core.values[0,0][11:19], "%Y-%m-%d %H:%M:%S")

    return ion3k, global_core


def file_cleanup(device_type):
    """
    : Clean up prediction output files
    """
    if device_type == "ion3k":
        yaml_file = base_path + 'data/yaml/fc_predict.yml'

    if device_type == "global_core":
        yaml_file = base_path + 'data/yaml/gc_predict.yml'

    singleConfig = SingleConfig(yml_file_loc=yaml_file, basepath=base_path)

    postproc_config = singleConfig.postproc_config

    prediction_folder = postproc_config.post_proc['filepaths']['prediction_folder_dir']
    archive_folder = postproc_config.post_proc['filepaths']['prediction_folder_dir'].replace('prediction', 'archive')

    import os
    import glob
    import shutil

    files = glob.glob(prediction_folder + "pso_online/*")

    if len(files) > 200:
        f1 = files[-1]

        shutil.rmtree(archive_folder)
        os.rename(prediction_folder, archive_folder)

        os.makedirs(prediction_folder + 'dashboard')
        os.makedirs(prediction_folder + 'pso_online')

        files = glob.glob(archive_folder + "pso_online/*")
        f2 = files[-1]
        os.rename(f2, f1)

    return


if __name__ == "__main__":

    assert False, "This should only be run on Phoenix"

    ion3k_time, global_core_time = get_last_prediction_time()
    pull_latest_data_for_prediction(ion3k_time, hours=4)

    run_prediction(device_type="ion3k", time_cut_off=ion3k_time)
    run_prediction(device_type="global_core", time_cut_off=global_core_time)

    file_cleanup("ion3k")
    file_cleanup("global_core")