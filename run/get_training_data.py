#!/usr/bin/env python
# coding: utf-8

import sg_splunk
from datetime import datetime, timedelta
from BasePath import base_path
import os


def build_directory_structure():
    """
    : Build input and output directories
    """
    for datatype in ("fc", "gc"):
        os.makedirs(base_path + "data/" + datatype + "/train_input/phase2/", exist_ok=True)
        os.makedirs(base_path + "data/" + datatype + "/train_output/", exist_ok=True)
        os.makedirs(base_path + "data/" + datatype + "/predict_output/prediction/dashboard/", exist_ok=True)
        os.makedirs(base_path + "data/" + datatype + "/predict_input/latest/", exist_ok=True)
        os.makedirs(base_path + "data/" + datatype + "/predict_input/phase2/", exist_ok=True)
    return


def pull_training_data_remedy(end_date=None, days=1, out_dir=None, device_type="ion3k"):
    """
    : Pull Remedy Incident data
    """


    yml = base_path + "data/yaml/input_data_config.yml"
    sg_splunk.set_config(yml)

    sg_splunk.set_credentials("ecslsplunk-ti")

    if end_date is None:
        now = datetime.utcnow()
    else:
        now = datetime.strptime(end_date, "%Y-%m-%d")
    before = now - timedelta(minutes=days * 24 * 60)
    end_date = now.strftime("%m/%d/%Y:%H:%M:%S")
    start_date = before.strftime("%m/%d/%Y:%H:%M:%S")

    out_dir = base_path + out_dir

    sg_splunk.pull_data(data_type="remedy_incident_data",
                        output_dir=out_dir,
                        device_type=device_type,
                        end_date=end_date,
                        start_date=start_date)
    return


def pull_training_data_syslog(end_date=None, days=1, out_dir=None, device_type="ion3k"):
    """
    : Pull Syslog data
    """


    yml = base_path + "data/yaml/input_data_config.yml"
    sg_splunk.set_config(yml)

    sg_splunk.set_credentials("ecslsplunk-ti")

    if end_date is None:
        now = datetime.utcnow()
    else:
        now = datetime.strptime(end_date, "%Y-%m-%d")
    before = now - timedelta(minutes=days * 24 * 60)
    end_date = now.strftime("%m/%d/%Y:%H:%M:%S")
    start_date = before.strftime("%m/%d/%Y:%H:%M:%S")

    out_dir = base_path + out_dir

    sg_splunk.pull_data(data_type="syslog_data_" + device_type,
                        output_dir=out_dir,
                        device_type=device_type,
                        end_date=end_date,
                        start_date=start_date,
                        name="syslog_data_" + device_type)
    return


def pull_training_data_remedy_association(end_date=None, days=1, out_dir=None, device_type="ion3k"):
    """
    : Pull Remedy Association data
    """

    yml = base_path + "data/yaml/input_data_config.yml"
    sg_splunk.set_config(yml)

    sg_splunk.set_credentials("ah-1064594-001.sdi.corp")

    if end_date is None:
        now = datetime.utcnow()
    else:
        now = datetime.strptime(end_date, "%Y-%m-%d")
    before = now - timedelta(minutes=days * 24 * 60)
    end_date = now.strftime("%m/%d/%Y:%H:%M:%S")
    start_date = before.strftime("%m/%d/%Y:%H:%M:%S")

    out_dir = base_path + out_dir

    sg_splunk.pull_data(data_type="remedy_associations",
                        output_dir=out_dir,
                        end_date=end_date,
                        start_date=start_date)

    return

def pull_feedback_data(end_date=None, days=14, out_dir=None, device_type="ion3k"):
    """
    : Pull Feedback Dashboard data
    """

    yml = base_path + "data/yaml/input_data_config.yml"
    sg_splunk.set_config(yml)

    sg_splunk.set_credentials("ah-1064594-001.sdi.corp")

    if end_date is None:
        now = datetime.utcnow()
    else:
        now = datetime.strptime(end_date, "%Y-%m-%d")
    before = now - timedelta(minutes=days * 24 * 60)
    end_date = now.strftime("%m/%d/%Y:%H:%M:%S")
    start_date = before.strftime("%m/%d/%Y:%H:%M:%S")

    out_dir = base_path + out_dir

    data_type_text = "feedback_data_" + device_type

    sg_splunk.pull_data(data_type=data_type_text,
                        output_dir=out_dir,
                        end_date=end_date,
                        start_date=start_date)

    return

def pull_training_data_cmdb_info(out_dir=None, device_type="ion3k"):
    """
    : Pull CMDB Info data
    """

    yml = base_path + "data/yaml/input_data_config.yml"
    sg_splunk.set_config(yml)

    sg_splunk.set_credentials("ecslsplunk-ti")

    out_dir = base_path + out_dir

    sg_splunk.pull_data(data_type="cmdb_info",
                        device_type=device_type,
                        output_dir=out_dir,
                        timing=False)
    return


def pull_gc_flap_data_dashboard(end_date=None, days=1, out_dir=None, device_type="global_core"):
    out_dir = base_path + out_dir

    yml = base_path + "data/yaml/input_data_config.yml"
    sg_splunk.set_config(yml)

    sg_splunk.set_credentials("ecslsplunk-ti")

    if end_date is None:
        now = datetime.utcnow()
    else:
        now = datetime.strptime(end_date, "%Y-%m-%d")
    before = now - timedelta(minutes=days * 24 * 60)
    end_date = now.strftime("%m/%d/%Y:%H:%M:%S")
    start_date = before.strftime("%m/%d/%Y:%H:%M:%S")


    sg_splunk.pull_dashboard(output_dir=out_dir,
                            end_date=end_date,
                            start_date=start_date)
    return



def pull_ion3k(days=14, prediction_data=False):
    """
    : Pull training data for ion3k
    """
    train_data_folder = 'data/fc/train_input/feb2022'
    end_date = '2022-02-28'

    pull_training_data_remedy(end_date=end_date, days=days, out_dir=train_data_folder, device_type="ion3k")
    pull_training_data_syslog(end_date=end_date, days=days, out_dir=train_data_folder, device_type="ion3k")
    pull_training_data_remedy_association(end_date=end_date, days=120, out_dir=train_data_folder, device_type="ion3k")
    pull_training_data_cmdb_info(out_dir=train_data_folder, device_type="ion3k")

    if prediction_data:
        predict_data_folder = "data/fc/predict_input/phase2"
        pull_training_data_syslog(days=1, out_dir=predict_data_folder, device_type="ion3k")
        pull_training_data_cmdb_info(out_dir=predict_data_folder, device_type="ion3k")

    return


def pull_global_core(days=14, prediction_data=False):
    """
    : Pull training data for global_core
    """
    train_data_folder = 'data/gc/train_input/feb2022'
    end_date = '2022-02-28'

    pull_gc_flap_data_dashboard(end_date=end_date, days=days, out_dir=train_data_folder, device_type="global_core")
    pull_training_data_remedy(end_date=end_date, days=days, out_dir=train_data_folder, device_type="global_core")
    pull_training_data_syslog(end_date=end_date, days=days, out_dir=train_data_folder, device_type="global_core")
    pull_training_data_remedy_association(end_date=end_date, days=120, out_dir=train_data_folder, device_type="global_core")
    pull_training_data_cmdb_info(out_dir=train_data_folder, device_type="global_core")

    if prediction_data:
        predict_data_folder = "data/gc/predict_input/phase2"

        pull_training_data_syslog(days=1, out_dir=predict_data_folder, device_type="global_core")
        pull_training_data_cmdb_info(out_dir=predict_data_folder, device_type="global_core")
        pull_gc_flap_data_dashboard(days=1, out_dir=predict_data_folder, device_type="global_core")

    return


if __name__ == "__main__":

    # Build any missing directories
    build_directory_structure()


    # Uncomment to pull data for training and prediction
    pull_ion3k(days=30)
    pull_global_core(days=30)

