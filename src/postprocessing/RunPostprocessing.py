#!/usr/bin/env python
# coding: utf-8


from postprocessing.MergeOutput import MergeOutput
import shutil
import pandas as pd
import os
from utils.splunk_utils import stream_final_output


def run_postprocessing(config, train, master_df_folder=None):

    if train:
        # Copy files to /data/predict_input
        print("Copying training output to prediction input directory.")
        assert master_df_folder is not None

        latest_predict_folder = config.post_proc['filepaths']['prediction_folder_dir'] + "latest/"

        if os.path.exists(latest_predict_folder):
            shutil.rmtree(latest_predict_folder)
        shutil.copytree(master_df_folder, latest_predict_folder)

        return

    # Merge output and save to csv
    print('Saving dashboard csv.')
    merge_output_obj = MergeOutput(config)
    merge_output_obj.merge_data(config)
    master_df = pd.read_csv(master_df_folder + '/' + config.post_proc['filepaths']["master_file_name"])
    merge_output_obj.save_output(config, master_df, train)

    # Try saving output to splunk index
    if config.post_proc['parameters']["streaming_to_index"]:
        response = stream_final_output(merge_output_obj.final_output, config.post_proc['parameters']["test_number"])
        if response.status_code == 200:
            print('Model output saved to Splunk index.')
        else:
            print('Failed to send output to Splunk index.')

    return merge_output_obj.final_output
