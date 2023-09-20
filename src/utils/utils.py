#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import yaml


def cleanup_output():
    #TODO: Function to delete all but 20 most recent predictions and training output
    return

def get_latest_folder(out_folder, parent_folder):
    """
    : Return latest folder inside specified parent folder
    """
    if out_folder == 'latest':
        out_folder = parent_folder + '/' + pd.Series(
            [i for i in os.listdir(parent_folder) if '20' in i]).max() + '/'
    return out_folder


# Read in the yml config file
class SingleConfig:
    """
    : Configuration object
    """

    class StepConfig():
        pass

    def  __init__(self, yml_file_loc, basepath=""):
        self.preproc_config = self.StepConfig()
        self.hyper_config = self.StepConfig()
        self.smart_config = self.StepConfig()
        self.postproc_config = self.StepConfig()
        self.yml_file_loc = yml_file_loc
        self.basepath = basepath

        with open(self.yml_file_loc) as stream:
            yml_dict = yaml.safe_load(stream)

            for config_key in yml_dict:
                for key in yml_dict[config_key]:
                    d = yml_dict[config_key][key]
                    for subkey in d:
                        if isinstance(d[subkey], str):
                            d[subkey] = d[subkey].replace('basepath/', self.basepath)
                        else:
                            for subsubkey in d[subkey]:
                                if isinstance(d[subkey][subsubkey], str):
                                    d[subkey][subsubkey] = d[subkey][subsubkey].replace('basepath/', self.basepath)
                if config_key == "Preprocessing":
                    self.preproc_config.master = yml_dict[config_key]['master_df']
                    self.preproc_config.feature = yml_dict[config_key]['feature_df']
                if config_key == "HyperTicketing":
                    if 'train_hyper_ticket' in yml_dict[config_key]:
                        self.hyper_config.train = yml_dict[config_key]['train_hyper_ticket']
                    if 'online_hyper_ticket' in yml_dict[config_key]:
                        self.hyper_config.online = yml_dict[config_key]['online_hyper_ticket']
                if config_key == "SmartPriority":
                    if 'train_smart_priority' in yml_dict[config_key]:
                        self.smart_config.train = yml_dict[config_key]['train_smart_priority']
                    if 'online_smart_priority' in yml_dict[config_key]:
                        self.smart_config.online = yml_dict[config_key]['online_smart_priority']
                if config_key == "Postprocessing":
                    self.postproc_config.post_proc = yml_dict[config_key]["post_proc"]
        return


