# !/usr/bin/env python
# coding: utf-8

import yaml
import hashlib


class SyslogNormalizer:

    def __init__(self, normalized_yml_filepath):

        with open(normalized_yml_filepath) as stream:
            try:
                self.yml_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.data = None
        self.device_types = []
        self.normalized_columns = self.yml_dict['normalized_event_df']['fields'] + ['device_type']
        return

    def normalize_ion3k(self, ion3k_df):
        """
        : Convert ION3000 syslog event data to normalized structure
        """
        self.device_types.append('ion3k')
        mapping = self.yml_dict['mapping']['ion3k']
        mapping_inverse = {mapping[key]: key for key in mapping}

        # Rename columns
        ion3k_df = ion3k_df.rename(columns=mapping_inverse)
        ion3k_df['event_id'] = (ion3k_df['device_name'] + "__" + ion3k_df['timestamp']).map(
            lambda x: hashlib.md5(x.encode()).hexdigest())
        ion3k_df['device_type'] = 'ion3k'
        ion3k_df['flap_count'] = 0

        return ion3k_df[self.normalized_columns]

    def normalize_global_core(self, global_core_df):
        """
        : Convert Global Core syslog event data to normalized structure
        """
        self.device_types.append('global_core')
        mapping = self.yml_dict['mapping']['global_core']
        mapping_inverse = {mapping[key]: key for key in mapping}

        # Rename columns
        global_core_df = global_core_df.rename(columns=mapping_inverse)
        global_core_df['device_name'] = global_core_df['device_name'].map(lambda x: x.split(".")[0].upper())
        global_core_df['event_id'] = (global_core_df['device_name'] + "__" + global_core_df['timestamp']).map(
            lambda x: hashlib.md5(x.encode()).hexdigest())
        global_core_df['device_type'] = 'global_core'

        return global_core_df[self.normalized_columns]
