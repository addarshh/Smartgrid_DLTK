# !/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
import yaml
from BasePath import base_path
from utils.encrypt import decrypt_message
from utils.splunk_utils import fetch_data_user, fetch_data_token, build_splunk_query


class SplunkDataModel:

    def __init__(self):

        with open(base_path + "data/yaml/credentials.yml") as stream:
            try:
                self.cred_yml = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.splk_u = None
        self.splk_p = None
        self.headers = None
        self.output_dir = None

        self.splk_query = None
        self.start_date = None
        self.end_date = None
        self.yml_dict = None
        self.splk_url = None
        return

    def fetch_data(self, filename, concatenate=False, token=False):
        """
        : Wrapper for fetching data from splunk with an HTTP request
        """
        if token:
            fetch_data_token(self, filename, concatenate)
        else:
            fetch_data_user(self, filename, concatenate)
        return

    def _set_start_end(self, start_date, end_date):
        if start_date and end_date:
            self.start_date = start_date
            self.end_date = end_date
        else:
            now = datetime.utcnow()
            before = datetime.utcnow() - timedelta(minutes=240)
            self.end_date = now.strftime("%m/%d/%Y:%H:%M:%S")
            self.start_date = before.strftime("%m/%d/%Y:%H:%M:%S")
        return

    def _set_output_dir(self, output_dir):
        self.output_dir = output_dir
        return

    def set_credentials(self, splk_env):
        """
        : Set username and pwd
        """
        if 'pwd' in self.cred_yml[splk_env]:
            self.splk_u = decrypt_message(self.cred_yml[splk_env]['user'].encode())
            self.splk_p = decrypt_message(self.cred_yml[splk_env]['pwd'].encode())
        if 'token' in self.cred_yml[splk_env]:
            auth_token = decrypt_message(self.cred_yml[splk_env]['token'].encode())
            self.headers = {'Content-type': 'application/json', 'Authorization': 'Bearer ' + auth_token}
        return

    def set_config(self, yml_file_loc):
        """
        : Set configurations from a yml file
        """
        with open(yml_file_loc) as stream:
            try:
                self.yml_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return

    def pull_data(self, data_type, device_type=None, start_date=None, end_date=None,
                  output_dir=None, fields=None, timing=True, name=None, concatenate=False, token=False, url='url'):
        """
        : Wrapper to set query, output directory, and pull data.
        """
        assert self.yml_dict is not None
        self.splk_query = build_splunk_query(self, data_type=data_type, start_date=start_date, end_date=end_date,
                                             device_type=device_type, fields=fields, timing=timing)
        data_name = data_type
        if device_type is not None:
            data_name = data_type + '_' + device_type
        if name is not None:
            data_name = name

        self.splk_url = self.yml_dict[data_type][url]

        self._set_output_dir(output_dir)
        self.fetch_data(data_name, concatenate=concatenate, token=token)
        return

    def pull_dashboard(self, start_date, end_date, output_dir, concatenate=False):
        """
        : Returns dashboard labels from UI
        """
        self.splk_query = """| inputlookup alert_lumen_routing_and_interface_results.csv """

        self.start_date = datetime.strptime(start_date,'%m/%d/%Y:%H:%M:%S').strftime('%Y-%m-%d')
        self.end_date = datetime.strptime(end_date,'%m/%d/%Y:%H:%M:%S').strftime('%Y-%m-%d')
        self._set_output_dir(output_dir)
        timing_splk = '| search  EventStart > "{0}", EventStart < "{1}" '.format(self.start_date, self.end_date)

        self.splk_query = self.splk_query + timing_splk
        print(self.splk_query)

        self.splk_url = "https://ah-1170346-001.sdi.corp.bankofamerica.com:8089/services/search/jobs/export"
        self.fetch_data('flaps_dashboard', concatenate=concatenate)
        return

    def set_inputlookups(self, lookup_file):
        # TODO: Add function to update inputlookups
        return


_inst = SplunkDataModel()
set_config = _inst.set_config
set_credentials = _inst.set_credentials
pull_data = _inst.pull_data
pull_dashboard = _inst.pull_dashboard
