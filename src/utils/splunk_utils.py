

import os
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import httplib2
import urllib
from xml.dom import minidom
from io import StringIO
import json
from datetime import datetime


def fetch_data_user(splunkmodel, filename, concatenate=False):
    """
    : Wrapper for fetching data from splunk with an HTTP request
    """
    with requests.post(splunkmodel.splk_url,
                       data={'output_mode': 'csv', 'search': splunkmodel.splk_query},
                       stream=True,
                       verify=False,
                       auth=HTTPBasicAuth(splunkmodel.splk_u, splunkmodel.splk_p)) as r:
        r.raise_for_status()
        with open("filename.tmp", 'wb') as dst_file:
            for chunk in r.iter_content(chunk_size=8192):
                dst_file.write(chunk)
    output_file = splunkmodel.output_dir + "/{0}.csv".format(filename)

    if concatenate and os.path.exists(output_file):
        existing_file_is_empty = False
        new_file_is_empty = False
        try:
            df_old = pd.read_csv(output_file)
        except pd.errors.EmptyDataError:
            print('Existing file {} was empty'.format(output_file))
            existing_file_is_empty = True

        try:
            df_new = pd.read_csv('filename.tmp')
        except pd.errors.EmptyDataError:
            print('No data was present in Splunk for the query {}'.format(splunkmodel.splk_query))
            new_file_is_empty = True

        if existing_file_is_empty==False and new_file_is_empty==False:
            df = pd.concat([df_old ,df_new]).drop_duplicates().reset_index(drop=True)
            df.to_csv(output_file, index=False)
        elif existing_file_is_empty==True:
            os.replace("filename.tmp", output_file)

    else:
        os.replace("filename.tmp", output_file)
    return


def fetch_data_token(splunkmodel, filename, concatenate=False):
    """
    : Wrapper for fetching data from splunk with httplib2 and auth token.
    """
    splk_url = splunkmodel.splk_url
    httpHeaders = splunkmodel.headers

    searchQuery = splunkmodel.splk_query
    searchQuery = searchQuery.strip()
    if not (searchQuery.startswith('search') or searchQuery.startswith("|")):
        searchQuery = 'search ' + searchQuery

    _data = (httplib2.Http(disable_ssl_certificate_validation=True).request(
             splk_url + '?output_mode=csv', 'POST',
             headers=httpHeaders,
             body=urllib.parse.urlencode({'search': searchQuery}))[1])

    _reqData = str(_data, 'utf-8')
    data = StringIO(_reqData)
    splunk_df = pd.read_csv(data)
    splunk_df.to_csv('filename.tmp', header=True, index=False)

    output_file = splunkmodel.output_dir + "/{0}.csv".format(filename)
    if concatenate and os.path.exists(output_file):
        existing_file_is_empty = False
        new_file_is_empty = False
        try:
            df_old = pd.read_csv(output_file)
        except pd.errors.EmptyDataError:
            print('Existing file {} was empty'.format(output_file))
            existing_file_is_empty = True

        try:
            df_new = pd.read_csv('filename.tmp')
        except pd.errors.EmptyDataError:
            print('No data was present in Splunk for the query {}'.format(splunkmodel.splk_query))
            new_file_is_empty = True

        if existing_file_is_empty == False and new_file_is_empty == False:
            df = pd.concat([df_old, df_new]).drop_duplicates().reset_index(drop=True)
            df.to_csv(output_file, index=False)
        elif existing_file_is_empty == True:
            os.replace("filename.tmp", output_file)
    else:
        os.replace("filename.tmp", output_file)
    return


def build_splunk_query(splunkmodel, data_type, start_date=None, end_date=None,
                       device_type=None, fields=None, timing=True):
    """
    : Return generic query built from yml instructions
    """

    # Index
    index = splunkmodel.yml_dict[data_type]["index"]

    # Time span
    timing_splk = ""
    if timing:
        splunkmodel._set_start_end(start_date, end_date)
        timing_splk = ' earliest="{0}" latest="{1}" '.format(splunkmodel.start_date, splunkmodel.end_date)
        
    # Lookup
    lookup = ""
    if "lookup" in splunkmodel.yml_dict[data_type]:
        assert device_type is not None
        lookup = splunkmodel.yml_dict[data_type]["lookup"][device_type]

    # Fields
    if fields is None:
        fields = splunkmodel.yml_dict[data_type]["fields"]
    fields_splk = ' | table'
    for field in fields:
        fields_splk = fields_splk + " " + field

    splk_query = 'search ' + index + timing_splk + lookup + fields_splk
    print(splk_query)

    return splk_query


def stream_final_output(final_output, test_number):

    splunk_url = "https://ah-1064594-001.sdi.corp.bankofamerica.com:8088/services/collector/event"

    final_output['first_occurrence'] = final_output['first_occurrence'].astype(str)

    # fill all nas with string "nan"
    final_output = final_output.fillna("nan")

    # Set test=3 for UAT filter
    final_output["test"] = test_number

    # convert df to string of concatenated json objects
    event_dict = final_output.to_dict(orient='index')
    payload = ''
    for idx in event_dict:
        payload += json.dumps({"host": splunk_url, "event": event_dict[idx]})

    headers = {'Authorization': 'Splunk 58927da6-0c57-459e-b220-4b28adf38aee'}

    response = requests.post(splunk_url, headers=headers, data=payload, verify=False)
    return response


def save_master_df_to_splunk(master_df, last_date=None):
    """
    : Stores master_df on splunk index
    """
    splunk_url = "https://ah-1064594-001.sdi.corp.bankofamerica.com:8088/services/collector/event"

    # fill all nas with string "nan"
    master_df = master_df.fillna("nan")

    master_df['test'] = 1
    master_df["datatype"] = "smartgrid_master_df"
    master_df = master_df.sort_values('first_occurrence').reset_index(drop=True)

    if last_date is not None:
        master_df2 = master_df[master_df['timestamp'] > last_date.values[0,0]].copy()
    else:
        master_df2 = master_df

    if len(master_df2) > 0:
        for i in range(int(len(master_df2)/1000)+1):
            a = i*1000
            b = a + 1000
            tmp = master_df2.iloc[a:b]

            # convert df to string of concatenated json objects
            event_dict = tmp.to_dict(orient='index')

            # Appending to this string takes too long for too many rows, hence the for loop.
            payload = ''
            for idx in event_dict:
                payload += json.dumps({"host": splunk_url, "event": event_dict[idx]})

            headers = {'Authorization': 'Splunk 1f1ba441-d0be-4baa-986a-82a99245816c'}

            response = requests.post(splunk_url, headers=headers, data=payload, verify=False)
    return


def pull_latest_training_date_from_splunk(splk_u, splk_p, device_type):
    """
    : Returns the latest date from previously saved master_df
    """
    splk_query = 'search index="sg_model_train" test=1 device_type=' + device_type + ' | stats max(timestamp) as timestamp'
    print(splk_query)
    splk_url = "https://ah-1064594-001.sdi.corp.bankofamerica.com:8089/services/search/jobs/export"
    with requests.post(splk_url,
                       data={'output_mode': 'csv', 'search': splk_query},
                       stream=True,
                       verify=False,
                       auth=HTTPBasicAuth(splk_u, splk_p)) as r:
        print(r)
        _reqData = str(r.content, 'utf-8')
        data = StringIO(_reqData)
        df = pd.read_csv(data)
        a = df[['timestamp']]
    return df[['timestamp']]

def pull_latest_prediction_date_from_splunk(splk_u, splk_p, device_type):
    """
    : Returns the latest date from previously saved master_df
    """
    splk_query = 'search index="baseline_ph2_test" test=3 device_type=' + device_type + ' | stats max(timestamp) as timestamp'
    print(splk_query)
    splk_url = "https://ah-1064594-001.sdi.corp.bankofamerica.com:8089/services/search/jobs/export"
    with requests.post(splk_url,
                       data={'output_mode': 'csv', 'search': splk_query},
                       stream=True,
                       verify=False,
                       auth=HTTPBasicAuth(splk_u, splk_p)) as r:
        print(r)
        _reqData = str(r.content, 'utf-8')
        data = StringIO(_reqData)
        df = pd.read_csv(data)
        a = df[['timestamp']]
    return df[['timestamp']]


def save_model_eval_to_splunk(model_eval_dict):
    """
    : Stores master_df on splunk index
    """
    splunk_url = "https://ah-1064594-001.sdi.corp.bankofamerica.com:8088/services/collector/event"

    assert 'device_type' in model_eval_dict
    assert 'recall' in model_eval_dict
    assert 'precision' in model_eval_dict

    model_eval_dict['timestamp']= datetime.now().strftime("%Y-%m-d %H:%M:%S")

    # Appending to this string takes too long for too many rows, hence the for loop.
    payload = ''
    for idx in model_eval_dict:
        payload += json.dumps({"host": splunk_url, "event": event_dict[idx]})

    headers = {'Authorization': 'Splunk 1f1ba441-d0be-4baa-986a-82a99245816c'}

    response = requests.post(splunk_url, headers=headers, data=payload, verify=False)
    return