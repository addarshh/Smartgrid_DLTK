import pandas as pd
import datetime
from BasePath import base_path
import numpy as np
import timeit
start_time = timeit.default_timer()

global_core_syslog = pd.read_csv(base_path + 'data/train_input/phase2/syslog_data_global_core.csv')

print(global_core_syslog['severity'].value_counts())
print('read time', timeit.default_timer() - start_time)
print(global_core_syslog.dtypes,
      global_core_syslog.drop_duplicates().shape)

global_core_syslog['host'] = global_core_syslog['host'].apply(lambda x:x.split('.')[0].upper())

# print(global_core_syslog['host'].nunique())

dashboard_log = pd.read_csv(base_path + 'data/train_input/phase2/dashboard.csv')

# print(dashboard_log['host'].nunique())

def add_interface(row, dashboard_log, time_window):
    host_dashboard_log = dashboard_log[dashboard_log['host']==row['host']]
    if host_dashboard_log.empty :
        interface_flaps = 0
    else:
        row_time = pd.to_datetime(row['_time']).tz_convert('UTC')
        host_dashboard_log['time'] = pd.to_datetime(host_dashboard_log['_time']).dt.tz_convert('UTC')
        backward_window = row_time - datetime.timedelta(minutes=time_window)
        forward_window = row_time + datetime.timedelta(minutes=time_window)
        # print(backward_window,forward_window)
        host_dashboard_log = host_dashboard_log[(host_dashboard_log['time']>backward_window)
                                                & (host_dashboard_log['time']<forward_window)]
        interface_flaps = host_dashboard_log['flap_count'].sum()
        # interface_flaps = 1

    return interface_flaps

# row1 = global_core_syslog[:1]
# host_dashboard_log = dashboard_log[dashboard_log['host'] == row1['host'].values[0]]
time_window = 30


global_core_syslog['_time'] = pd.to_datetime(global_core_syslog['_time']).dt.tz_localize(None)
# global_core_syslog['timestamp'] = global_core_syslog['_time'].apply(lambda x:datetime.datetime.timestamp(x))
print('localize_time', timeit.default_timer() - start_time)
global_core_syslog['timestamp'] = global_core_syslog['_time'].values.astype(np.int64) // 10 ** 9
print('converted to int', timeit.default_timer() - start_time)
global_core_syslog['timestamp'] = pd.to_datetime(global_core_syslog['timestamp'],unit='s')

# global_core_syslog['timestamp'] = pd.to_datetime(global_core_syslog['_time'],unit='s')

mid_convert_time = timeit.default_timer()

print(global_core_syslog[['timestamp','_time']].head())
print(mid_convert_time- start_time)

dashboard_log['_time'] = pd.to_datetime(dashboard_log['_time']).dt.tz_localize(None)
# dashboard_log['timestamp'] = dashboard_log['_time'].apply(lambda x:datetime.datetime.timestamp(x))
dashboard_log['timestamp'] = dashboard_log['_time'].values.astype(np.int64) // 10 ** 9
dashboard_log['timestamp'] = pd.to_datetime(dashboard_log['timestamp'],unit='s')

converted_time = timeit.default_timer()
print(converted_time - start_time)

global_core_syslog.sort_values(by=['timestamp','host'],inplace=True)
dashboard_log.sort_values(by=['timestamp','host'],inplace=True)
# print(dashboard_log.shape)
merged = pd.merge_asof(dashboard_log[['timestamp','host','flap_count']],
                       global_core_syslog,
                       by='host',on='timestamp',tolerance=pd.Timedelta('60min'))
# print(global_core_syslog.drop_duplicates(subset=['host','_time','messageName']).shape)
print(merged.shape)
print(global_core_syslog[['host','_time']].drop_duplicates().shape)
cols = list(global_core_syslog.columns)

global_core_syslog = pd.merge(global_core_syslog,
                              merged,
                              on=cols,
                              how='left')

# global_core_syslog = global_core_syslog.join(merged[['_time','flap_count','host','messageName']],
#                                              on=['host','_time','messageName'])
# merged['flap_count'] = merged['flap_count'].fillna(0)
cols = list(global_core_syslog.columns)
cols = cols.remove('flap_count')
global_core_syslog = global_core_syslog.drop_duplicates(subset=cols)
global_core_syslog['timestamp'] = global_core_syslog['timestamp'].apply(lambda x:pd.Timestamp(x))
end_time = timeit.default_timer()



print(end_time - start_time)

print(merged.shape, global_core_syslog.shape)

global_core_syslog['timestamp']= global_core_syslog['timestamp'].astype(str)
print(global_core_syslog.head())
# print(merged.groupby(['_time','host'])['flap_count'].sum().reset_index())
# global_core_syslog['interface_flaps'] = global_core_syslog.apply(lambda row: add_interface(row,dashboard_log,time_window),axis=1)


global_core_hosts = global_core_syslog['host'].drop_duplicates()
# print(dashboard_log[dashboard_log['host'].isin(global_core_hosts.values)]['host'].nunique())
dashboard_hosts = dashboard_log['host'].drop_duplicates()
common_hosts = global_core_hosts[global_core_hosts.isin(dashboard_hosts)]

# global_core_syslog = global_core_syslog[global_core_syslog['host'].isin(common_hosts)].reset_index(drop=True)


# global_core_syslog = global_core_syslog[global_core_syslog['host'].isin(hosts)]
# dashboard_log = dashboard_log[dashboard_log['host'].isin(hosts)]
#
# print(global_core_syslog.columns)
# print(dashboard_log.columns)
# print(global_core_syslog.shape, dashboard_log.shape)
# print(global_core_syslog['interface_flaps'].value_counts())
# global_core_syslog['_time'] = pd.to_datetime(global_core_syslog['_time']).dt.tz_convert(tz='UTC')
# dashboard_log['_time'] = pd.to_datetime(dashboard_log['_time']).dt.tz_convert(tz='UTC')
# global_core_time =  global_core_syslog['_time'].drop_duplicates()
# dashboard_time = dashboard_log['_time'].drop_duplicates()

# print(global_core_time[global_core_time.isin(dashboard_time)])
# print(global_core_time)
# print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
# print(dashboard_time)
