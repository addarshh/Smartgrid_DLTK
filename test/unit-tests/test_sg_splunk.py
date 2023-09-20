
# Unit tests for sg_splunk module

import sg_splunk
from datetime import datetime, timedelta
from BasePath import base_path
import os

class TestSGSplunk:

    def test_sg_splunk_1(self):
        """
        : Test that the connection to splunk works and all data can be pulled.
        """
        out_dir = base_path + "data/samples/unit_tests/test1"
        os.makedirs(out_dir, exist_ok=True)

        yml = base_path + "data/yaml/input_data_config.yml"
        sg_splunk.set_config(yml)

        days = 1
        now = datetime.utcnow()
        before = datetime.utcnow() - timedelta(minutes=days * 24 * 60)
        end_date = now.strftime("%m/%d/%Y:%H:%M:%S")
        start_date = before.strftime("%m/%d/%Y:%H:%M:%S")

        sg_splunk.set_credentials("ecslsplunk-ti")

        # MasterDF
        sg_splunk.set_credentials("ah-1064594-001.sdi.corp")
        sg_splunk.pull_data(data_type="master_df",
                            output_dir=out_dir,
                            timing=False,
                            device_type='ion3k')

        # CMBD Info
        sg_splunk.pull_data(data_type="cmdb_info", output_dir=out_dir, device_type="ion3k", timing=False)
        sg_splunk.pull_data(data_type="cmdb_info", output_dir=out_dir, device_type="global_core", timing=False)

        # Remedy Incident Data
        sg_splunk.pull_data(data_type="remedy_incident_data", output_dir=out_dir, device_type="ion3k")
        sg_splunk.pull_data(data_type="remedy_incident_data", output_dir=out_dir, device_type="global_core")

        # Syslog data
        sg_splunk.pull_data(data_type="syslog_data_ion3k", device_type="ion3k",
                            output_dir=out_dir, name="syslog_data_ion3k")
        sg_splunk.pull_data(data_type="syslog_data_global_core", device_type="global_core",
                            output_dir=out_dir, name="syslog_data_global_core")

        # Remedy Association Data
        sg_splunk.set_credentials("ah-1064594-001.sdi.corp")
        sg_splunk.pull_data(data_type="remedy_associations",
                            output_dir=out_dir,
                            end_date=end_date,
                            start_date=start_date)
        return

    def test_sg_splunk_2(self):
        """
        : Test that the connection to splunk works and all data can be pulled (new authentication)
        """
        out_dir = base_path + "data/samples/unit_tests/test3"
        os.makedirs(out_dir, exist_ok=True)

        yml = base_path + "data/yaml/input_data_config.yml"
        sg_splunk.set_config(yml)

        sg_splunk.set_credentials("reli0")

        # CMBD Info
        sg_splunk.pull_data(data_type="cmdb_info", output_dir=out_dir, device_type="ion3k", timing=False, url='url2', token=True)
        sg_splunk.pull_data(data_type="cmdb_info", output_dir=out_dir, device_type="global_core", timing=False, url='url2', token=True)

        # Remedy Incident Data
        sg_splunk.pull_data(data_type="remedy_incident_data", output_dir=out_dir, device_type="ion3k", url='url2', token=True)
        sg_splunk.pull_data(data_type="remedy_incident_data", output_dir=out_dir, device_type="global_core", url='url2', token=True)

        # Syslog data
        sg_splunk.pull_data(data_type="syslog_data_ion3k", device_type="ion3k",
                            output_dir=out_dir, name="syslog_data_ion3k", url='url2', token=True)
        sg_splunk.pull_data(data_type="syslog_data_global_core", device_type="global_core",
                            output_dir=out_dir, name="syslog_data_global_core", url='url2', token=True)

        return

if __name__ == "__main__":
    t = TestSGSplunk()
    t.test_sg_splunk_1()
    t.test_sg_splunk_2()