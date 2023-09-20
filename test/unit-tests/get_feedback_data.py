import sg_splunk
from BasePath import base_path
from datetime import datetime, timedelta

yml = base_path + "data/yaml/input_data_config.yml"
sg_splunk.set_config(yml)

sg_splunk.set_credentials("ah-1064594-001.sdi.corp")
end_date = None

days=7

if end_date is None:
    now = datetime.utcnow()
else:
    now = datetime.strptime(end_date, "%Y-%m-%d")
before = now - timedelta(minutes=days * 24 * 60)
end_date = now.strftime("%m/%d/%Y:%H:%M:%S")
start_date = before.strftime("%m/%d/%Y:%H:%M:%S")

out_dir_fc = "data/fc"
out_dir_fc = base_path + out_dir_fc

sg_splunk.pull_data(data_type="feedback_data_ion3k",
                    output_dir=out_dir_fc,
                      end_date=end_date,
                      start_date=start_date)


out_dir_gc = "data/gc/"
out_dir_gc = base_path + out_dir_gc

sg_splunk.pull_data(data_type="feedback_data_global_core",
                    output_dir=out_dir_gc,
                      end_date=end_date,
                      start_date=start_date)