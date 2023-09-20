

from smartpriority.RunSmartPriority import train_smart_priority_model
from postprocessing.RunPostprocessing import run_postprocessing
from BasePath import base_path
from utils.utils import get_latest_folder, SingleConfig


yaml_file = base_path + 'data/yaml/fc_train.yml'
singleConfig = SingleConfig(yml_file_loc=yaml_file, basepath=base_path)

smart_config = singleConfig.smart_config
postproc_config = singleConfig.postproc_config

# create master data from data sources
master_df_folder = get_latest_folder("latest", base_path + "data/fc/train_output")
swarm_label_folder = master_df_folder + "/pso_filters_run"

# Run Smart Priority Training Step
smart_config.train['filepaths']["master_df_dir"] = master_df_folder
smart_config.train['filepaths']["saved_swarm_folder_dir"] = swarm_label_folder
mp, model_eval = train_smart_priority_model(config=smart_config)

# Postprocessing
# run_postprocessing(postproc_config, train=True, master_df_folder=master_df_folder)


print("Finished.")
