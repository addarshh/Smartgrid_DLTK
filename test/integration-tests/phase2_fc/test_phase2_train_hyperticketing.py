
from hyperticketing.RunHyperTicketing import run_pso
from BasePath import base_path
from utils.utils import get_latest_folder, SingleConfig


yaml_file = base_path + 'data/yaml/fc_train.yml'
singleConfig = SingleConfig(yml_file_loc=yaml_file,basepath=base_path)

hyper_config = singleConfig.hyper_config

# Hyper Ticketing
master_df_folder = get_latest_folder("latest", base_path + "data/fc/train_output")

hyper_config.train['filepaths']["master_df_folder_dir"] = master_df_folder
hyper_config.train['filepaths']["saved_model_path"] = master_df_folder
swarm_label_folder = run_pso(config=hyper_config)

print("Finished.")
