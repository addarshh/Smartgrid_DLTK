
from feature_generation.RunFeatureGeneration import create_feature_data
from BasePath import base_path
from utils.utils import get_latest_folder, SingleConfig


yaml_file = base_path + 'data/yaml/fc_train.yml'
singleConfig = SingleConfig(yml_file_loc=yaml_file,basepath=base_path)

preproc_config = singleConfig.preproc_config


# Feature Generation
master_df_folder = get_latest_folder("latest", base_path + "data/fc/train_output")
preproc_config.feature['filepaths']["train_pso_data_dir"] = master_df_folder + "/pso_filters_run"
preproc_config.feature['filepaths']["train_master_data_dir"] = master_df_folder
create_feature_data(config=preproc_config, train=True)

print("Finished.")
