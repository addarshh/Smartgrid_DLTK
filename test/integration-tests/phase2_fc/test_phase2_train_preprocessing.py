
from preprocessing.RunPreprocessing import create_master_data
from BasePath import base_path
from utils.utils import SingleConfig

yaml_file = base_path + 'data/yaml/fc_train.yml'
singleConfig = SingleConfig(yml_file_loc=yaml_file,basepath=base_path)

preproc_config = singleConfig.preproc_config

# Preprocessing
master_df_folder = create_master_data(config=preproc_config, train=True)

print("Finished.")
