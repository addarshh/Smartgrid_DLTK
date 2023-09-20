
from preprocessing.RunPreprocessing import create_master_data
from feature_generation.RunFeatureGeneration import create_feature_data
from hyperticketing.RunHyperTicketing import run_pso
from smartpriority.RunSmartPriority import train_smart_priority_model
from postprocessing.RunPostprocessing import run_postprocessing
from BasePath import base_path
from utils.utils import SingleConfig


yaml_file = base_path + 'data/yaml/fc_train.yml'
singleConfig = SingleConfig(yml_file_loc=yaml_file,basepath=base_path)

preproc_config = singleConfig.preproc_config
hyper_config = singleConfig.hyper_config
smart_config = singleConfig.smart_config
postproc_config = singleConfig.postproc_config

# Preprocessing
master_df_folder = create_master_data(config=preproc_config, train=True)

# Hyper Ticketing
hyper_config.train['filepaths']["master_df_folder_dir"] = master_df_folder
hyper_config.train['filepaths']["saved_model_path"] = master_df_folder
swarm_label_folder = run_pso(config=hyper_config)

# Feature Generation
preproc_config.feature['filepaths']["train_pso_data_dir"] = swarm_label_folder
preproc_config.feature['filepaths']["train_master_data_dir"] = master_df_folder
create_feature_data(config=preproc_config, train=True)

# Smart Priority Training
smart_config.train['filepaths']["master_df_dir"] = master_df_folder
smart_config.train['filepaths']["saved_swarm_folder_dir"] = swarm_label_folder
mp, model_eval = train_smart_priority_model(config=smart_config)

# Postprocessing
run_postprocessing(postproc_config, train=True, master_df_folder=master_df_folder)

print("Finished.")
