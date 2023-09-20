YAML parameters

1. preproc_config.yml
In this file, parameters are split into two dictionaries: master_df (for creation of master data) and feature_df (for creation of feature data).

feature_df:

  (1) association_file: association_label.csv 
  name of csv file with association group labels for each incident, used in FlipLabel.turn_association_label() to flip labels for non-actionable events in the same association group as actionable events
  
  (2) backward_rolling_windows:   
  - 300
  - 4500
  - 86400
  list of backward-looking time windows in seconds, used in EventFeatures.backward_rolling_num_features() to calculate rolling aggregation of features for events from the same device for every window in the list

  (3) backward_sequence_window: 4500 
  backward-looking time window in seconds, used in EventFeatures.padding_time_sequence() to calculate 10 minute aggregations for events from the same device within the window
  
  (4) backward_single_event_window: 4500
  backward-looking time window in seconds, used in EventFeatures.pad_single_event() as a max limit for the time elapsed since the last event from the same device
  
  (5) cat_features: 
  - em7_source_desc
  - em7_event_policy_name
  list of categorical features, used in EventFeatures.catboost_encoding() as list of features to convert to numeric type using catboost encoding
  
  (6) catboost_encoder_file: catboost_encoder.pkl
  name of pickle file where catboost encoder can be saved to (during training) or loaded from (during prediction), used in EventFeatures.catboost_encoding()
  
  (7) cmsp_ticket_file: CMSP_Event_Data_6_months.csv
  name of csv file with CMSP ticket data, used in EventFeatures.drop_crq_events() to drop events with keywords related to change request or decomissioned devices in em7_partner_ticket field
  
  (8) data_dir: /srv/notebooks/aiops_accelerator/data/
  main data directory where raw data, master data, and feature data are stored
  
  (9) drop_crq: true
  boolean to control execution of EventFeatures.drop_crq_events()
  
  (10) feat_list: post_5_min
  string - if 'post_5_min', limit forward looking windows to 300 seconds, else use windows provided in config yaml
  
  (11) feature_df_file: feature_df.csv
  name of csv file where feature data will be stored at the end of create_feature_data()
  
  (12) flip_association_label: true
  boolean to control execution of FlipLabel.turn_association_label()
  
  (13) flip_swarm_label: true
  boolean to control execution of FlipLabel.turn_swarm_label()
  
  (14) forward_rolling_windows:
  - 300
  - 1800
  - 4500
  list of forward-looking time windows in seconds, used in EventFeatures.forward_rolling_num_features() to calculate rolling aggregation of features for events from the same device for every window in the list
  
  (15) forward_sequence_window: 4500
  forward-looking time window in seconds, used in EventFeatures.padding_time_sequence() to calculate 10 minute aggregations for events from the same device within the window
  
  (16) forward_single_event_window: 4500
  forward-looking time window in seconds, used in EventFeatures.pad_single_event() as a max limit for the time elapsed until the next event from the same device
  
  (17) master_df_file: master_df.csv
  name of csv file where master data will be stored at the end of create_master_data()
  
  (18) num_feature:
  - em7_severity
  list of numerical features, used in EventFeatures
  
  (19) pso_dir: /srv/notebooks/aiops_accelerator/pso_filters/
  directory where output of run_PSO() is stored, used in FlipLabel
  
  (20) raw_data_dir: /srv/notebooks/aiops_accelerator/data/raw_data/
  directory where all raw data files are stored
  
  (21) save_catboost_encoder: true
  boolean to control saving and rewriting of catboost encoder pickle file during training, used in EventFeatures.catboost_encoding()
  
  (22) save_feature_df: true
  boolean to control saving the feature data
  
  (23) swarm_file: swarm_label.csv
  name of csv file with group labels obtained as a result, used in FlipLabel.turn_swarm_label()
  
  (24) timezone_lookup_file: us-zip-code-latitude-and-longitude.csv
  name of csv file with timezone and state information, used in EventFeatures.local_time_features()
  
  (25) train_master_data_dir: latest
  directory where master_df_file is stored; if 'latest', find the latest folder in the 'train' folder
  
  (26) train_pso_data_dir: latest
  directory where pso training outputs (association_file and swarm_file) are stored; if 'latest', find the latest folder in the 'pso_filters' folder, used in FlipLabel
  
  (27) ts_target: USEFUL?
  label to be used to train the catboost encoder, used in EventFeatures.catboost_encoding()
  
master_df:

  (1) FC_only: true
  boolean to control whether to inner join (true) or left join (false) the event data and device data as device lookup has only FC devices for now, used in MergeData.merge_device()
  
  (2) date_range:
  - '2020-06-01'
  - '2021-04-01'
  list of earliest and latest dates to filter the event em7_first_occurrence, used in RawData.get_cmsp_event()
  
  (3) device_filter:
    MODEL_NUMBER: ION 3000
  dictionary with format {column:value} to filter on the specified column and value for a particular type of device, used in MergeData.merge_device()
  
  (4) device_lookup_file: fc_devicesLocation.csv
  name of csv file where device information is stored
  
  (5) incidents_reported_after: '2020-10-01'
  earliest date to filter the Remedy incident REPORTED_DATE, used in RawData.get_remedy()
  
  (6) master_df_file: master_df.csv
  name of csv file where master data would be stored
  
  (7) predict_cmsp_event_file: TwoDayEventData_Oct31-Nov1_2020.csv
  name of csv file to be used as the event data during prediction
  
  (8) raw_data_dir: /srv/notebooks/aiops_accelerator/data/raw_data/
  directory where all raw data files are stored
  
  (9) remedy_inc_file: all_inc_2020-21.csv
  name of csv file where Remedy incident data is stored
  
  (10) remedy_labels_file: direct_use_labels.csv
  name of csv file with the mapping between the 5 status and resolution Remedy fields and the USEFUL? label
  
  (11) save_data_dir: /srv/notebooks/aiops_accelerator/data
  data directory which contains the train, prediction and raw_data folders
  
  (12) save_master_df: true
  boolean to control whether to save the master data (rewrite during prediction)
  
  (13) single_file: false
  boolean to control whether to use a single file for event data, or read in a list of monthwise event files and concatenate them (need to specify leave_out_months separately in this case)
  
  (14) train_cmsp_event_file: eventData_October2020.csv
  name of csv file to be used as the event data during training

  (15) user_label: false
  whether or not using user input in place of remedy label
  
  (16) user_label_dir: /srv/notebooks/aiops_accelerator/data/userlabel/userlabel.csv
  directory of the user label file from the dashboard


2. hyper_config.yml
In this file, parameters are splitted in two dictionaries: train_hyper_ticket (for training) and online_hyper_ticket (for online filtering).

online_hyper_ticket:

  (1) existing_online_swarm: false
  Whether or not the new predicted swarm should be combined with an existing swarm. if existing_on_swarm = false,then the new events swarm will be combined with the latest training swarm.
  
  (2) master_df_name: /srv/notebooks/aiops_accelerator/data/prediction/master_df.csv
  Directory to the master file for online filtering, currently will always have the same directory and file name
  
  (3) raw_data_dir: /srv/notebooks/aiops_accelerator/data/raw_data
  Directory to read in the raw data used in hyper ticketing, including remedy association table
  
  (4) save_swarm_dir: /srv/notebooks/aiops_accelerator/pso_online
  Directory to the folder to save the online filtering result
  
  (5) saved_model_dir: /srv/notebooks/aiops_accelerator/pso_filters
  Directory to the folders with saved pretrained filters
  
  (6) saved_model_folder_dir: latest
  Whether read in the filter in a specified folder under saved_model_dir, or read in the latest training result. if saved_model_folder_dir = latest, then read in the latest.
  
train_hyper_ticket:

  (1) cost_function: Overall Jaccard
  Cost function for PSO. Overall Jaccard: the weighted harmonic mean of swarm to association Jaccard index and association to swarm Jaccard index. The weight is determined by eval_beta. If eval_beta = 1, then it's the balanced harmonic mean. if not Overall Jaccard, any other value will use rand score for PSO. 
  
  (2) data_dir: /srv/notebooks/aiops_accelerator/data
  Overall data directory, this folder should include raw_data folder and train folder
  
  (3) device_type: all
  Devices used in PSO. if not all, will filter on the device type from master_df. If master_df is filteres already by device_type, this value should be all in hyper ticketing training
  
  (4) eval_beta: 1
  Weight assigned to calcualte overall jaccard cost function. Wen eval_beta = 1, we assign equal weight to swarm to association Jaccard index and association to swarm Jaccard index. With bigger eval_bets, more weight will be assigned to swarm to association Jaccard index.
  
  (5) init_pos:
  - 5
  - 5
  PSO initialization position for time (first number) and space (second number). We scaled the potentail time window and cateogrical spatial filter to -10 to 10.  
  
  (6) master_df_folder_dir: latest
  Directory to the folder containing master_df for PSO filter training. if master_df_folder_dir = latest, will read in the master_df in the most recent folder for training.
  
  (7) master_df_name: master_df.csv
  File name for master_df. 
  
  (8) max_iteration: 100
  Number of iterations for PSO. Recommend in the first PSO trial run, provide a large number ~100 for the model to converge. Then the max_iteration can be reduced based on the convergence curve
  
  (9) min_max: 10
  The scale of which the time window and codified spatial filters. The pso_params should change according to the min_max scale
  
  (10) num_par: 15
  Number of particles in PSO
  
  (11) penalty: 0.02
  Penalty for the maxmimum swarm size in terms of the time delta between the first event and last event in the same swarm. The penalty increases linearly with 0.02 rate per week, e.g. the penalty for a 1-week long swarm is .02, and .04 for a 2-week long swarm
  
  (12) pso_bound:
    1st dim:
    - -10
    - 10
    2nd dim:
    - -10
    - 10
    Upper and lower bound for the search space. The search space should coordinate with min_max. 
    
  (13) pso_params:
  - 1
  - 1
  - 1.5
  Inertia = 1, social constant = 1 , cognifitive constant = 1.5 are the hyperparameters for the pso algorithm, and in this order. The magnitude of the hyperparameters should be in line with the (13) pso bound and (9) min_max scale.
  
  (14) range_of_time:
  - 1800
  - 604800
  - 1800
  Search space for the temporal filter for PSO in seconds. -1800: lower bound, -604800: upper bound, -1800: step size.
  
  (15) raw_data_dir: /srv/notebooks/aiops_accelerator/data/raw_data
  Directory for the folder that hosts raw data such as remedy_association file
  
  (16) saved_model_path: /srv/notebooks/aiops_accelerator/pso_filters
  Directory to save the training weights

  (17) associations_file: remedy_associations_data.csv
  Name of csv file with Remedy associations data


3. smart_config.yml
In this file, parameters are splitted in two dictionaries: train_smart_priority (for training) and online_smart_priority (for online prediction).

online_smart_priority:

  (1) feature_file_dir: /srv/notebooks/aiops_accelerator/data/prediction/feature_df.csv
  full path for where the feature data for prediction is saved
  
  (2) save_output_dir: /srv/notebooks/aiops_accelerator/classification_online
  directory where prediction outputs will be saved
  
  (3) saved_model_dir: /srv/notebooks/aiops_accelerator/classification_output
  directory where all trained model folders are saved
  
  (4) saved_model_folder_dir: latest
  folder path with training outputs (config and model pickle files); if 'latest', find the latest folder in saved_model_dir
  
  (5) saved_swarm_dir: /srv/notebooks/aiops_accelerator/pso_online
  directory where all online filtering folders are saved
  
  (6) saved_swarm_folder_dir: latest
  folder path with the swarm labels for the prediction data; if 'latest', find the latest folder in saved_swarm_dir
  
train_smart_priority:

  (1) adjust_threshold: true
  boolean to control whether to use the OptimizeThreshold class to adjust the classification threshold
  
  (2) classification_model_param_dict:
    learning_rate: 0.1
    max_depth: 8
    n_estimators: 100
  dictionary of parameters used in the classification model, used in ModelPipeline.fit_classification_model()
  
  (3) classification_model_type: rf
  type of model to be used for classification, used in ModelPipeline.fit_classification_model() (options: rf=Random Forest, gb=Gradient Boosting, xg_boost=XGBoost)
  
  (4) classification_threshold: 0.5
  threshold for calculating 0/1 prediction labels using the model predicted probabilities (will be overriden if adjust_threshold=true), used in ModelEval
  
  (5) cluster_method: kmeans
  type of clustering method, used in ModelPipeline.clustering() (options: hierarchical=AgglomerativeClustering, any other value defaults to KMeans clustering)
  
  (6) data_dir: /srv/notebooks/aiops_accelerator/data
  main data directory which contains the train, prediction and raw_data folders
  
  (7) enc_dim_1_ae: 128
  number of neurons in the first hidden layer of the autoencoder, used in ModelPipeline.dim_reduction()
  
  (8) k_fold: 5
  number of splits in K-Fold cross validation, used in OptimizeThreshold
  
  (9) master_df_folder_dir: latest
  directory to the folder containing master_df for training; if 'latest', read in the master_df from the most recent training folder
  
  (10) max_iter_ae: 100
  maximum number of iterations for training the autoencoder, used in ModelPipeline.dim_reduction()
  
  (11) max_iter_nmf: 500
  maximum number of iterations for Non-negative Matrix Factorization, used in ModelPipeline.dim_reduction() (only when red_dim_method=NMF)
  
  (12) min_recall: 0.8
  minimum limit for recall while finding the best threshold, used in OptimizeThreshold
  
  (13) n_cluster: 12
  number of clusters that the events will be divided into, used in ModelPipeline.clustering()
  
  (14) n_comp: 32
  number of components (features) after dimensionality reduction, used in ModelPipeline.dim_reduction()
  
  (15) n_top_cluster: 4
  number of top clusters (clusters with largest number of USEFUL?=1 events) to be used for classification, used in ModelPipeline.post_process_clusters()
  
  (16) post_feature: true
  boolean to control whether to use forward-looking features, used in ReadFeatureFile.feature_selection()
  
  (17) predict_var: USEFUL?
  label to be used for prediction/classification
  
  (18) preprocess_method: minmax
  type of preprocessing method, used in ModelPipeline.preprocess() (options: minmax=MinMaxScaler, any other value defaults to StandardScaler)
  
  (19) preprocess_nmax: 1
  upper limit of range of feature values after preprocessing, used in ModelPipeline.preprocess()
  
  (20) preprocess_nmin: -1
  lower limit of range of feature values after preprocessing, used in ModelPipeline.preprocess()
  
  (21) red_dim_method: AE
  type of dimensionality reduction method, used in ModelPipeline.dim_reduction() (options: AE=autoencoder, PCA=Principal Component Analysis, NMF=Non-negative Matrix Factorization, any other value defaults to TruncatedSVD)
  
  (22) reduced_feature: false
  boolean to control which feature set to use for classification (if true, use components created after dimensionality reduction, use preprocessed original features otherwise), used in ModelPipeline.post_process_clusters()
  
  (23) save_output: true
  boolean to control whether to save feature data with prediction labels
  
  (24) save_output_dir: /srv/notebooks/aiops_accelerator/classification_output
  directory where all training model output folders are saved
  
  (25) save_weights: true
  boolean to control whether to save trained model pickle files and config json file
  
  (26) saved_swarm_dir: /srv/notebooks/aiops_accelerator/pso_filters
  directory to the folders with saved pretrained filters (swarm labels)
  
  (27) saved_swarm_folder_dir: latest
  folder path where swarm label file is saved; if 'latest', find the latest folder in saved_swarm_dir
  
  (28) threshold_grid: 50
  number of divisions within (0.5,1) for finding the best threshold, used in OptimizeThreshold
  
  (29) top_cluster_for_cv: true
  boolean to control whether to use only the top clusters for cross-validation, used in OptimizeThreshold
  
  
  

4. postproc_config.yml

post_proc:

  (1) classification_model_output_name: model_output.csv
  name of csv file with the training output with predicted labels
  
  (2) classification_pipeline_output_folder_dir: /srv/notebooks/aiops_accelerator/classification_output/
  directory where all trained model folders are saved
  
  (3) dashboard_folder_dir: /srv/notebooks/aiops_accelerator/dashboard/
  directory where the dashboard file is saved
  
  (4) dashboard_output_name: dashboard_output.csv
  name of csv file which feeds the dashboard
  
  (5) from_training: false
  boolean to control whether to use training output (true) or prediction output
  
  (6) master_col:
  - em7_event_id
  - em7_first_occurrence
  - LOCATION
  - em7_aligned_resource_name
  - em7_device_ip
  - em7_severity
  - em7_source_desc
  - em7_event_policy_name
  - em7_message
  list of columns to be included from the master_df
  
  (7) master_file_name: master_df.csv
  name of csv file containing the master data
  
  (8) online_classification_folder_dir: /srv/notebooks/aiops_accelerator/classification_online/
  directory where all prediction output folders are saved
  
  (9) online_classification_output_name: online_prediction.csv
  name of csv file with the prediction output
  
  (10) output_col:
  - em7_event_id
  - Useful_pred
  - Useful_prob
  - group_label
  - Group_Useful_pred
  list of columns to be included from the prediction output
  
  (11) prediction_folder_dir: /srv/notebooks/aiops_accelerator/data/prediction/
  directory where prediction master data is saved
  
  (12) rewrite: true
  boolean to control whether to rewrite the dashboard output file
  
  (13)streaming_to_index: false
  boolean to stream data to splunk index directly or save as csv
  
  (14) trainig_data_dir: /srv/notebooks/aiops_accelerator/data/train/
  directory where all folders with training master data are saved
  
  (15) training_folder_dir: latest
  folder path where master data is saved; if 'latest', find the most recent folder
