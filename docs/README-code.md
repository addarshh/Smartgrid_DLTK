


#<font size ="5"> Hyper-ticketing </font>

##<font size ="4"> class Config </font>
1. setup the params used in the notebook
2. all the params are stored in a yaml file

##<font size ="4"> class RawData </font>
1. read master data with both remedy and cmsp event, this is the output of preproc master
2. read remedy association label
3. calculate the association group by DFS
4. merge the association group label with cmsp events

##<font size ="4"> class Graph </font>
1. DFS to find cycles in the data 
2. use it to find association groups from association table

##<font size ="4"> class RawData </font>
1. read master data with both remedy and cmsp event, this is the output of preproc master
2. read remedy association label
3. calculate the association group by DFS
4. merge the association group label with cmsp events

##<font size ="4"> class EventSwarmEval </font>
1. create swarms from cmsp event based on spatial temporal filters
2. compare the swarms with association groups by overall jaccard index and rand index

##<font size ="4"> class Particle </font>
1. update position of each particle 
2. record best position for both individual and group

##<font size ="4"> class PSO </font>
1. particle swarm optimization main class
2. run PSO to minimize objective function, chosen from overall jaccard or rand index plus penalty

##<font size ="4"> class OptimizeFilters </font>
1. wrapper class to run PSO for training 
2. save the best filters and the course of PSO in each iteration in json
3. save the swarm labels using the best filtering conditions

##<font size ="4"> class OnlineFiltering </font>
1. applying the filtering condition to incoming cmsp event, and reassign labels by considering exiting swarm labels
2. read the json for filtering coniditions and the swarm label from previous filtering. if it is the first time after training, the existing swarm label will be from training data

##<font size ="4"> func run_PSO </font> 
1. main function to train PSO using training data

##<font size ="4"> func run_online_filtering </font>
1. main function to perform online filtering for new events



## <font size ="5"> Smart-priority </font>

## <font size ="4"> class SmartConfig </font>
1. setup the params used in the notebook
2. all the params are stored in a yaml file

##<font size ="4"> autoencoder functions </font>
1. func ae_red_dim to train an autoencoder on the standardized feature data
2. need to specify inputs - max_iteration, encoding_dim_1, encoding_dim_2(=n_components)
3. func encoder to transform standardized feature data to reduced dimensionality features using trained AE weights

## <font size ="4"> class CreateModelData </font>
1. create mixed sampling to balance actionable and non actionable class
2. up and down sampling is used (sklearn implementation)

## <font size ="4"> class ModelEvaluation </font>
1. evaluate model performance using 70/30 train test split
2. func calculate_classification_metrics: calculate classification report and AUC for top clusters, all clusters, and all swarms
3. func create_output_df: create output file with confusion (TP, FP, TN, FN)
4. func calculate_output_metrics: calculate number of events, cmsp tickets, and remedy incidents for actionable and non actionable tickets respectively
5. func get_feature_importance: calculate overall feature imporatance

## <font size ="4"> class ModelPipeline </font>
1. model pipeline: standardize --> dimensionality reduction --> clustering --> classification
2. func preprocess: standardize the feature data. MinMax from -1 to 1 is the default, standard scaler is an option
3. func dim_reduction: reduce dimension of the feature data. ae is the default, but with options of PCA, SVD, NMF, and other nonlinear methods.
4. func clustering: cluster events based on reduced features. Default is kmean, with hierarchical clustering as alternative method.
5. func cluster_plot: plot k mean clustering with the first two reduced features, overlaying with useful evnts 
6. func post_process_clusters: only select clusters with high concentration of useful events as the input of the classification model with original features (default) or reduced features (alternative). Balance actionable and non-actionable class using CreateModelData class.
7. func fit_classification_model:use rf (default) with gb and xgb (alternatives) to predict whether an event is actionable or not
8. func add_labels_back: utiltily function to make sure the columns remain the same after each process
9. func save_model_files: save model weights as pickle file and config as json

## <font size = "4"> class OnlinePrediction </font>
1. class to read in the model weights and make prediction of whether an event is actionable or nonactionable online
2. for each batch of prediction, save feature and model prediction in the same csv file

## <font size ="4"> class ReadFeatureFile </font>
1. class to read master_df from the latest training data folder if directory is not spedicifed
2. option to remove all forward-looking features in training

## <font size = "4"> func run_model </font>
main function to run model pipeline for training

## <font size = "4"> func run_online_prediction </font>
main function to run model pipeline with pretrained weights for online prediction


# <font size ="5"> Preprocessing </font>

##<font size ="4"> raw data include </font>
1. cmsp event
2. remedy ticket
3. remedy label
4. timezone lookup
4. device list

## <font size ="4"> raw data merging include </font>
1. parse cmsp ticket id from remedy
2. filter cmsp events on only relevant device or all
3. merge cmsp event to remedy based on cmsp ticket id

## <font size ="4"> feature engineering include </font>
1. converting to local time and get meaningful features
2. parsing em7_message to obtain interface features
3. creating lookback forward window and count statistics
4. catboost encoding for categorical variables
5. pad long time interval with no events with healthy flag


## <font size ="4.5"> Code Components </font>

## <font size ="4"> class PreprocConfig </font>
1. setup the params used in the notebook
2. all the params are stored in a yaml file
3. read_config() assigns all params as attributes to config object

## <font size ="4"> class RawData </font>
1. class to read and clean all raw data sources needed to create master data
2. all raw data files should be present in working_dir
3. leave_out_months is a legacy param, used when there were monthwise event data files, need not be used in production (put single_file=True to bypass using this param)
4. get_remedy() reads in Remedy incident data, filters on REPORTED_DATE after incidents_reported_after, and drops duplicate INCIDENT_NUMBER
5. get_remedy_label() reads in the label file which maps the 5 status and resolution columns in Remedy data to the USEFUL? label; blanks are filled with 'N' for NO ACTION TAKEN and 'Y' for all others
6. get_cmsp_event() reads in the cmsp em7 event data file(s) (if single_file is true, read file specified in yaml, else read monthwise files except leave_out_months and concatenate them); convert em7_first_occurrence from unix timestamp to datetime and filter using specified date_range, drop duplicate em7_event_id, also drop duplicate events with different em7_event_id but all other fields identical
7. get_device() reads in the device lookup file containing device information like model_number, manufacturer, location etc

##<font size ="4"> helper functions for parsing CMSP ticket ids </font>
1. parse_parent_tid() finds the 6/7-digit CMSP Parent Ticket ID from the raw field of Remedy incident data, returns nan if not found
2. parse_child_tid() finds all the CMSP Child Ticket IDs from the raw field of Remedy incident data
3. refine_child_tid() refines the child ticket ids found by ensuring they are legitimate CMSP ticket ids
4. merge_parent_child_cmsp() creates a dataframe with all INCIDENT_NUMBERs mapped with corresponding CMSP ticket ids (both parent/child)

## <font size ="4"> class MergeData </font>
1. class to merge all raw data sources
2. merge_device() joins event data and device data on the device name (inner join if FC_only as the device lookup file has only FC devices, left join otherwise); if device_filter={column:value} is provided, filter on the specified column and value (e.g., {MODEL_NUMBER: ION 3000}); also keep relevant columns after merging
3. assign_remedy_labels() assigns the USEFUL? labels to all Remedy incidents using the label mapping, while dropping incidents which do not find a match in the label mapping; also adds ACTIONABLE? (0 if RESOLUTION_CATEGORY='NO ACTION TAKEN', 1 otherwise) and IS_LOR (1 if CATEGORIZATION_TIER_3='NETWORK CONNECTIVITY \ LOSS OF REDUNDANCY', 0 otherwise) labels
4. parse_cmsp_ticket_id() calls the helper functions for parsing CMSP ticket ids and adds the columns ALERT_TID and ALERT_TYPE to Remedy data
4. merge_remedy_event() joins the cmsp event and device data with the Remedy data on the CMSP ticket id (left join because all events do not have a CMSP ticket or Remedy incident); also drop events for which Remedy incident is Pending, Assigned or In Progress 
5. merge_user_label(user_label_dir) overwrites the remedy label if a user label is provided. the user label should be from the dashboard input in user_label_dir

## <font size ="4"> helper functions for feature engineering </font>
1. forward_rolling_num_features_on_time() calculates aggregate statistics (mean, mode, variance, kurtosis, slope) for numeric features by grouping events from the same device occurring within a specified time window after the current event
2. backward_rolling_num_features_on_time() calculates aggregate statistics (mean, mode, variance, kurtosis, slope) for numeric features by grouping events from the same device occurring within a specified time window before the current event
3. pad_single_event() finds the last/next event from the same device within the specified backward/forward window and returns for that event the em7_event_policy_name_encoded value and the time difference with respect to the current event
4. padding_time_sequence() divides the backward/forward-looking time window into 10 minute slots and calculates the mean and sum of em7_event_policy_name_encoded values for events occurring in every 10 minute slot from the same device
5. interface_features_from_message() uses regex templates (re1, re2, re3) to parse out interface related information (downtime, number of flaps, number of messages) from the em7_message field

## <font size ="4"> class EventFeatures </font>
1. main class to derive feature data from master data
2. if feat_list=post_5_min, limit all forward looking time windows to 5 minutes
3. drop_crq_events() joins the master data with CMSP ticket data and drops the events which have certain keywords related to change request or decommissioned devices in the em7_partner_ticket or RESOLUTION_VARCHAR field
4. catboost_encoding() converts the categorical features into numeric types using Catboost encoding; during training, it uses labels from 5% of all events (dropped from feature data to avoid target leaking) to train the catboost encoder and save it in a pickle file, and during prediction, it loads the pickled encoder file to transform the categorical features
5. backward_rolling_num_features() calls the helper function backward_rolling_num_features_on_time() for the list of backward rolling windows provided
6. forward_rolling_num_features() calls the helper function forward_rolling_num_features_on_time() for the list of forward rolling windows provided
7. local_time_features() parses the state from the LOCATION field, uses the timezone lookup file to find the timezone for that device, and shifts the first occurrence of the event by the timezone difference to calculate local time features like day of week, hour of day, whether it is a business day (Mon-Fri) and business hour (8AM-4PM)
8. pad_single_event() calls the helper function pad_single_event() for the specified backward and forward looking time windows
9. padding_time_sequence() calls the helper function padding_time_sequence() for the specified backward and forward looking time windows
10. interface_features_from_message() calls the helper function interface_features_from_message()
11. make_features() joins all the derived features created from the previous functions based on the em7_event_id
12. add_labels() adds the IS_REMEDY, ACTIONABLE? and USEFUL? labels to the feature data, and fills the blanks with 0s

## <font size = "4"> class FlipLabel </font>
1. class to flip the USEFUL? labels for non-actionable events which are in the same association group or the same swarm group as an actionable event
2. turn_association_label() flips the USEFUL? label based on the association_label file
3. turn_swarm_label() flips the USEFUL? label based on the swarm_label file, but ensure that events which do not have an associated CMSP ticket remain non-actionable

## <font size = "4"> func create_master_data() </font>
1. main function to create the master data using methods from the RawData and MergeData classes
2. train=True => training, train=False => prediction
3. all remedy related functions (reading remedy data, assigning labels, parsing CMSP ticket ids, merging remedy data etc) are called only during training
4. for saving the master_df, create a new folder every time we train, while rewrite the existing master_df during prediction (since we run prediction every 5 minutes)

## <font size = "4"> func create_feature_data() </font>
1. main function to create the feature data using methods from the EventFeatures class
2. train=True => training, train=False => prediction
3. during training, identify the latest folder in the train folder if master_data_dir=latest
4. call FlipLabel methods and drop_crq_events() if corresponding boolean flags are True during training only; labels are also added during training only
5. save feature_df in the same folder as the corresponding master_df




# <font size ="5"> Post processing </font>

## <font size ="4"> class PostprocConfig </font>
1. setup the params used in the notebook
2. all the params are stored in a yaml file

## <font size = "4"> class MergeOutput </font>
1. func merge_data: merge existing csv file for dashboard with new prediction to make one csv for the dashboard (curr time - 2hr)
2. rewrite current 

##<font size = "4"> func run_postproc </font>
Main function to run post proc and try to save the output into the splunk index. If fails, will save the output to a csv file.