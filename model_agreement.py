# %%
# !pip install codecarbon

# %%
# Remove in py file
# ! codecarbon init

# %% [markdown]
# Compile all the results in a dataframe and save it as csv per experiment.
# 
# name the file appropriately.
# 
# Write another script to combine these results model-wise- for each model there are 36 combinations of (n_model, ar)

# %%
# import sklearn
import pandas as pd
import numpy as np
# from numpy import errstate,isneginf
import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.stats as sc

from sklearn.preprocessing import LabelEncoder
# , OneHotEncoder
from sklearn.model_selection import train_test_split
# , cross_val_score
from sklearn.model_selection import StratifiedKFold
# , StratifiedShuffleSplit

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# from sklearn.model_selection import BayesSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
from xgboost import XGBClassifier

from timeit import default_timer as timer

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from tqdm import tqdm
# from tqdm import trange
from collections import Counter

from sklearn.decomposition import PCA

import time

#import umap

# import zipfile
import os
import sys
import argparse
import logging
# import json
# import requests
# import io
# import copy
#import biomart
#import umap


# %% [markdown]
# # set the parameters for notebook or Read arguments from the command line in .py file 

# %%
# dataset = "Heart-Disease"
# dataset ="Gene-Expr"
debug = 0

dataset ="HD"

save_fig_path = "./"+dataset+"/"

model_name = "DT"
random_state = 42
n_models = 3
frac_agree = 1.0

test_split_size = 0.2

opt_model= 1

final_results = pd.DataFrame(columns=["s_no", "description","value"])

pd.set_option('display.max_colwidth', None)

# %%
def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser(
                                    prog="ModelAgreementExpr",
                                    description="Set up model, no. of models, agreement rate among models, file path to save the results.")

    # Positional mandatory arguments
    parser.add_argument("-sfg","--save_fig_path", help="Absolute folder location to save the plots", type=str)
    parser.add_argument("-m","--model_name", help="name of the model", type=str)
    parser.add_argument("-n","--no_of_models", help="number of models to be used", type=int)
    parser.add_argument("-ar","--agreement_rate", help="Agreement rate among models", type=float)
    parser.add_argument("-dataset", help="Dataset to use (HD for heart disease, GE for gene expression)", type=str)
    parser.add_argument("-optm","--optimise_model", help="Whether to use default models or optimised models- 0: default 1: optimise", type=float)

    # Parse arguments
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    if len(sys.argv) != 13:
        print(len(sys.argv))                                                                                                       
        print("usage: ", sys.argv[0], "[-h] [-sfg SAVE_FIG_PATH] [-m MODEL_NAME] [-n NO_OF_MODELS] [-ar AGREEMENT_RATE] [-dataset DATASET] [-optm Whether to optimise model]")  
        sys.exit()                                                                                                         
    # Parse the arguments
    args = parseArguments()
    save_fig_path = args.__dict__['save_fig_path']
    model_name = args.__dict__['model_name']
    n_models = args.__dict__['no_of_models']
    frac_agree = args.__dict__['agreement_rate']
    dataset = args.__dict__['dataset']        
    opt_model = args.__dict__['optimise_model']        

    save_fig_path = "./"+dataset+"/"+model_name+"/"  


# %%
# from codecarbon import EmissionsTracker
# tracker = EmissionsTracker()
# tracker.start()
# try:
#      # Compute intensive code goes here
#      _ = 1 + 1
# finally:
     # tracker.stop()

# %% [markdown]
# ~30 seconds to read TCGA data, ~7 seconds to read STMFR data

# %%
# Remove in py file
# %matplotlib inline

# %% [markdown]
# # Select appropriate dataset
# 
# (1) HD for Heart-Disease
# 
# (2) GE for Gene-Expr

# %%
if dataset == "GE":

    base_dataset = "STMFR"
    derived_dataset = "TCGA"

    # #On bhavesh local machine
    base_file_path = '/Users/bhavesh/Documents/AU/PhD/model_agreement_expr/Dataset/Gene_Expr/stmfr_4_classes.csv'
    to_be_labelled_file_path = '/Users/bhavesh/Library/CloudStorage/GoogleDrive-bhavesh.neekhra_phd18@ashoka.edu.in/My Drive/CR_Data/Data/tcga_final_data.csv'

    # #On lab machine
    # #base_file_path = '/home/bhavesh/Documents/PhD/model_agreement_expr/stmfr_4_classes.csv'
    # #to_be_labelled_file_path = '/home/swapnanil_mukherjee/cancer_data/tcga_final_data.csv'

    # #On lab machine: Works only when Google Drive is mounted 
    # ## stmfr_file_path = '/Users/bhavesh/Library/CloudStorage/GoogleDrive-bhavesh.neekhra_phd18@ashoka.edu.in/My Drive/CR_Data/Data/stmfr_4_classes.csv'

    # #On HPC
    # #base_file_path = '/storage/bhavesh.neekhra_phd18/model_agreement_expr/stmfr_4_classes.csv'
    # #to_be_labelled_file_path = '/storage/bhavesh.neekhra_phd18/model_agreement_expr/tcga_final_data.csv'

elif dataset == "HD":

    base_dataset = "Cleveland"
    derived_dataset = "Hungary"

    # #On bhavesh local machine
    base_file_path = "/Users/bhavesh/Documents/AU/PhD/model_agreement_expr/Dataset/Heart_Disease_UCI_1988/UCI_processed_cleveland.csv"
    to_be_labelled_file_path = "/Users/bhavesh/Documents/AU/PhD/model_agreement_expr/Dataset/Heart_Disease_UCI_1988/UCI_preprocessed_hungary.csv"
    
    
    # #On lab machine
    # base_file_path = './UCI_cleveland.csv'
    # to_be_labelled_file_path = './UCI_hungary.csv'


else:
    print("Name of the dataset is not valid!")



# %%
# # Check whether the specified path exists or not
isExist = os.path.exists(save_fig_path)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(save_fig_path)
   print("The new directory ", save_fig_path, "is created!")

final_results_save_path = "./Final_Results/"+dataset+"/"+model_name+"/"

isExist = os.path.exists(final_results_save_path)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(final_results_save_path)
   print("The new directory ", final_results_save_path, "is created!")

# %%
final_results_save_path

# %%
# from pathlib import Path
# Path(final_results_save_path).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Set up log file

# %%
fname_1 = str(n_models)+"_"+str(model_name)+"_"+str(frac_agree)

log_file = final_results_save_path+"/"+str(n_models)+"_"+str(model_name)+"_"+str(frac_agree)+".log"

try:
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
except NameError:
    pass

logger = logging.getLogger()
logger.setLevel(logging.DEBUG) # process everything, even if everything isn't printed

fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG) # or any level you want
logger.addHandler(fh)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# %% [markdown]
# # Log initial values

# %%
logger.info("\n\n\nExperiments is starting with the following settings:\n")
logger.info("Base Dataset: %s, To be Labelled Dataset: %s", base_dataset, derived_dataset)
logger.info("Model Name: %s", model_name)


logger.info("No. of models: %d", n_models)
logger.info("Agreement rate among models: %.3f", frac_agree)
logger.info("Test size other than 5-fold cross-validation: %.3f", test_split_size)

logger.info("Starting the timer now!")
prog_start = timer()

# %% [markdown]
# # Datasets 

# %%
# Heart Disease dataset for Cleveland (Similar use case as STMFR)

orig_df = pd.read_csv(base_file_path)
df = orig_df

# %%
# Heart Disease dataset for Hungary (Similar use case as TCGA)

orig_tcga = pd.read_csv(to_be_labelled_file_path)
tcga = orig_tcga

# %%
orig_tcga.columns

# %%
# Derived dataset is labelled only for heart disease dataset not for gene expression dataset

if dataset == "HD":
    orig_tcga.rename(columns={"diagnosis": "numeric_label"}, errors="raise",inplace=True)
else:
    pass

# %%
orig_tcga

# %%
# uci_cl_df
df.shape, tcga.shape

# %%
# base_df.columns
df

# %%
tcga

# %% [markdown]
# # Preprocess datasets 

# %%
# Set the index according to the dataset
# This should be standarised - modify in future

if dataset == "GE":
    df = df.set_index('0')
    df = df.rename_axis(index=None)

    tcga = tcga.set_index('Unnamed: 0')
    tcga = tcga.rename_axis(index=None)

elif dataset == "HD":
    df = df.set_index('Unnamed: 0')
    df = df.rename_axis(index=None)

    tcga = tcga.set_index('Unnamed: 0')
    tcga = tcga.rename_axis(index=None)

# %%
tcga

# %%
tcga.columns

# %%
# print(df.isnull().sum().sort_values(ascending=False)), 

# print(tcga.isnull().sum().sort_values(ascending=False))

# %%
df.columns, df.shape

# %%
# The following modification is for TCGA dataset

# tcga.set_index(tcga.loc[:, "Unnamed: 0"], inplace=True)
# tcga.drop("Unnamed: 0", axis=1, inplace=True)

if dataset == "GE":
    cols = df.columns.tolist()
    cols_ = cols[:-1]

    tcga = tcga[cols_]

# Use the following for hungarian datasaet
elif dataset == "HD":
    tcga = tcga[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
       'thalch', 'exang', 'oldpeak']]

# %%
df.shape, tcga.shape

# %%
if dataset == "GE":
    print(df['label'].value_counts())
else:
    pass

# %%
df

# %% [markdown]
# # If using Cleveland dataset from UCI, no need to encode the labels 

# %%
# Used for STMFR dataset

if dataset == "GE":
    label_encoder = LabelEncoder()
    df['numeric_label'] = label_encoder.fit_transform(df.label)
    df.drop('label', axis=1, inplace=True)

# For cleveland dataset, use the following
elif dataset == "HD":
    df['numeric_label'] = df['diagnosis']
    df.drop('diagnosis', axis=1, inplace=True)




# For clevland dataset in df
# Encode the categorical columns: sex, cp, fbs, restecg, exang: For the same columns in hungarian dataset, same encoder should be used or independent? I think it should be same encoder. Verify. 

# label_encoder = LabelEncoder()
# df['sex'] = label_encoder.fit_transform(df.sex)
# logging.info("For sex: ")
# logging.info(label_encoder.classes_)
# Resulting encoding: 1--> Male, 0-->Female

# label_encoder = LabelEncoder()
# df['cp'] = label_encoder.fit_transform(df.cp)
# logging.info("For cp: ")
# logging.info(label_encoder.classes_)
# Resulting encoding: 0--> asymptomatic, 1-->atypical angina, 2-->non-anginal, 3-->typical angina

# label_encoder = LabelEncoder()
# df['fbs'] = label_encoder.fit_transform(df.fbs)
# logging.info("For fbs: ")
# logging.info(label_encoder.classes_)
# Resulting encoding: 0--> False, 1-->True

# label_encoder = LabelEncoder()
# df['restecg'] = label_encoder.fit_transform(df.restecg)
# logging.info("For restecg: ")
# logging.info(label_encoder.classes_)
# Resulting encoding: 0--> lv hypertrophy, 1-->normal, 2-->st-t abnormality


# label_encoder = LabelEncoder()
# df['exang'] = label_encoder.fit_transform(df.exang)
# logging.info("For exang: ")
# logging.info(label_encoder.classes_)
# Resulting encoding: 0--> False, 1-->True

# %%
# Both dataframes should have same column order 

df.columns, tcga.columns

# %% [markdown]
# # If using Hungary dataset from UCI, no need to encode the labels 

# %%
# For hungarian dataset
# Encode the categorical columns: sex, cp, fbs, restecg, exang

# logging.info("For hungarian dataset: ")
# label_encoder = LabelEncoder()
# tcga['sex'] = label_encoder.fit_transform(tcga.sex)
# logging.info("For sex: ")
# logging.info(label_encoder.classes_)
# Resulting encoding: 1--> Male, 0-->Female

# label_encoder = LabelEncoder()
# tcga['cp'] = label_encoder.fit_transform(tcga.cp)
# logging.info("For cp: ")
# logging.info(label_encoder.classes_)
# Resulting encoding: 0--> asymptomatic, 1-->atypical angina, 2-->non-anginal, 3-->typical angina

# label_encoder = LabelEncoder()
# tcga['fbs'] = label_encoder.fit_transform(tcga.fbs)
# logging.info("For fbs: ")
# logging.info(label_encoder.classes_)
# Resulting encoding: 0--> False, 1-->True

# label_encoder = LabelEncoder()
# tcga['restecg'] = label_encoder.fit_transform(tcga.restecg)
# logging.info("For restecg: ")
# logging.info(label_encoder.classes_)
# Resulting encoding: 0--> lv hypertrophy, 1-->normal, 2-->st-t abnormality


# label_encoder = LabelEncoder()
# tcga['exang'] = label_encoder.fit_transform(tcga.exang)
# logging.info("For exang: ")
# logging.info(label_encoder.classes_)
# Resulting encoding: 0--> False, 1-->True

# %%
df.groupby(['numeric_label'])['numeric_label'].value_counts()

# %%
# Update the numeric_label as following for UCI health disease dataset
# there are 5 classes- we can combine these in two 0-->0, 1,2,3,4-->1
if dataset == "HD":
    df.loc[df["numeric_label"] == 2, "numeric_label"] = 1
    df.loc[df["numeric_label"] == 3, "numeric_label"] = 1
    df.loc[df["numeric_label"] == 4, "numeric_label"] = 1


# %%
# tcga

# %%
logging.info("Class wise distribution in original dataset %s: ", base_dataset)
logging.info(df['numeric_label'].value_counts())

# Commented for heart disease dataset
# logging.info(label_encoder.classes_)

# %% [markdown]
# # Define models

# %%
def get_model(model_name):
  if dataset == "GE":

    model_dict = {
          'Logistic Regression':LogisticRegression(n_jobs=-1, max_iter=250, random_state=random_state),
          'LR':LogisticRegression(n_jobs=-1, max_iter=250, random_state=random_state),

          #  Prefer dual=False when n_samples > n_features
          # default value for dual =- True but in future it will be auto
          'SVM':LinearSVC(dual = True, max_iter=10000, random_state=random_state), #For STMFR+TCGA
          # 'SVM':LinearSVC(max_iter=10000, dual=False, random_state=random_state),

          # 'Decision Tree':DecisionTreeClassifier(min_samples_split=16, min_samples_leaf=3, max_depth= 15, criterion= 'entropy',random_state=random_state),
          # 'DT':DecisionTreeClassifier(min_samples_split=16, min_samples_leaf=3, max_depth= 15, criterion= 'entropy', random_state=random_state),
          'Decision Tree':DecisionTreeClassifier(random_state=random_state),
          'DT':DecisionTreeClassifier(random_state=random_state),
          
          'Random Forest':RandomForestClassifier(n_estimators=10, random_state=random_state),
          'RF':RandomForestClassifier(n_estimators=10, random_state=random_state),

          # Had 512, 256 setting for GE data
          "Neural Network":MLPClassifier(hidden_layer_sizes=[512, 256], solver='sgd', batch_size=32, max_iter=100, random_state=random_state),
          "NN":MLPClassifier(hidden_layer_sizes=[512, 256], solver='sgd', batch_size=32, max_iter=100, random_state=random_state),
          "MLP":MLPClassifier(hidden_layer_sizes=[512, 256], solver='sgd', batch_size=32, max_iter=100, random_state=random_state),
          
          "XGBoost": XGBClassifier(num_class=4, objective='multi:softmax', random_state=random_state),
          "XGB": XGBClassifier(num_class=4, objective='multi:softmax', random_state=random_state)
          }
  elif dataset == "HD" and not(opt_model):
            # Use default models if opt_model = 0 
    model_dict = {
          # 'Logistic Regression':LogisticRegression(n_jobs=-1, max_iter=250, random_state=random_state),
          # 'LR':LogisticRegression(n_jobs=-1, max_iter=250, random_state=random_state),

          'Logistic Regression':LogisticRegression(n_jobs=-1, random_state=random_state),
          'LR':LogisticRegression(n_jobs=-1, random_state=random_state),

        #  Prefer dual=False when n_samples > n_features 
          # 'SVM':LinearSVC(dual = True, max_iter=10000), For STMFR+TCGA
          # 'SVM':LinearSVC(max_iter=10000, dual=False, random_state=random_state),
          'SVM':LinearSVC(dual=False, random_state=random_state),

          'Decision Tree':DecisionTreeClassifier(random_state=random_state),
          'DT':DecisionTreeClassifier(random_state=random_state),

          # 'Random Forest':RandomForestClassifier(n_estimators=10, random_state=random_state),
          # 'RF':RandomForestClassifier(n_estimators=10, random_state=random_state),

          'Random Forest':RandomForestClassifier(random_state=random_state),
          'RF':RandomForestClassifier(random_state=random_state),

          # Had 512, 256 setting for GE data
          # "Neural Network":MLPClassifier(hidden_layer_sizes=[256, 128], solver='sgd', batch_size=32, max_iter=100, random_state=random_state),
          # "NN":MLPClassifier(hidden_layer_sizes=[256, 128], solver='sgd', batch_size=32, max_iter=100, random_state=random_state),
          # "MLP":MLPClassifier(hidden_layer_sizes=[256, 128], solver='sgd', batch_size=32, max_iter=100, random_state=random_state),

          "Neural Network":MLPClassifier(hidden_layer_sizes=[256, 128], batch_size=32, random_state=random_state),
          "NN":MLPClassifier(hidden_layer_sizes=[256, 128], batch_size=32, random_state=random_state),
          "MLP":MLPClassifier(hidden_layer_sizes=[256, 128], batch_size=32, random_state=random_state),
          
          "XGBoost": XGBClassifier(objective='binary:logistic', random_state=random_state),
          "XGB": XGBClassifier(objective='binary:logistic', random_state=random_state)
          }
  
  elif dataset == "HD" and opt_model:  
      # Use optimised models if opt_model = 1
      model_dict = {
          
          # 'Logistic Regression':LogisticRegression(n_jobs=-1, max_iter=250, random_state=random_state),
          # 'LR':LogisticRegression(n_jobs=-1, max_iter=250, random_state=random_state),

          'Logistic Regression':LogisticRegression(max_iter=50, n_jobs=-1, random_state=42, solver='liblinear'),
          'LR': LogisticRegression(max_iter=50, n_jobs=-1, random_state=42, solver='liblinear'),

          'SVM':LinearSVC(dual=False, random_state=random_state),

          'Decision Tree':DecisionTreeClassifier(min_samples_split = 8, min_samples_leaf = 15, max_depth = 3, criterion = 'gini',random_state=random_state),
          'DT':DecisionTreeClassifier(min_samples_split = 8, min_samples_leaf = 15, max_depth = 3, criterion = 'gini',random_state=random_state),

          'Random Forest':RandomForestClassifier(n_estimators=10, random_state=random_state),
          'RF':RandomForestClassifier(n_estimators=10, random_state=random_state),

          "Neural Network":MLPClassifier(hidden_layer_sizes=[256, 128], batch_size=32, random_state=random_state),
          "NN":MLPClassifier(hidden_layer_sizes=[256, 128], batch_size=32, random_state=random_state),
          "MLP":MLPClassifier(hidden_layer_sizes=[256, 128], batch_size=32, random_state=random_state),
          
          "XGBoost": XGBClassifier(objective='binary:logistic', random_state=random_state),
          "XGB": XGBClassifier(objective='binary:logistic', random_state=random_state)
          }


  model_name = model_name.strip()

  if model_name not in model_dict:
    print("Model name should be one of", list(model_dict.keys()))
    print("Please check the entered model name for spelling errors.")
    sys.exit()

  else:
    model = model_dict[model_name]
    return model

# %%
# plt.rcParams.update({'font.size': 8})

logging.info("\n\nUsing the model: %s\n\n", get_model(model_name))

# %% [markdown]
# # helper functions

# %%
global box_plot_num 
box_plot_num = 0
global model_prediction_agreement 
model_prediction_agreement = 0

# %%
# Creating plot
# plt.boxplot(model_acc_list, patch_artist = True,
#                 notch ='True', vert = 0)
def box_plot_models_accuracy(model_acc_list):

    global box_plot_num
    # fig = plt.figure(figsize =(6, 4))    
    plt.boxplot(model_acc_list, vert = 0, )
    
    plt.xlabel("accuracy")
    # plt.title('PCA with 500 components\n 90-10 data split')   # Change the font setting for the title 

    

    fname = fname_1+'_b_p_'+str(box_plot_num)+'.png'
    box_plot_num = box_plot_num + 1
    plt.savefig(save_fig_path+fname)
    plt.close()


# %%
def get_agreement_rate(model_pred, n_models, len_tcga, frac_agree):

    # print(model_pred)
    model_pred_results = pd.DataFrame(model_pred)
    # print(model_pred_results)
    # model_pred_results.to_csv("model_agreements.csv", index=False)
    len_model_pred_results = len(model_pred_results.columns)
    min_agree_list = []

    count_models = n_models # For prototyping use 5, and change it back to 100
    samples_index = []

    low = np.int16(np.ceil(0.4*count_models))
    high = int(count_models*frac_agree) + 1

    # model_pred_results.to_csv("./model_prediction_results.csv")

    # print("Low: ",low, "High ", high)
    # for min in range(40,101):
    for min in range(low, n_models+1):     
        c = 0
        for i in range(len_model_pred_results):
            counter_values = list(Counter(model_pred_results[i]).values())
            # logger.info("Testing: %s ",Counter(model_pred_results[i]))

            if any(j >= min for j in counter_values):
                c+=1
                # print("Agreement Rate: ",count_models*frac_agree)
                if(min == int(count_models*frac_agree)):

                    # Counter returns the counter per item in the list
                    # Get the most common item 
                    # Get the item for which the count is the highest 
                    label = (Counter(model_pred_results[i]).most_common(1)[0])[0]
                    samples_index.append([i, label])
                    
                    # logger.info("************************************************************")
                    # logger.info("Sample: %d, model Prediction: %s, Counter: %s", i, model_pred_results[i][0], (Counter(model_pred_results[i]).most_common(1)[0])[0])
                    # logger.info("**********************************************************************")
                    # logger.info(model_pred_results[i])
                    # logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    
                    exit
                    # print ("i ",i, "Results: ",model_pred_results[i])
        agree_score = c/len_tcga
        min_agree_list.append([min, agree_score])

    min_agree_df = pd.DataFrame(min_agree_list)
    logger.info("No of samples for which all models agreed: %d", len(samples_index))
    # logger.info(len(samples_index))
    
    # pd.DataFrame(samples_index).to_csv("./samples_index.csv")
    # min_agree_df.to_csv("./min_agree_df.csv")
    
    return samples_index, min_agree_df

# %% [markdown]
# The followin code helps undestand the logic in the function get_agreement_rate for getting the label agreed upon by frac_agree models
# 
# a = [1, 2, 1, 1,1,1,2,2,2,3,3,3,3,3,3,3,3,3]
# 
# print(Counter(a))
# 
# print(Counter(a).most_common(1))
# 
# (Counter(a).most_common(1)[0])[0]

# %%
# low = np.int16(np.ceil(0.4*n_models))
# high = n_models+1
# x = range(low, high, 10)
# for n in x:
#     print(n)

# %%
def plot_model_accuracy(min_agree_df, len_train, len_test, len_ood):

    global model_prediction_agreement
    x = min_agree_df[0]
    y = min_agree_df[1]

    # make data
    # x = np.linspace(0, 10, 100)
    # y = 4 + 2 * np.sin(2 * x)

    # plot
    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2.0)

    # ax.set(xlim=(0, 100),
    #        ylim=(0, 1.1),)

    low = np.int16(np.ceil(0.4*n_models))
    high = n_models+1
    annot_plot_at = range(low, high, 10)

    for i,j in zip(x,y):
        if(i in annot_plot_at):
            ax.annotate(f"{j:.1%}",xy=(i,j))


    ax.set(ylim=(0, 1.1))
    plt.xlabel('Number of models')
    plt.ylabel('% agreement')
    plt.title('Train: '+str(len_train)+' Test: '+str(len_test)+' OOD: '+str(len_ood))
    fname_1 = str(n_models)+"_"+str(model_name)+"_"+str(frac_agree)
    fname = fname_1+'_m_a_p_'+str(model_prediction_agreement)+'.png'
    model_prediction_agreement = model_prediction_agreement + 1
    plt.savefig(save_fig_path+fname)
    plt.close()

# %%
def merge_samples_tcga_stmfr(samples_index, stmfr_df, tcga_df):
    
    if len(samples_index) == 0:
        return stmfr_df, "", tcga_df
    
    samples_index_df = pd.DataFrame(samples_index)
    # samples_index_df
    selected_tcga_samples = tcga_df.iloc[samples_index_df[0]]

    # a = len(tcga_df)
    # b = range(a)
    # c = set(b)
    # d = list(c)

    # e = set(samples_index_df[0])
    
    ood_index = list(set(range(len(tcga_df))) - set(samples_index_df[0]))
    
    ood_tcga_samples = tcga_df.iloc[ood_index]

    selected_tcga_samples.loc[:,"numeric_label"] = list(samples_index_df[1])


    combined_train_stmfr_tcga = pd.concat([stmfr_df, selected_tcga_samples])
    return combined_train_stmfr_tcga, "", ood_tcga_samples
    
    # if len(selected_tcga_samples) == 1:
    #     combined_train_stmfr_tcga = pd.concat([stmfr_df, selected_tcga_samples])
    #     return combined_train_stmfr_tcga, "", ood_tcga_samples
    # else:
    #     selected_tcga_samples_train, selected_tcga_samples_test = train_test_split(selected_tcga_samples, test_size=0.2, random_state=42, shuffle=True) 
    #     combined_train_stmfr_tcga = pd.concat([stmfr_df, selected_tcga_samples_train])
    #     return combined_train_stmfr_tcga, selected_tcga_samples_test, ood_tcga_samples

# %%
# !pip install hyperopt

# %%
# import hyperopt.pyll.stochastic

# print(hyperopt.pyll.stochastic.sample(space))

# %% [markdown]
# # hyperparameter selection

# %% [markdown]
# # 

# %%
# train, test = train_test_split(df, test_size=test_split_size, random_state=random_state, shuffle=True)

# X_train, y_train = train.iloc[:, :-1], train['numeric_label']
# X_test, y_test = test.iloc[:, :-1], test['numeric_label']

# %% [markdown]
# # LR

# %%
# # LR

# model = LogisticRegression(n_jobs=-1, random_state=random_state)

# # model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)

# model.fit(X_train, y_train)
# logger.info("Default model LR's performance: ")
# report_before = classification_report(model.predict(X_test), y_test)
# logger.info(report_before)


# %%
# param_grid_lr = {
#     'max_iter': [50, 100, 200, 250, 500, 750, 1000],                      
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],   
#     # 'class_weight': ['balanced']                                    
# }

# # logModel_grid = GridSearchCV(estimator=model, param_grid=param_grid_lr, verbose=1, cv=10, n_jobs=-1)
# logModel_grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid_lr, cv=10, n_iter=100, n_jobs=-1, verbose=True, scoring='accuracy')

# logModel_grid.fit(X_train, y_train)

# model = logModel_grid.best_estimator_
# # model = LogisticRegression(max_iter=100, solver='liblinear', random_state=42, n_jobs=-1)

# model.fit(X_train, y_train)

# logger.info("Best parameteres: ")
# logger.info(logModel_grid.best_estimator_)
# logger.info("model LR's performance after best parameter search: ")
# report_after = classification_report(model.predict(X_test), y_test)
# logger.info(report_after)

# %% [markdown]
# HD two classes
# 
# LogisticRegression(max_iter=50, n_jobs=-1, random_state=42, solver='liblinear')

# %%
# print(logModel_grid.best_estimator_)
# print(report_after)

# %%
# print(report_before)

# %% [markdown]
# # SVM

# %%
# # log-uniform: understand as search over p = exp(x) by varying x


# # LinearSVC(max_iter=10000, dual=False, random_state=random_state),

# opt = RandomizedSearchCV(
#     LinearSVC(),
#     {
#         'max_iter': [100, 250, 500, 750, 1000, 5000, 10000], 
#         'dual': [True, False]
#     },
#     n_iter=100,
#     cv=5,
#     error_score='raise',
#      n_jobs=-1,
#     verbose=True, 
#     scoring='accuracy'
# )

# opt.fit(X_train, y_train)

# print("val. score: %s" % opt.best_score_)
# print("test score: %s" % opt.score(X_test, y_test))

# %%
# print(opt.best_estimator_)
# opt.best_estimator_


# %% [markdown]
# LinearSVC(dual=False, max_iter=100)

# %% [markdown]
# # DT

# %%
# # DecisionTreeClassifier(min_samples_split = 8, min_samples_leaf = 15, max_depth = 3, criterion = 'gini',random_state=random_state)


# opt = RandomizedSearchCV(
#     DecisionTreeClassifier(),
#     {
#         'max_depth':[3, 5, 7, 10, 15],
#         'min_samples_leaf':[1, 3, 5, 10, 15,20],
#         'min_samples_split':[2, 4, 6, 8, 10, 12, 16, 18, 20],
#         'criterion':['log_loss', 'gini','entropy']
#     },
#     n_iter=100,
#     cv=5,
#     error_score='raise',
#     n_jobs=-1,
#     verbose=True, 
#     scoring='accuracy'
# )

# opt.fit(X_train, y_train)


# print("Best model: %s" % opt.best_estimator_)
# print("val. score: %s" % opt.best_score_)
# print("test score: %s" % opt.score(X_test, y_test))

# %%
# DT_model = DecisionTreeClassifier(min_samples_split = 8, min_samples_leaf = 15, max_depth = 3, criterion = 'gini',random_state=random_state)
# DT_model.fit(X_train, y_train)

# DT_model.score(X_test, y_test)

# %% [markdown]
# # RF 

# %%
# # RandomForestClassifier(n_estimators=10, random_state=random_state),


# opt = RandomizedSearchCV(
#     RandomForestClassifier(),
#     {
#         'n_estimators':[20,50, 100, 200, 500],
#         'criterion':['log_loss', 'gini','entropy']
#     },
#     n_iter=100,
#     cv=5,
#     error_score='raise',
#     n_jobs=-1,
#     verbose=True, 
#     scoring='accuracy'
# )

# opt.fit(X_train, y_train)


# print("Best model: %s" % opt.best_estimator_)
# print("val. score: %s" % opt.best_score_)
# print("test score: %s" % opt.score(X_test, y_test))

# %%
# RF_model = RandomForestClassifier(n_estimators=10, random_state=random_state)
# RF_model.fit(X_train, y_train)

# RF_model.score(X_test, y_test)

# %% [markdown]
# # MLP

# %%
# # "MLP":MLPClassifier(hidden_layer_sizes=[256, 128], batch_size=32, random_state=random_state),



# opt = RandomizedSearchCV(
#     MLPClassifier(),
#     {
#         'hidden_layer_sizes':[(128, 64), (256, 128)],
#         'max_iter': [50, 100, 200, 500, 1000, 5000], # Default 200,
#     },
#     n_iter=12,
#     cv=5,
#     error_score='raise',
#     n_jobs=-1,
#     # verbose=True, 
#     scoring='accuracy'
# )

# opt.fit(X_train, y_train)


# print("Best model: %s" % opt.best_estimator_)
# print("val. score: %s" % opt.best_score_)
# print("test score: %s" % opt.score(X_test, y_test))

# %%
# MLP_model = MLPClassifier(hidden_layer_sizes=[256, 128], batch_size=32, random_state=random_state)
# MLP_model.fit(X_train, y_train)

# MLP_model.score(X_test, y_test)

# %%


# model = DecisionTreeClassifier(random_state=42)

# params = {'max_depth':[3, 5, 7, 10, 15],
#           'min_samples_leaf':[3, 5, 10, 15,20],
#           'min_samples_split':[8, 10, 12, 16, 18, 20],
#           'criterion':['gini','entropy']}
# RS_DT = RandomizedSearchCV(estimator=model, param_distributions=params, cv=5, n_iter=300, n_jobs=-1, verbose=True, scoring='accuracy')

# RS_DT.fit(X_train, y_train)

# # print(X_train[0])
# print("Dataset: ", base_dataset, df.shape)
# print("Count of labels: ", df.groupby(['numeric_label'])['numeric_label'].value_counts())
# print("Best Parameters:", RS_DT.best_params_,end='\n\n')
# print("Best Score:", RS_DT.best_score_)


# import xgboost as xgb
# from hyperopt import fmin, tpe, hp
# from hyperopt import STATUS_OK

# # Define the hyperparameter space
# space = {
#     'max_depth': hp.uniformint('max_depth', 2, 8),
#     'learning_rate': hp.loguniform('learning_rate', -5, -2),
#     'subsample': hp.uniform('subsample', 0.5, 1)
# }

# train, test = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

# # train, test = train_test_split(stmfr_data_train, test_size=0.1, random_state=i, shuffle=True)

# X_train, y_train = train.iloc[:, :-1], train['numeric_label']
# X_test, y_test = test.iloc[:, :-1], test['numeric_label']

# # Define the objective function to minimize
# def objective(params):
#     xgb_model = xgb.XGBClassifier(**params)
#     xgb_model.fit(X_train, y_train)
#     y_pred = xgb_model.predict(X_test)
#     score = accuracy_score(y_test, y_pred)
#     return {'loss': -score, 'status': STATUS_OK}

# # Perform the optimization
# best_params = fmin(objective, space, algo=tpe.suggest, max_evals=10)
# print("Best set of hyperparameters: ", best_params)

# %% [markdown]
# hyperopt with 10 evaluations, for XGBoost, Best set of hyperparameters:  {'learning_rate': 0.11716612264320947, 'max_depth': 4.0, 'subsample': 0.8195573834624639}
#                                                             
#                                                             default: eta [default=0.3, alias: learning_rate], max_depth [default=6], subsample [default=1] 
#                                                                         (Source: https://xgboost.readthedocs.io/en/release_0.82/parameter.html)

# %%


# %% [markdown]
# For cleveland dataset:
# 
# For Decision Tree, RandomizedSearchCV
# 
# 
# With only two classes: 0 and (1,2,3,4)-->1
# 
# Best Parameters: {'min_samples_split': 8, 'min_samples_leaf': 15, 'max_depth': 3, 'criterion': 'gini'}
# 
# Best Score: 0.7441326530612244
# 
# 
# with 5 classes: 0, 1, 2, 3, 4 
# 
# Best Parameters:  {'min_samples_split': 8, 'min_samples_leaf': 3, 'max_depth': 5, 'criterion': 'gini'}
# 
# Best Score: 1.0
# 
# Why is there such difference? Understand it. 

# %% [markdown]
# For STMFR dataset:
# 
# For Decision Tree, RandomizedSearchCV
# 
# Best Parameters: {'min_samples_split': 16, 'min_samples_leaf': 3, 'max_depth': 15, 'criterion': 'entropy'}
# 
# Defatul Parameters: {'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': None, 'criterion': 'gini'}
# 
# Best Score: 0.869449715370019

# %%
# accuracy_score(y_test, RS_DT.predict(X_test))

# %% [markdown]
# # Check model's performance on datasets
# 

# %%
 # import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")   /. 

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# train, test = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

# train, test = train_test_split(stmfr_data_train, test_size=0.1, random_state=i, shuffle=True)

# X_train, y_train = train.iloc[:, :-1], train['numeric_label']
# X_test, y_test = test.iloc[:, :-1], test['numeric_label']

def check_model_stmfr(df, name_df):
        
    x = df.iloc[:, :-1]
    y = df['numeric_label']

    # scaler = StandardScaler()

    # x_rescaled = pd.DataFrame(scaler.fit_transform(x), index=x.index, columns=x.columns)

    acc_list = []
    b_acc_list = []
    f1_list = []
    prec_list = []
    rec_list = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(x, y):
        
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # bst = XGBClassifier(n_estimators=2, max_depth=20, learning_rate=1, num_class=4, objective='multi:softmax')
        
        model = get_model(model_name)
        
        # model = DecisionTreeClassifier(min_samples_split=16, min_samples_leaf=3, max_depth= 15, criterion= 'entropy')
        # model = XGBClassifier(num_class=4, objective='multi:softmax')  #, max_depth=4, learning_rate=0.117, subsample=0.82)

        # model = LogisticRegression(n_jobs=-1, max_iter=250)
        
        # pca = PCA(n_components=0.99)
        # X_train_pca = pca.fit_transform(X_train)
        # X_test_pca = pca.transform(X_test)
        if debug:
            print("\n\n******** Train ************\n\n")
            print(X_train.shape, y_train.shape)
            print("\n\n********* Test ***********\n\n")
            print(X_test.shape, y_test.shape)
            print("\n\n**************************\n\n")
        # fit model
    
        model.fit(X_train, y_train)
        # make predictions
        predictions = model.predict(X_test)

        # dtrain = xgb.DMatrix(X_train_pca, label=y_train)
        # dtrain.save_binary('train.buffer')

        # param = {'max_depth': 20, 'eta': 1, 'objective': 'multi:softmax'}
        # param['nthread'] = 4
        # param['eval_metric'] = 'auc'
        # param['num_class'] = 4

        # num_round = 10
        # bst = xgb.train(param, dtrain, num_round)

        # dtest = xgb.DMatrix(X_test_pca)
        # predictions = bst.predict(dtest)

        # target_names = ['hESC', 'hMSC', 'hUSC', 'iPSC']
        # print(classification_report(y_test, predictions, target_names=target_names))
        # print(classification_report(y_test, predictions))
        logger.info(classification_report(y_test, predictions))
        # model_acc = accuracy_score(y_test, predictions, normalize=True)
        # model_acc

        accuracy = accuracy_score(y_test, predictions)
        acc_list.append(accuracy)

        b_accuracy = balanced_accuracy_score(y_test, predictions)
        b_acc_list.append(b_accuracy)

        f1score = f1_score(y_test, predictions, average='weighted')
        f1_list.append(f1score)

        precision = precision_score(y_test, predictions, average='weighted')
        prec_list.append(precision)

        recall = recall_score(y_test, predictions, average='weighted')
        rec_list.append(recall)

        # print(accuracy, end='\t')
        # print(b_accuracy, end='\t')
        # print(f1score, end='\t')
        # print(precision, end='\t')
        # print(recall)

        # break

    avg_acc = np.average(acc_list)
    print("Average for 5-fold cross-validation:\n")

    logger.info("Accuracy of model %s on %s dataset is %f",model_name, name_df, avg_acc)

    print("Accuracy: ", avg_acc, end='\t')
    print("Balanced Accuracy: ", np.average(b_acc_list), end='\t')
    print("F1 score: ", np.average(f1_list), end='\t')
    print("Precision: ", np.average(prec_list), end='\t')
    print("Recall: ", np.average(rec_list))
        # xgb.plot_importance(bst)
        # xgb.plot_tree(bst, num_trees=2)
        # xgb.to_graphviz(bst, num_trees=2)
    return avg_acc

# %% [markdown]
# Source: https://mikulskibartosz.name/pca-how-to-choose-the-number-of-components 
# 
# For PCA = variance required
# 
# Default setting of XGBoost (What is it for n_estimators (100?), max_depth (3?), learning rate (?)
# 
# Max depth = 20, 90/10
# with all features Xgboost acc = 94.55
# with 100 PCA , 0.9363636363636364 , 0.9545454545454546 , 0.9575757575757575, 0.9515151515151515
# with 55 PCA, 0.9424242424242424
# 
# 
# 70/30
# 
# 100 PCA , 0.9332659251769464 , 0.9362992922143579 
# 
# Max depth = 20, 70/30, w/o PCA,0.9433771486349848, 2m 34.2s
# Max depth = 20, 70/30, 100 PCA,0.9332659251769464, 58.2s, 0.9322548028311426(45.0s)

# %% [markdown]
# Dataset: cleveland
# 
# Model: LR 
# 
# Results for five class setting:
# Accuracy:  0.8085792349726775	Balanced Accuracy:  0.5924891774891774	F1 score:  0.7962348755636871	Precision:  0.8028553301474588	Recall:  0.8085792349726775
# 
# Results for two class setting:
# Accuracy:  0.7920765027322403	Balanced Accuracy:  0.7877269721019722	F1 score:  0.7909195083310058	Precision:  0.7941178143153096	Recall:  0.7920765027322403

# %% [markdown]
# # 1.1 Train-test on base dataset

# %%
check_model_stmfr(df, base_dataset)

# %%
logger.info("1.1: Train-test on %s (%s): ", base_dataset, df.shape)
avg_acc = check_model_stmfr(df, base_dataset)

# "s_no.", "description","value"

add_to_final_results = {"s_no": '1.1', 'description': 'Train-test on '+base_dataset, 'value': avg_acc}

# final_results = final_results.append(add_to_final_results, ignore_index=True)

final_results.loc[len(final_results)] = add_to_final_results


# %% [markdown]
# # 1.1a: Train-test on to_be_labelled dataset 
# (if labels area available)

# %%
# orig_tcga.columns

orig_tcga = orig_tcga.set_index('Unnamed: 0')
orig_tcga = orig_tcga.rename_axis(index=None)

# %%
orig_tcga

# %%
# tcga.groupby(['numeric_label'])['numeric_label'].value_counts()
# orig_tcga['numeric_label']

# %%
if dataset == "HD":
    tcga['numeric_label'] = orig_tcga['numeric_label']
else:
    pass

# %%
tcga

# %%
if dataset == "HD":

    logger.info("1.1a: Train-test on %s (%s): ", derived_dataset, tcga.shape)
    avg_acc = check_model_stmfr(tcga, derived_dataset)

    # "s_no.", "description","value"

    add_to_final_results = {"s_no": '1.1a', 'description': 'Train-test on '+derived_dataset, 'value': avg_acc}

    tcga.drop('numeric_label', axis=1, inplace=True)
    tcga.shape
elif dataset == "GE":
    add_to_final_results = {"s_no": '1.1a', 'description': 'Train-test on '+derived_dataset, 'value': np.nan}

final_results.loc[len(final_results)] = add_to_final_results

# %%
final_results

# %% [markdown]
# Default DT on STMFR
# 
# 0.8679817905918058	0.8580144133550927	0.8674333245540867	0.8671897067674726	0.8679817905918058
# 
# 0.8801213960546282	0.8684092785881865	0.8794070838121345	0.879206748287741	0.8801213960546282
# 
# 0.8619119878603946	0.8611241916606627	0.8622920460212726	0.8635377529870443	0.8619119878603946
# 
# 0.8831562974203339	0.8752958891515941	0.8826011778450076	0.8829919950745394	0.8831562974203339
# 
# 0.8768996960486323	0.8711173085436268	0.8769341574013189	0.8771383272601282	0.8768996960486323
# 
# Average for 5-fold cross-validation:
# 
# Accuracy:  0.8740142335951588	Balanced Accuracy:  0.8667922162598325	F1 score:  0.873733557926764	Precision:  0.8740129060753852	Recall:  0.8740142335951588

# %% [markdown]
# in vim:
# 
# %s/findfont:.*\n//gc

# %% [markdown]
# # main function
# 

# %%
def prediction_agreement_models(stmfr_df, tcga_df, held_out_data = "", n_models = 100, frac_agree = 1, pca = False, pca_n_components = 100):

    n = n_models
    model_acc_list = []
    val_acc_list = []
    model_pred = []
    df = stmfr_df
    tcga = tcga_df

    val_data = False

    # pca_n_comps = [1, 5, 10, 50, 100, 500, 1000, 2000]
    # pca_n_comps = [200, 300, 400, 600, 700, 800, 900]

    for i in tqdm(range(n)):
    # for i in pca_n_comps:
        train, test = train_test_split(df, test_size=test_split_size, random_state=i, shuffle=True)

        # train, test = train_test_split(stmfr_data_train, test_size=0.1, random_state=i, shuffle=True)

        X_train, y_train = train.iloc[:, :-1], train['numeric_label']
        X_test, y_test = test.iloc[:, :-1], test['numeric_label']

        # scaler = preprocessing.StandardScaler().fit(X_train)
        # pipe = make_pipeline(StandardScaler(), LinearSVC(dual = True, max_iter=10000))
        # pipe.fit(X_train, y_train)
        # pipe.score(X_test, y_test)

        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        if pca:
                
            pca = PCA(n_components=pca_n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            tcga_pca = pca.transform(tcga)
            
            # model = LinearSVC(dual = False, max_iter=10000)
        else:
            X_train_pca = X_train
            X_test_pca = X_test
            tcga_pca = tcga

            # model = LinearSVC(dual = True, max_iter=10000)
        
        if (i==0):
            logging.info(model_name)

        model =get_model(model_name)
        # model = LogisticRegression(n_jobs=-1, max_iter=250)
        
        # model_coeffs = model.coef_
        # mlp_model = MLPClassifier(hidden_layer_sizes=[10, 10], solver='sgd', batch_size=256, max_iter=100)

        # dtrain = xgb.DMatrix(X_train_pca, label=y_train)
        # # dtrain.save_binary('train.buffer')

        # param = {'max_depth': 20, 'eta': 1, 'objective': 'multi:softmax'}
        # param['nthread'] = 4
        # param['eval_metric'] = 'auc'
        # param['num_class'] = 4

        # num_round = 10
        # model = xgb.train(param, dtrain, num_round)

        # dtest = xgb.DMatrix(X_test_pca)
        # predictions = model.predict(dtest)

        model.fit(X_train_pca, y_train)
        predictions = model.predict(X_test_pca)

        try:   
            if not held_out_data.empty:
                val_data = True
                X_validation, y_validation = held_out_data.iloc[:, :-1], held_out_data['numeric_label']
                if pca:
                    X_validation_pca = pca.transform(X_validation)
                else:
                    X_validation_pca = X_validation

                # dvalid = xgb.DMatrix(X_validation_pca)
                # validation_prediction = model.predict(dvalid)
                validation_prediction = model.predict(X_validation_pca)
                validation_prediction_acc = accuracy_score(y_validation, validation_prediction, normalize=True)
                val_acc_list.append(validation_prediction_acc)
                if(i%10 == 0):
                    logger.info("For "+str(i)+" model's accuracy on heldout OOD data: "+str(validation_prediction_acc))
        except Exception as e:
            #Handle the exception
            # print("An exception occurred: ", e)
            # logging.warning("An exception occurred: ")
            # logging.warning(e) 
            pass

        # report = classification_report(y_test, predictions, output_dict=True)
        model_acc = accuracy_score(y_test, predictions, normalize=True)
        model_acc_list.append(model_acc)

        # print("For", i, "LinearSVC accuracy: ", model_acc)
        # dvalid = xgb.DMatrix(tcga_pca)
        # model_pred.append(model.predict(dvalid))

        model_pred.append(model.predict(tcga_pca))

        # model_pred.append(model.predict(stmfr_data_predict))

        # results = pd.DataFrame(report).transpose()
        # results = results.round({'precision': 2, 'recall': 2, 'f1-score': 2, 'support':1})
        # model.predict(tcga)
        

    logging.info("For "+str(n)+" models, test accuracy avg and stddev: "+str(np.mean(model_acc_list))+" "+str(np.std(model_acc_list)))
    if(val_data):
        logging.info("For "+str(n)+" models, validation accuracy (on held out OOD) avg and stddev: "+str(np.mean(val_acc_list))+" "+str(np.std(val_acc_list)))

    # box_plot_models_accuracy(model_acc_list)
    samples_index, min_agree_df = get_agreement_rate(model_pred, n ,len(tcga), frac_agree)
    logging.info("No of OOD samples for which labelling is done: "+str(len(samples_index)))
    plot_model_accuracy(min_agree_df, len(X_train), len(X_test), len(tcga))
    # if len(samples_index) != 0:
    combined_train_stmfr_tcga, selected_tcga_samples_test, ood_tcga_samples = merge_samples_tcga_stmfr(samples_index, stmfr_df, tcga_df)
    # else:
        # logging.info("No agreement among models. Returning the dfs without any merger")
        # return stmfr_df, [], tcga_df
    
    
    return combined_train_stmfr_tcga, selected_tcga_samples_test, ood_tcga_samples

# %% [markdown]
# # Forward Pass

# %%
start = timer()

logger.info("for model: ")
logger.info(model_name)

logger.info("Round 1 starts.")
combined_train_stmfr_tcga, selected_tcga_samples_test, ood_tcga_samples = prediction_agreement_models(df, tcga, "", n_models=n_models, frac_agree=frac_agree)


logger.info("Length of combined_train_stmfr_tcga, selected_tcga_test, ood_tcga: "+
                str(len(combined_train_stmfr_tcga))+" "+str(len(selected_tcga_samples_test))+" "+str(len(ood_tcga_samples)))
    
    # For TCGA dataset
if dataset == "GE":
    tcga_curr_round_labelled = combined_train_stmfr_tcga.filter(like='TCGA', axis=0)

    # For hungarian dataset
elif dataset == "HD":
    tcga_curr_round_labelled = combined_train_stmfr_tcga.filter(like='Hungary', axis=0)

# tcga_curr_round_combined = pd.concat([tcga_curr_round_labelled, selected_tcga_samples_test])
logger.info("Labels Distribution in current round:")           
logger.info(tcga_curr_round_labelled.groupby(['numeric_label'])['numeric_label'].count())

curr_round_added_samples_count = len(combined_train_stmfr_tcga) - len(df)    
logger.info("Round 1 ends. Added %d samples.", curr_round_added_samples_count)

end = timer()

exec_time = end-start
# print(exec_time)

logger.info("Forward Pass: Round 1 took %s time.", exec_time)


# %%
tcga_curr_round_labelled.groupby(['numeric_label'])['numeric_label'].count()

# %% [markdown]
# 2m 57.8s, 3 models, 100 PCA
# For 3 models, test accuracy avg and stddev:  0.9555555555555556 0.01737843488291437
# Length of combined_train_stmfr_tcga, selected_tcga_test, ood_tcga:  7009 929 6426
# 
# w/o PCA, 3 models 

# %%
# For MLP and other models which take more time, stopping criterion can be 5%
if dataset == "HD":
    stop_at = 0.02
elif dataset == "GE":
    stop_at = 0.05


start = timer()
round = 2

if curr_round_added_samples_count == len(tcga):
    # If all the samples in the derived datasets are labelled then no need for further rounds
    pass
else:

    while True:
        try:
            if curr_round_added_samples_count >= stop_at*len(tcga):
                # logger.debug/info/warning/error
                logger.info("\nRound "+str(round)+" starts:\n")   
                logger.info("Length of combined_train_stmfr_tcga, selected_tcga_test, ood_tcga: "+
                    str(len(combined_train_stmfr_tcga))+" "+str(len(selected_tcga_samples_test))+" "+str(len(ood_tcga_samples)))
                
                prev_round_combined_samples_count = len(combined_train_stmfr_tcga)

                combined_train_stmfr_tcga, selected_tcga_samples_test, ood_tcga_samples = prediction_agreement_models(
                    combined_train_stmfr_tcga, ood_tcga_samples, selected_tcga_samples_test, n_models=n_models, frac_agree=frac_agree)
                
                logger.info("\nRound "+str(round)+" Details:\n")   
                logger.info("Length of combined_train_stmfr_tcga, selected_tcga_test, ood_tcga: "+
                    str(len(combined_train_stmfr_tcga))+" "+str(len(selected_tcga_samples_test))+" "+str(len(ood_tcga_samples)))

                curr_round_combined_samples_count = len(combined_train_stmfr_tcga)

                curr_round_added_samples_count = curr_round_combined_samples_count - prev_round_combined_samples_count

                # For TCGA dataset
                if dataset == "GE":
                    tcga_curr_round_labelled = combined_train_stmfr_tcga.filter(like='TCGA', axis=0)

                # For hungarian dataset
                elif dataset == "HD":
                    tcga_curr_round_labelled = combined_train_stmfr_tcga.filter(like='Hungary', axis=0)
                

                # tcga_curr_round_combined = pd.concat([tcga_curr_round_labelled, selected_tcga_samples_test])    
                # logger.info("Samples labelled in the current round %s", tcga_curr_round_labelled.shape)    
                if(curr_round_added_samples_count!=0):
                    logger.info("Labels Distribution in current round:")   
                    logger.info(tcga_curr_round_labelled.groupby(['numeric_label'])['numeric_label'].count())
                    
                    logger.info("\nRound "+str(round)+" ends.\n\n") 
                else:
                    #Handle the exception
                    logger.warning("No samples were added in the current round.")
                    logger.info("\nRound "+str(round)+" ends.\n\n") 
                    break
                round = round + 1
                # if len(selected_tcga_samples_test*4) <= 100:
                #     logger.info("No of selected TCGA samples for next round is less than 100")
                    
                #     break
            else:
                logger.info("STOPPING: No of samples added in the current round, %d, is less than %.2f fraction of the original.", curr_round_added_samples_count, stop_at)
                break
        except Exception as e:
            #Handle the exception
            logger.warning("An exception occurred:")
            logger.warning(e) #
            break
        
logger.info("Forward experiment ended in %d rounds!!!\n", round-1)    

# For hungarian dataset
tcga_labelled_forward_expr = combined_train_stmfr_tcga.filter(like='Hungary', axis=0)
logger.info("Total samples for which labelling is done in forward pass %s", tcga_labelled_forward_expr.shape)

end = timer()

exec_time = end-start
# print(exec_time)
logger.info("Forward Pass: Round 2 till %d took %s time.", round-1, exec_time)

# %%
len(combined_train_stmfr_tcga), len(selected_tcga_samples_test), len(ood_tcga_samples)

# %%
tcga_curr_round_labelled.groupby(['numeric_label'])['numeric_label'].count()

# %% [markdown]
# # 1.2 Train-test on derived dataset 

# %%
# Forward experiment, in each round, in labeled TCGA, plot the distribution of labels

# Saving the labelled TCGA took ~ 1m 32.4s 

# For TCGA dataset
if dataset == "GE":
    tcga_labelled_forward_expr = combined_train_stmfr_tcga.filter(like='TCGA', axis=0)

# For hungarian dataset
elif dataset == "HD":
    tcga_labelled_forward_expr = combined_train_stmfr_tcga.filter(like='Hungary', axis=0)

# tcga_curr_round_combined = pd.concat([tcga_curr_round_labelled, selected_tcga_samples_test])
    
# fname = model_name+"_TCGA_labelled_Dec23.csv"    
# tcga_curr_round_combined.to_csv(fname)

# %%
# str((tcga_labelled_forward_expr.shape)[0])

# %%
logger.info("1.2: Train-test on derived %s (%s): ", derived_dataset, tcga_labelled_forward_expr.shape)
avg_acc = check_model_stmfr(tcga_labelled_forward_expr, "Derived_dataset")

add_to_final_results = {"s_no": '1.2', 'description': 'Train-test on derived '+derived_dataset+" "+str((tcga_labelled_forward_expr.shape)[0]), 'value': avg_acc}
# final_results = final_results.append(add_to_final_results, ignore_index=True)

final_results.loc[len(final_results)] = add_to_final_results

# %% [markdown]
# # 1.2a: label %age match with the origial labels for base dataset 

# %%
tcga

# %%
tcga_labelled_forward_expr

# How to compare these with acutal labels
# Preserve sample ids if there
# I think they are there- need to preserve them so that these derived labels can be compared with the original labels 

# %%
orig_tcga

# %% [markdown]
# # 1.2a label match with the origial labels for derived dataset 

# %%

if dataset == "HD":
    
    # match_labels['match'] = match_labels[match_labels['num'] == match_labels['numeric_label']]
    match_labels = pd.merge(orig_tcga, tcga_labelled_forward_expr, left_index=True, right_index=True)[['numeric_label_x','numeric_label_y']]

    match_labels['match'] = np.where(match_labels['numeric_label_x'] == match_labels['numeric_label_y'], 1, 0)
    count_match = match_labels.groupby(['match'])['match'].value_counts()[1].item()
    perc_match = (count_match / len(match_labels)) 

    if debug:
        print("\n\n******************\n\n")
        print(count_match)
        print(perc_match)
        print("\n\n******************\n\n")
    #perc_match

    # df['que'] = np.where((df['one'] >= df['two']) & (df['one'] <= df['three'])
    #                      , df['one'], np.nan)

    logger.info("1.2a: label %%age match with the origial labels for %s dataset", derived_dataset)
    logger.info("For forward experiment, %% of match for derived %s and original %s dataset: %f!!!\n", derived_dataset, derived_dataset, perc_match)    

    add_to_final_results = {"s_no": '1.2a', 'description': 'label match with the origial labels for '+derived_dataset, 'value': perc_match}

elif dataset == "GE":

    add_to_final_results = {"s_no": '1.2a', 'description': 'label match with the origial labels for '+derived_dataset, 'value': np.nan}

final_results.loc[len(final_results)] = add_to_final_results

# %%
a=pd.DataFrame(columns=['match'],data=[1, 2,1,2,1,3,4,1,2])

a.groupby(['match'])['match'].value_counts()[1]

# %% [markdown]
# # Reverse Experiment
# 
# (1) Use derived TCGA to label STMFR
# 
# (2) Using derived hungarian dataset to label cleveland dataset 

# %%
# Took 18.9s to read from local 

# tcga_labeled_df = pd.read_csv("/Users/bhavesh/Documents/AU/PhD/model_agreement_expr/gitrepo/model_agreement/XGB_TCGA_labelled_Dec23.csv")

tcga_labeled_df = tcga_labelled_forward_expr

# %%
logger.info("\n\nSetting up Reverse pass:\n")
logger.info("Labelled %s: shape %s", derived_dataset, tcga_labeled_df.shape)
# logger.info(tcga_labeled_df.shape)
# logger.info("")

# %%
tcga_labeled_df

# %%
# tcga_labeled_df.set_index(tcga_labeled_df['Unnamed: 0'], inplace=True)
# tcga_labeled_df.drop('Unnamed: 0', axis=1, inplace=True)

# %%
stmfr_unlabelled_df = df.iloc[:, :-1]

# %%
stmfr_unlabelled_df.shape

# %%
logger.info("\n Reverse experiment: \n\n")
logger.info("Using derived %s to label %s", derived_dataset, base_dataset)

logger.info("\n Round 1 begins")
combined_train_stmfr_tcga, selected_stmfr_test, ood_stmfr = prediction_agreement_models(tcga_labeled_df, stmfr_unlabelled_df, "", n_models=n_models, frac_agree=frac_agree)

logger.info("Length of combined_train_stmfr_tcga, selected_tcga_test, ood_tcga: "+
                str(len(combined_train_stmfr_tcga))+" "+str(len(selected_stmfr_test))+" "+str(len(ood_stmfr)))


# For STMFR dataset 
if dataset == "GE":
    stmfr_curr_round_labelled = combined_train_stmfr_tcga.filter(regex='^((?!TCGA).)*$', axis=0)

# For cleveland dataset
elif dataset == "HD":
    stmfr_curr_round_labelled = combined_train_stmfr_tcga.filter(like='Cleveland', axis=0)

# stmfr_curr_round_combined = pd.concat([stmfr_curr_round_labelled, selected_stmfr_test])        

logger.info(stmfr_curr_round_labelled.groupby(['numeric_label'])['numeric_label'].count())

curr_round_added_samples_count = len(combined_train_stmfr_tcga) - len(tcga_labeled_df)    

logger.info("\n Round 1 ends. Added %d samples", curr_round_added_samples_count)

# %%
round = 2

if dataset == "HD":
    stop_at = 0.02
elif dataset == "GE":
    stop_at = 0.05


if curr_round_added_samples_count==len(stmfr_unlabelled_df):
    pass
else:

    while True:
        try:
            if curr_round_added_samples_count >= stop_at*len(stmfr_unlabelled_df):
                logger.info("\nRound "+str(round)+" starts:")   
                logger.info("\nLength of combined_train_stmfr_tcga, selected_stmfr_test, ood_stmfr: "+
                    str(len(combined_train_stmfr_tcga))+" "+str(len(selected_stmfr_test))+" "+str(len(ood_stmfr)))
                
                prev_round_combined_samples_count = len(combined_train_stmfr_tcga)

                #Function call 
                combined_train_stmfr_tcga, selected_stmfr_test, ood_stmfr = prediction_agreement_models(
                    combined_train_stmfr_tcga, ood_stmfr, selected_stmfr_test, n_models=n_models, frac_agree=frac_agree)
                
                logger.info("\nRound "+str(round)+" details:\n\n")   
                logger.info("\nLength of combined_train_stmfr_tcga, selected_stmfr_test, ood_stmfr: "+
                    str(len(combined_train_stmfr_tcga))+" "+str(len(selected_stmfr_test))+" "+str(len(ood_stmfr)))
                
                curr_round_combined_samples_count = len(combined_train_stmfr_tcga)

                curr_round_added_samples_count = curr_round_combined_samples_count - prev_round_combined_samples_count


                # For STMFR dataset 
                if dataset == "GE":
                    stmfr_curr_round_labelled = combined_train_stmfr_tcga.filter(regex='^((?!TCGA).)*$', axis=0)

                # For cleveland dataset
                elif dataset == "HD":
                    stmfr_curr_round_labelled = combined_train_stmfr_tcga.filter(like='Cleveland', axis=0)

                # stmfr_curr_round_combined = pd.concat([stmfr_curr_round_labelled, selected_stmfr_test])     
                if(curr_round_added_samples_count!=0):    
                    logger.info("Labels Distribution in current round:")   
                    logger.info(stmfr_curr_round_labelled.groupby(['numeric_label'])['numeric_label'].count())
                    logger.info("Round "+str(round)+" ends.")
                else:
                    logger("No new samples are labeled in the current round %d", round)    
                    logger.info("\nRound "+str(round)+" ends.\n\n")     
                    break
                round = round + 1
                # if len(selected_stmfr_test)*4 <= 100:
                #     logger.info("No of selected STMFR samples for next round is less than 100")
                #     break
            else:
                logger.info("STOPPING: No of samples added in the current round, %d, is less than %.2f fraction of the original.",curr_round_added_samples_count, stop_at)
                break
        except Exception as e:
            #Handle the exception
            logger.warning("An exception occurred: ")
            logger.warning(e) 
            break
        
logger.info("Reverse experiment ended in %d rounds!!!\n", round-1)    
stmfr_curr_round_labelled = combined_train_stmfr_tcga.filter(like='Cleveland', axis=0)
logger.info("Reverse Pass: Total number of samples for which labelling is done: %s", stmfr_curr_round_labelled.shape)

# %%
print("Length of combined_train_stmfr_tcga, selected_stmfr_test, ood_stmfr: ",
                    len(combined_train_stmfr_tcga), len(selected_stmfr_test), len(ood_stmfr))

# %% [markdown]
# 
# combined_train_stmfr_tcga: Total lengh: 10349
#                         
#                          like TCGA: 8003
#                          like GSM: 2013
#                          Total: 10016
# 
#                          Difference of 333: These samples are in stemformatics dataset 

# %%
combined_train_stmfr_tcga

# %%
# combined_train_stmfr_tcga.filter(regex='^((?!TCGA).)*$', axis=0)

stmfr_curr_round_labelled.shape

# %%
# For STMFR
if dataset == "GE":
    stmfr_curr_round_labelled = combined_train_stmfr_tcga.filter(regex='^((?!TCGA).)*$', axis=0)

# For cleveland
elif dataset == "HD":
    stmfr_curr_round_labelled = combined_train_stmfr_tcga.filter(like='Cleveland', axis=0)

# stmfr_curr_round_combined = pd.concat([stmfr_curr_round_labelled, selected_stmfr_test])        
print(stmfr_curr_round_labelled.groupby(['numeric_label'])['numeric_label'].count())

# %%
# stmfr_curr_round_labelled[:,-2:]
combined_train_stmfr_tcga.shape

# %%
# stmfr_curr_round_labelled.to_csv("labelled_smtfr_Dec_19.csv")

# %%
# type(stmfr_curr_round_labelled.shape)

logger.info("Labelled Derived %s  %s", base_dataset, stmfr_curr_round_labelled.shape)
# logger.info(stmfr_curr_round_labelled.shape)


# %% [markdown]
# # 1.8a Match the labels of derived base dataset with the real labels

# %%
stmfr_curr_round_labelled

# %%
df

# %%
reverse_match_labels = pd.merge(stmfr_curr_round_labelled, df, left_index=True, right_index=True)[['numeric_label_x','numeric_label_y']]
reverse_match_labels

# %%
reverse_match_labels['match'] = np.where(reverse_match_labels['numeric_label_x'] == reverse_match_labels['numeric_label_y'], 1, 0)
count_match = reverse_match_labels.groupby(['match'])['match'].value_counts()[1].item()
perc_match = (count_match / len(reverse_match_labels)) 
# perc_match

if debug:
        print("\n\n******************\n\n")
        print(count_match)
        print(perc_match)
        print("\n\n******************\n\n")

logger.info("\n\n1.8a: Reverse derived label %%age match with original labels for %s", base_dataset)
logger.info("For reverse experiment, %% match of labels between derived %s and original %s dataset: %f!!!\n", base_dataset, base_dataset, perc_match)    

add_to_final_results = {"s_no": '1.8a', 'description': 'Reverse derived label match with original labels for '+base_dataset, 'value': perc_match}
# final_results = final_results.append(add_to_final_results, ignore_index=True)

final_results.loc[len(final_results)] = add_to_final_results

# %% [markdown]
# # Derived STMFR train-test expr

# %% [markdown]
# # 1.8a Train-test on reverse derived labels for base dataset

# %%
# x = df.iloc[:, :-1]
# y = df['numeric_label']

def check_derived_stmfr(derived_stmfr_test_df):

    x = derived_stmfr_test_df.iloc[:, :-1]
    y = derived_stmfr_test_df['numeric_label']

    # scaler = StandardScaler()

    # x_rescaled = pd.DataFrame(scaler.fit_transform(x), index=x.index, columns=x.columns)

    acc_list = []
    b_acc_list = []
    f1_list = []
    prec_list = []
    rec_list = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(x, y):
        
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # bst = XGBClassifier(n_estimators=2, max_depth=20, learning_rate=1, num_class=4, objective='multi:softmax')
        # model = XGBClassifier(num_class=4, objective='multi:softmax', max_depth=4, learning_rate=0.117, subsample=0.82)

        model = get_model(model_name)
        # model = LogisticRegression(n_jobs=-1, max_iter=250)
        

        # pca = PCA(n_components=0.99)
        # X_train_pca = pca.fit_transform(X_train)
        # X_test_pca = pca.transform(X_test)

        # fit model
        model.fit(X_train, y_train)
        # make predictions
        predictions = model.predict(X_test)

        # dtrain = xgb.DMatrix(X_train_pca, label=y_train)
        # dtrain.save_binary('train.buffer')

        # param = {'max_depth': 20, 'eta': 1, 'objective': 'multi:softmax'}
        # param['nthread'] = 4
        # param['eval_metric'] = 'auc'
        # param['num_class'] = 4

        # num_round = 10
        # bst = xgb.train(param, dtrain, num_round)

        # dtest = xgb.DMatrix(X_test_pca)
        # predictions = bst.predict(dtest)

        target_names = ['hESC', 'hMSC', 'hUSC', 'iPSC']
        # print(classification_report(y_test, predictions, target_names=target_names))

        # model_acc = accuracy_score(y_test, predictions, normalize=True)
        # model_acc

        accuracy = accuracy_score(y_test, predictions)
        acc_list.append(accuracy)

        b_accuracy = balanced_accuracy_score(y_test, predictions)
        b_acc_list.append(b_accuracy)

        f1score = f1_score(y_test, predictions, average='weighted')
        f1_list.append(f1score)

        precision = precision_score(y_test, predictions, average='weighted')
        prec_list.append(precision)

        recall = recall_score(y_test, predictions, average='weighted')
        rec_list.append(recall)
        

    avg_acc = np.average(acc_list)
    # logging.info("accuracy: %f", np.average(acc_list))
    print("accuracy", avg_acc)
    
    # # logging.info(accuracy)
    # # m.append(accuracy)
    # logging.info("b_accuracy: %f", np.average(b_accuracy))
    # # logging.info(b_accuracy)
    # logging.info("f1score: %f", np.average(f1score))
    # # logging.info(f1score)
    # logging.info("precision: %f", np.average(precision))
    # # logging.info(precision)
    # logging.info("recall: %f", np.average(recall))
    # # logging.info(recall)

    # logging.info("\n")
    
    logging.info("\n For reverse experiment on Labelled %s Average accuracy: %f", base_dataset, avg_acc)
    return avg_acc
    # logging.info(np.average(acc_list))

# %%
str((stmfr_curr_round_labelled.shape)[0])

# %%
final_results

# %%
logger.info("1.8: Train-test on reverse derived %s dataset", base_dataset)
avg_acc = check_derived_stmfr(stmfr_curr_round_labelled)


add_to_final_results = {"s_no": '1.8', 'description': 'Train-test on reverse derived '+base_dataset+" "+str((stmfr_curr_round_labelled.shape)[0]), 'value': avg_acc}
# final_results = final_results.append(add_to_final_results, ignore_index=True)

final_results.loc[len(final_results)] = add_to_final_results

# %% [markdown]
# # Get classification  probabilities for the samples for which labels did not match between base dataset and derived dataset
# 
# find for which samples in STMFR (df) and derived-STMFR (stmfr_curr_round_labelled) the labels match and for which samples labels do not match.
# For samples, labels not matching, check their LR probs calculated for STMFR. If the probs are similar for top-2 then these samples were tough to classify even in the original dataset. 
# Thus derived dataset also has labelling different from orignal. 
# 
# Verify.

# %%

def get_probs(df):

    stmfr_test_probs_all = pd.DataFrame()

    x = df.iloc[:, :-1]
    y = df['numeric_label']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    i = 1
    for train_index, test_index in skf.split(x, y):
    
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # derived_stmfr_train_df, derived_stmfr_test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)


        # X_train = derived_stmfr_train_df.iloc[:, :-1]
        # y_train = derived_stmfr_train_df['numeric_label']

        # X_test = derived_stmfr_test_df.iloc[:, :-1]
        # y_test = derived_stmfr_test_df['numeric_label']

        # model = LogisticRegression(n_jobs=-1, max_iter=250)
        model = get_model(model_name)

        # fit model
        model.fit(X_train, y_train)
        # make predictions
        predictions = model.predict(X_test)

        probs_test = model.predict_proba(X_test)
        probs_train = model.predict_proba(X_train)

        target_names = ['hESC', 'hMSC', 'hUSC', 'iPSC']
        # print(classification_report(y_test, predictions, target_names=target_names))

        accuracy = accuracy_score(y_test, predictions)
        # acc_list.append(accuracy)

        b_accuracy = balanced_accuracy_score(y_test, predictions)
        # b_acc_list.append(b_accuracy)

        f1score = f1_score(y_test, predictions, average='weighted')
        # f1_list.append(f1score)

        precision = precision_score(y_test, predictions, average='weighted')
        # prec_list.append(precision)

        recall = recall_score(y_test, predictions, average='weighted')
        # rec_list.append(recall)

        print(accuracy, end='\t')
        print(b_accuracy, end='\t')
        print(f1score, end='\t')
        print(precision, end='\t')
        print(recall)

        
        stmfr_test_probs = pd.DataFrame(probs_test)
        stmfr_test_probs.index = X_test.index
        # derived_stmfr_test_probs = pd.concat([X_test, derived_test_probs.reindex(X_test.index)],axis=1)
        stmfr_test_probs = pd.concat([X_test, stmfr_test_probs], axis=1)

        # stmfr_train_probs = pd.DataFrame(probs_train)
        # stmfr_train_probs.index = X_train.index
        # derived_stmfr_test_probs = pd.concat([X_test, derived_test_probs.reindex(X_test.index)],axis=1)
        # stmfr_train_probs = pd.concat([X_train, stmfr_train_probs], axis=1)
        fname = "stmfr_test_probs_"+str(i)+".csv"
        
        # print(fname, i)

        # For STMFR and TCGA datasets there are four classes
        # stmfr_test_probs_all = pd.concat([stmfr_test_probs_all, pd.DataFrame(stmfr_test_probs.iloc[:,-4:])])
        
        # For cleveland and hungary datasets there are two classes
        stmfr_test_probs_all = pd.concat([stmfr_test_probs_all, pd.DataFrame(stmfr_test_probs.iloc[:,-2:])])

        # stmfr_test_probs.iloc[:,-4:].to_csv(fname)
        i = i + 1
        # stmfr_train_probs.iloc[:,-4:].to_csv("stmfr_train_probs.csv")

    return stmfr_test_probs_all
        

# %%
# stmfr_test_probs_all = get_probs(df)
# stmfr_test_probs_all

# %%
# stmfr_test_probs[['0','0.1','1','2','3']].to_csv("stmfr_test_probs.csv")
# stmfr_test_probs.iloc[:,-4:].to_csv("stmfr_test_probs.csv")
# stmfr_train_probs.iloc[:,-4:].to_csv("stmfr_train_probs.csv")


# %%
# # derived_test_probs.index = X_test.index

# # X_test.index

# stmfr_test_probs = pd.read_csv("../../Rounds/stmfr_test_probs.csv")
# derived_stmfr_test_probs = pd.read_csv("../../Rounds/derived_stmfr_test_probs.csv")

# %%
# df[['numeric_label']].to_csv("STMFR_labels.csv")

# %%
# fname = model_name+"_Derived_STMFR_labels.csv"
# stmfr_curr_round_labelled[['numeric_label']].to_csv(fname)

# %%
# derived_stmfr_test_probs[['Unnamed: 0', '0','1','2','3']].to_csv("derived_stmfr_test_probs.csv")

# stmfr_test_probs[['0','0.1','1','2','3']].to_csv("stmfr_test_probs.csv")

# stmfr_test_probs

# %% [markdown]
# # After Reverse pass: Labels which did not match for base dataset and its derived labels

# %%
a = pd.DataFrame(stmfr_curr_round_labelled['numeric_label'])
b = pd.DataFrame(df['numeric_label'])

c = pd.merge(a, b, left_index=True, right_index=True)
c.columns = ['derived', 'original']
# print (c)

# %%
# fname = model_name+"_compare_CL_labels.csv"
# c.to_csv(fname)

# %%
not_matched_stmfr = c[c['derived'] != c['original']]

# %%
# print("For %d samples in derived dataset (out of total: %d), predicted labels did not match with that of original dataset: %.3f%%" % (len(not_matched_stmfr), len(stmfr_curr_round_labelled), (len(not_matched_stmfr)/len(stmfr_curr_round_labelled)*100)))

logger.info("For %d samples in derived dataset (out of total: %d), predicted labels did not match with that of original dataset: %.3f%%" % (len(not_matched_stmfr), len(stmfr_curr_round_labelled), (len(not_matched_stmfr)/len(stmfr_curr_round_labelled) *100)))
# 

# %%
# fname = model_name+"_probs_mismatch_CL.csv"
# pd.merge(not_matched_stmfr, stmfr_test_probs_all, left_index=True, right_index=True).to_csv(fname)

# logger.info("Check %s for the probabilities of the samples that did not match", fname)

# %% [markdown]
# # 1.3-1.7

# %% [markdown]
# Reading combined file takes around 28 secs for STMFR and TCGA

# %%
# combined_stmfr_tcga_labeled = pd.read_csv('/Users/bhavesh/Library/CloudStorage/GoogleDrive-bhavesh.neekhra_phd18@ashoka.edu.in/My Drive/CR_Data/Results/MLCB/combined_stmfr_tcga_labelled.csv')

# %% [markdown]
# 

# %%
combined_train_stmfr_tcga

# %%
combined_stmfr_tcga_labeled_df = combined_train_stmfr_tcga

# %% [markdown]
# If reading from a csv of the combined dataset of labels for base and derived dataset 

# %%
# tcga_labeled_df = combined_stmfr_tcga_labeled.loc[combined_stmfr_tcga_labeled['Unnamed: 0'].str.contains('TCGA')==True]
# tcga_labeled_df.set_index(tcga_labeled_df['Unnamed: 0'], inplace=True)
# tcga_labeled_df.drop('Unnamed: 0', axis=1, inplace=True)


# %%
# combined_stmfr_tcga_labeled_df.set_index(combined_stmfr_tcga_labeled_df['Unnamed: 0'], inplace=True)
# combined_stmfr_tcga_labeled_df.drop('Unnamed: 0', axis=1, inplace=True)

# %%
combined_stmfr_tcga_labeled_df

# %%
combined_stmfr_tcga_labeled_df.columns

# %%
tcga

# %%
tcga_labeled_df

# %% [markdown]
# Train-test on various combination on STMFR and TCGA 

# %%
# combined_stmfr_tcga_labeled_df
df.shape, tcga_labeled_df.shape

# %%

# Train on STMFR and test on TCGA



# Train-test on combined STMFR+TCGA


# Train on combined STMFR (50%)+TCGA and test on 50% STMFR


# Train on combined STMFR+TCGA(50%) and test on 50% TCGA



# X_train = train_df.iloc[:, :-1]
# y_train = train_df['numeric_label']

# X_test = tcga_test_df.iloc[:, :-1]
# y_test = tcga_test_df['numeric_label']

def check_model_comb_base_derived(base_df, derived_df):

    X_train = base_df.iloc[:, :-1]
    y_train = base_df['numeric_label']

    X_test = derived_df.iloc[:, :-1]
    y_test = derived_df['numeric_label']

    # model = LogisticRegression(n_jobs=-1, max_iter=250)
    model = get_model(model_name)
    # fit model
    model.fit(X_train, y_train)
    # make predictions
    predictions = model.predict(X_test)

    # target_names = ['hESC', 'hMSC', 'hUSC', 'iPSC']
    # print(classification_report(y_test, predictions, target_names=target_names))

    accuracy = accuracy_score(y_test, predictions)
    # acc_list.append(accuracy)

    b_accuracy = balanced_accuracy_score(y_test, predictions)
    # b_acc_list.append(b_accuracy)

    f1score = f1_score(y_test, predictions, average='weighted')
    # f1_list.append(f1score)

    precision = precision_score(y_test, predictions, average='weighted')
    # prec_list.append(precision)

    recall = recall_score(y_test, predictions, average='weighted')
    # rec_list.append(recall)

    logger.info("Accuracy: %f", accuracy)
    print(accuracy, end='\t')
    print(b_accuracy, end='\t')
    print(f1score, end='\t')
    print(precision, end='\t')
    print(recall)

    return accuracy


# %% [markdown]
# # 1.1b Train on original base dataset and test on original derived dataset 
# 
# (if labels are availalbe for derived dataset)

# %%
# df
# orig_tcga
# 1.1b: Train on original base dataset and test on original derived dataset, if labels are available
if dataset == "HD":
    logging.info("1.1b: Train on original %s (%s), test on original %s (%s)", base_dataset, df.shape, derived_dataset, orig_tcga.shape)
    acc = check_model_comb_base_derived(df, orig_tcga)

    add_to_final_results = {"s_no": '1.1b', 'description': 'Train on original '+base_dataset+' '+"test on original "+derived_dataset, 'value': acc}
    
elif dataset == "GE":
    add_to_final_results = {"s_no": '1.1b', 'description': 'Train on original '+base_dataset+' '+"test on original "+derived_dataset, 'value': np.nan}

# final_results = final_results.append(add_to_final_results, ignore_index=True)

final_results.loc[len(final_results)] = add_to_final_results

# %%
tcga_labelled_forward_expr.shape[0]

# %%
# 'Train on original '+base_dataset+' '+str(df.shape[0])+" test on derived "+derived_dataset+' '+str(tcga_labelled_forward_expr.shape[0])

# %%
# tcga_labeled_df
# 1.3: Train on complete base dataset, and test on derived labelled dataset 

logging.info("\n\n1.3: Train on original %s (%s), test on derived %s (%s)", base_dataset, df.shape, derived_dataset, tcga_labeled_df.shape)
acc = check_model_comb_base_derived(df, tcga_labeled_df)


add_to_final_results = {"s_no": '1.3', 'description': 'Train on original '+base_dataset+" test on derived "+derived_dataset, 'value': acc}
# final_results = final_results.append(add_to_final_results, ignore_index=True)

final_results.loc[len(final_results)] = add_to_final_results

# %%
# 1.4:  Train on derived labelled dataset and test on original base dataset 
logging.info("\n\n1.4: Train on derived %s (%s), test on %s (%s)", derived_dataset, tcga_labeled_df.shape, base_dataset, df.shape)
acc = check_model_comb_base_derived(tcga_labeled_df, df)

add_to_final_results = {"s_no": '1.4', 'description': 'Train on derived '+derived_dataset+" test on "+base_dataset, 'value': acc}
# final_results = final_results.append(add_to_final_results, ignore_index=True)

final_results.loc[len(final_results)] = add_to_final_results

# %%
# 1.5 Train-test on combined base_derived

logger.info("1.5: Train and test on combined %s and %s", base_dataset, derived_dataset)
train, test = train_test_split(pd.concat([df, tcga_labeled_df]), test_size=0.2, random_state=42, shuffle=True) 
# X_train, y_train = train.iloc[:, :-1], train['numeric_label']
# X_test, y_test = test.iloc[:, :-1], test['numeric_label']
avg_acc = check_model_stmfr(pd.concat([df, tcga_labeled_df]), base_dataset+"+"+derived_dataset)


add_to_final_results = {"s_no": '1.5', 'description': 'Train and test on combined '+base_dataset+' '+derived_dataset, 'value': avg_acc}
# final_results = final_results.append(add_to_final_results, ignore_index=True)

final_results.loc[len(final_results)] = add_to_final_results

# %%
# 1.6 Train-test on combined base (50%) + derived, Test on base (remaining 50%)

logger.info("\n1.6: Train on combined %s (50%%) + %s and test on 50%% %s:\n", base_dataset, derived_dataset, base_dataset)

stmfr_train_df, stmfr_test_df = train_test_split(df, test_size=0.5, random_state=42, shuffle=True)

train_df = pd.concat([tcga_labeled_df, stmfr_train_df], ignore_index=True)

acc = check_model_comb_base_derived(train_df, stmfr_test_df)


add_to_final_results = {"s_no": '1.6', 'description': 'Train on combined '+base_dataset+"(50%) "+derived_dataset+" and test on 50% "+base_dataset, 'value': acc}
# final_results = final_results.append(add_to_final_results, ignore_index=True)

final_results.loc[len(final_results)] = add_to_final_results


# %%
# 1.7 Train-test on combined base + derived (50%), Test on derived (remaining 50%)

logger.info("\n1.7: Train on combined %s + %s (50%%) and test on 50%% %s:\n", base_dataset, derived_dataset, derived_dataset)

tcga_train_df, tcga_test_df = train_test_split(tcga_labeled_df, test_size=0.5, random_state=42, shuffle=True)

train_df = pd.concat([df, tcga_train_df], ignore_index=True)

acc = check_model_comb_base_derived(train_df, tcga_test_df)

add_to_final_results = {"s_no": '1.7', 'description': 'Train on combined '+base_dataset+" "+derived_dataset+"(50%) and test on 50% "+derived_dataset, 'value': acc}
# final_results = final_results.append(add_to_final_results, ignore_index=True)

final_results.loc[len(final_results)] = add_to_final_results

# %% [markdown]
# # Save the final results dataframe

# %%
final_results.description


# %%
final_results.sort_values(by=['s_no'])

# %%
# tracker.stop()
#!mv "./emissions.csv" $model_name+"emissions.csv"

# %%
logger.info("Ending the timer now!")
prog_end = timer()

total_time_code_exec = prog_end - prog_start
add_to_final_results = {"s_no": '1.9', 'description': 'Time taken for code execution', 'value': total_time_code_exec}
# final_results = final_results.append(add_to_final_results, ignore_index=True)

final_results.loc[len(final_results)] = add_to_final_results

# %%
final_results.sort_values(by=['s_no']).to_csv(final_results_save_path+"/"+fname_1+"_final_results.csv", index=False)

# %%
logger.info("Time to execute the code for the current settings: %.3f", total_time_code_exec)

# %%
print("\n\n**************\n\n")
print("Find details in the log file: ",log_file)
print("\n\n**************\n\n")

# %% [markdown]
# # End of code


