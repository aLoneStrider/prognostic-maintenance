#!/usr/bin/env python
# coding: utf-8

import sys
import warnings
warnings.filterwarnings("ignore")
import os.path
from math import *
import numpy as np
np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
palette = sns.color_palette("husl", n_colors = 4, desat = .9)
sns.set_palette(palette)
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.cross_decomposition
import sklearn.neighbors
import sklearn.model_selection

# data and plots folders
data_path = "data"
if not os.path.exists(data_path):
    os.mkdir(data_path)
plot_path = "plots"
if not os.path.exists(plot_path):
    os.mkdir(plot_path) 

# reading data from original txt files
def read_set(data_path, set_number, settype):
    name = os.path.join(data_path, settype + "_" + str(set_number) + ".txt")
    data = pd.read_csv(name, delim_whitespace = True, header = None)
    new_cols = ["id", "cycle", "setting1", "setting2", "setting3"] + ["s" + str(x) for x in range(1, 22)]
    data.columns = new_cols
    return data

# finding col datatype 
def find_col_types(df):
    id_columns = ["id", "cycle"]
    df_columns = df.columns.difference(id_columns)
    categorical_columns = [x for x in df_columns if( df[x].dtype == "int" or df[x].dtype == "int64")]
    scalable_columns = [x for x in df_columns if x not in categorical_columns]
    return categorical_columns, scalable_columns

# setting RUL in training set
def calculate_train_RUL(df):
    for part_id in df["id"].unique():
        max_cycle = df.loc[df["id"] == part_id, "cycle"].max()
        df.loc[df["id"] == part_id,"RUL"] = max_cycle - df.loc[df["id"] == part_id, "cycle"]
    return df

# setting RUL in test set
def calculate_test_RUL(df, label_df):
    for part_id in df["id"].unique():
        max_cycle = df.loc[df["id"] == part_id, "cycle"].max()
        label_RUL = label_df.loc[label_df["id"] == part_id, "RUL"].values[0]
        df.loc[df["id"] == part_id,"RUL"] = max_cycle + label_RUL + (max_cycle - df.loc[df["id"] == part_id, "cycle"])
    return df

# plotting distribution of all cols
def plot_all_measurements(df, plot_path = "plots", plot_name = "raw_sequences.png"):
    cols = df.columns[2:26]
    fig, axs = plt.subplots(len(cols), figsize = (18, 12))
    axs = axs.flatten()
    for i, col in enumerate(cols):
        axs[i].plot(df["cycle"], df[col], "--")
        h = axs[i].set_ylabel("              " + col)
        h.set_rotation(0)
        axs[i].yaxis.set_label_position("right")
    plt.savefig(os.path.join(plot_path, plot_name))
    plt.show()
        
# plotting correlatinal figs
def plot_correlations(df, drop_cols = [], title = "", plot_path = "plots", plot_name = "correlation.png"):
    tmp_df = df.drop(drop_cols, 1)
    corr = tmp_df.corr()
    plt.figure(figsize = (8, 8))
    g = sns.heatmap(corr)
    g.set_xticklabels(g.get_xticklabels(), rotation = 30, fontsize = 8)
    g.set_yticklabels(g.get_yticklabels(), rotation = 30, fontsize = 8)
    plt.title(title)
    plt.savefig(os.path.join(plot_path, plot_name))
    plt.show()

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# merging default training and test datasets
merge_data = True

# selecting the dataset (1 ~ 4)
dataset_index = 3
set_name = "FD00" + str(dataset_index)


# classify RUL into number of classes
n_classes = 4

# portion of data for each task
train_perc = 0.5
test_perc = 0.3
validate_perc = 0.2

# PCA components
n_comps = 21

# sampling training set
train = read_set(data_path, set_name, "train")
cat_train, scale_train = find_col_types(train)

# sampling test set
test = read_set(data_path, set_name, "test")
cat_test, scale_test = find_col_types(test)

# fetching labels
label = pd.read_csv(os.path.join(data_path, "RUL_" + set_name + ".txt"), header = None)
label.reset_index(level = [0], inplace = True)
label.columns = ["id", "RUL"]
label["id"] = label["id"] + 1  # index is 0-bound but part_ids are 1-bound

# adding labels
train = calculate_train_RUL(train)
test = calculate_test_RUL(test, label)

# Makeing all values float
train = train.astype("float64")
test = test.astype("float64")

# Add training and test set to form a full set which is then used for more RUL range coverage
# First, let's fix the problem of "id"s
last_train_id = train.id.max()
test.id = test["id"].map(lambda x: x + last_train_id)
# Then add them all to make the full dataset
full = pd.concat([train, test], ignore_index = True)
# Lastly, we split the full dataset into train, test and validation sets (by id)
full_ids = full.id.unique()

train_len = int(floor(len(full_ids) * train_perc))
test_len = int(floor(len(full_ids) * test_perc))

train_ids = np.random.choice(full_ids, size = train_len, replace = False)
test_validate_ids = np.setdiff1d(full_ids, train_ids)
test_ids = np.random.choice(test_validate_ids, size = test_len, replace = False)
validate_ids = np.setdiff1d(test_validate_ids, test_ids)

if merge_data:
    del train, test
    train = full.loc[full.id.isin(train_ids)]
    test = full.loc[full.id.isin(test_ids)]
    validate = full.loc[full.id.isin(validate_ids)]
    train.reset_index(drop = True, inplace = True)
    test.reset_index(drop = True, inplace = True)
    validate.reset_index(drop = True, inplace = True)
else:
    # split test set into test and validation sets
    msk = np.random.rand(len(test)) < 0.5
    validate = test[~msk].reset_index(drop = True)
    test = test[msk].reset_index(drop = True)
    
#Save intermediate file
if merge_data:
    train.to_csv(os.path.join(data_path, "train_" + set_name + "_merge.csv"), index = False)
    test.to_csv(os.path.join(data_path, "test_" + set_name + "_merge.csv"), index = False)
    full.to_csv(os.path.join(data_path, "full_" + set_name + "_merge.csv"), index = False)
    validate.to_csv(os.path.join(data_path, "validate_" + set_name + "_merge.csv"), index = False)
else:
    train.to_csv(os.path.join(data_path, "train_" + set_name + ".csv"), index = False)
    test.to_csv(os.path.join(data_path, "test_" + set_name + ".csv"), index = False)
    full.to_csv(os.path.join(data_path, "full_" + set_name + ".csv"), index = False)
    validate.to_csv(os.path.join(data_path, "validate_" + set_name + ".csv"), index = False)    

plt.figure()
sns.distplot(train.RUL, label = "train")
sns.distplot(test.RUL, label = "test")
sns.distplot(validate.RUL, label = "validate")
sns.distplot(full.RUL, label = "full")
plt.legend()
plt.title("unscaled RUL values")
if merge_data:
    plt.savefig(os.path.join(plot_path, set_name + "_RULs_unscaled_merge.png"))
else:
    plt.savefig(os.path.join(plot_path, set_name + "_RULs_unscaled.png"))
plt.show()

## plot correlations for unit i
i = np.random.choice(train.id)
df = train.loc[train.id==i].sort_values(by = "cycle")
plot_correlations(df, drop_cols = ["id", "cycle", "RUL"], 
                      title = "%s unit %s"%(set_name, i), plot_path = plot_path, plot_name = set_name)
plot_all_measurements(df, plot_path = "plots", plot_name = "raw_sequences.png")

## SCALE DATA
#normalize features (using MinMaxScaler)
train_scalables = train[train.columns.difference(["id", "cycle", "status"])].values
test_scalables = test[test.columns.difference(["id", "cycle", "status"])].values
validate_scalables = validate[validate.columns.difference(["id", "cycle", "status"])].values
full_scalables = full[full.columns.difference(["id", "cycle", "status"])].values

scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(train_scalables)

full_values = scaler.transform(full_scalables)
train_values = scaler.transform(train_scalables)
test_values = scaler.transform(test_scalables)
validate_values = scaler.transform(validate_scalables)
    
train[train.columns.difference(["id", "cycle"])] = train_values
test[test.columns.difference(["id", "cycle"])] = test_values
validate[validate.columns.difference(["id", "cycle"])] = validate_values
full[full.columns.difference(["id", "cycle"])] = full_values

# Bins for scaled RUL values corresponding to urgent, short, medium, long
RUL_min = full.RUL.min()
RUL_max = full.RUL.max()

bins = np.linspace(RUL_min - (RUL_max-RUL_min) / n_classes, RUL_max + RUL_min, n_classes + 1)
status_labels = ["RUL%d"%x for x in range(n_classes)]

# bin RUL values
full["status"] = pd.cut(full["RUL"], bins, labels = status_labels)
train["status"] = pd.cut(train["RUL"], bins, labels = status_labels)
test["status"] = pd.cut(test["RUL"], bins, labels = status_labels)
validate["status"] = pd.cut(validate["RUL"], bins, labels = status_labels)

# print(bins)
# print(status_labels)

## SEPARATE FEATURES AND LABELS
## Drop setting columns
full_X = full[full.columns.difference(["id", "cycle", "status", "setting1", "setting2", "setting3"])].values
full_y = full["status"].values

train_X = train[train.columns.difference(["id", "cycle", "status", "setting1", "setting2", "setting3"])].values
train_y = train["status"].values

test_X = test[test.columns.difference(["id", "cycle", "status", "setting1", "setting2", "setting3"])].values
test_y = test["status"].values

validate_X = validate[validate.columns.difference(["id", "cycle", "status", "setting1", "setting2", "setting3"])].values
validate_y = validate["status"].values

## COMPONENT ANALYSIS
pca = sklearn.decomposition.PCA(n_components = n_comps).fit(train_X)

fullX = pca.transform(full_X)
trainX = pca.transform(train_X)
testX = pca.transform(test_X)
validateX = pca.transform(validate_X)

evals = pca.explained_variance_ratio_
evals_cs = evals.cumsum()
plt.plot(range(1, n_comps + 1), evals, "o", label = "individual")
plt.plot(range(1, n_comps + 1), evals_cs, "o", label = "cumulative")
plt.legend()
plt.savefig(os.path.join(plot_path, "PCA_variance_ratio.png"))
plt.show()

## PLOT SAMPLES ALONG THE FIRST 2 PRINCIPAL COMPONENTS
X_full = pd.DataFrame(fullX, columns=["comp_" + str(x) for x in range(1, fullX.shape[1] + 1)])
X_full["RUL"] = full["RUL"]
X_full["status"] = full["status"]

X_train = pd.DataFrame(trainX, columns=["comp_" + str(x) for x in range(1, trainX.shape[1] + 1)])
X_train["RUL"] = train["RUL"]
X_train["status"] = train["status"]

X_test = pd.DataFrame(testX, columns=["comp_" + str(x) for x in range(1, testX.shape[1] + 1)])
X_test["RUL"] = test["RUL"]
X_test["status"] = test["status"]

X_validate = pd.DataFrame(validateX, columns=["comp_" + str(x) for x in range(1, validateX.shape[1] + 1)])
X_validate["RUL"] = validate["RUL"]
X_validate["status"] = validate["status"]

sns.lmplot("comp_1", "comp_2", hue = "status", data = X_full, 
           markers = "o", fit_reg = False, scatter_kws = {"alpha":0.5, "s":20})
plt.title("full set")
plt.xlim(-1.5, 1.5)
plt.ylim(-1., 1.)
if merge_data:
    plt.savefig(os.path.join(plot_path, "full_set_merge.png"))
else:
    plt.savefig(os.path.join(plot_path, "full_set.png"))
plt.show()

sns.lmplot("comp_1", "comp_2", hue = "status", data = X_train, 
           markers = "o", fit_reg = False, scatter_kws = {"alpha":0.5, "s":20})
plt.title("training set")
plt.xlim(-1.5, 1.5)
plt.ylim(-1., 1.)
if merge_data:
    plt.savefig(os.path.join(plot_path, "training_set_merge.png"))
else:
    plt.savefig(os.path.join(plot_path, "training_set.png"))
plt.show()

sns.lmplot("comp_1", "comp_2", hue = "status", data = X_test, 
           markers = "o", fit_reg = False, scatter_kws = {"alpha":0.5, "s":20})
plt.title("test set")
plt.xlim(-1.5, 1.5)
plt.ylim(-1., 1.)
if merge_data:
    plt.savefig(os.path.join(plot_path, "test_set_merge.png"))
else:
    plt.savefig(os.path.join(plot_path, "test_set.png"))
plt.show()

sns.lmplot("comp_1", "comp_2", hue = "status", data = X_validate, 
           markers = "o", fit_reg = False, scatter_kws = {"alpha":0.5, "s":20})
plt.title("validate set")
plt.xlim(-1.5, 1.5)
plt.ylim(-1., 1.)
if merge_data:
    plt.savefig(os.path.join(plot_path, "validate_set_merge.png"))
else:
    plt.savefig(os.path.join(plot_path, "validate_set.png"))
plt.show()

# Save the preprocessed data
X_train["id"] = train["id"]
X_train["cycle"] = train["cycle"]
X_validate["id"] = validate["id"]
X_validate["cycle"] = validate["cycle"]
X_test["id"] = test["id"]
X_test["cycle"] = test["cycle"]

if merge_data:
    X_full.to_csv(os.path.join(data_path, "full_" + set_name + "_PC_merge.csv"), index = False)
    X_train.to_csv(os.path.join(data_path, "train_" + set_name + "_PC_merge.csv"), index = False)
    X_test.to_csv(os.path.join(data_path, "test_" + set_name + "_PC_merge.csv"), index = False)
    X_validate.to_csv(os.path.join(data_path, "validate_" + set_name + "_PC_merge.csv"), index = False)
else:
    X_full.to_csv(os.path.join(data_path, "full_" + set_name + "_PC.csv"), index = False)
    X_train.to_csv(os.path.join(data_path, "train_" + set_name + "_PC.csv"), index = False)
    X_test.to_csv(os.path.join(data_path, "test_" + set_name + "_PC.csv"), index = False)
    X_validate.to_csv(os.path.join(data_path, "validate_" + set_name + "_PC.csv"), index = False)

g = sns.jointplot(x = "comp_1", y = "comp_2", data = X_train, 
                      kind = "hex", stat_func = None,
                     xlim = (-1.0, 1.5), ylim = (-0.5, 0.5))
plt.show()

g = sns.jointplot(x = "comp_1", y = "comp_2", data = X_test, 
                      kind = "hex", stat_func = None,
                     xlim = (-1.0, 1.5), ylim = (-0.5, 0.5))
plt.show()

g = sns.jointplot(x = "comp_1", y = "comp_2", data = X_validate, 
                      kind = "hex", stat_func = None,
                     xlim = (-1.0, 1.5), ylim = (-0.5, 0.5))
plt.show()