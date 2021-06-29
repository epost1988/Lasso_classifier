import pandas as pd
#import seaborn as sns; sns.set(color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import functions_bin as functions_bin

rfc = RandomForestClassifier(random_state=5)
lb = LabelEncoder()

# Parameter grid for random forest classifier optimization using gridsearch
param_grid_rfc = {
    'bootstrap': [True],
    'max_depth': [8, 10, 13, 16, 20],
    'max_features': [2, 4, 6, 8],
    'min_samples_leaf': [2, 3, 4],
    'min_samples_split': [2, 4, 6, 8],
    'n_estimators': [1400, 1600]
}

scores = ['precision', 'recall']

# Parameter grid for SVC optimization using gridsearch
param_grid_SVC = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 0.001, 1e-3, 1e-4],
                     'C': [0.1, 1, 10, 100, 1000]},
                                 {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
             {'kernel': ['poly'], 'gamma': [0.01, 0.001, 1e-3, 1e-4], 'C': [0.001, 0.01, 0.1, 1, 10], 'degree' : [1]}]

# Load_sampleinfo
print("loading sampleinfo")
sample_info = pd.read_csv("sampleinfo.csv", sep=",")
sample_info = sample_info.set_index("SampleID")
sample_info = functions_bin.binarize_sampleinfo(sample_info)

# Load datafiles, htseq count = standard method, all counts includes fusion genes detected.
print("Loading count tables")
htseq_count_table = pd.read_csv("htseq_counts.csv", sep=",", index_col=0)
all_counts_table = pd.read_csv("fusion_htseq_combined_counts.csv", sep=",", index_col=0)

# Remove samples with too low read count.
print("checking cohort for read count:")
Htseq_table_filtered, sample_info, dropped_samples = functions_bin.min_reads_filter(htseq_count_table, sample_info)
sample_info = sample_info.loc[Htseq_table_filtered.T.index]

# Make sure all the files in use only contain the samples we'll be using.
print("Removing samples not included")
all_counts_filtered = all_counts_table.T.loc[Htseq_table_filtered.T.index]
print("Normalizing to CPM")
all_counts_CPM = functions_bin.normalize_counttable(all_counts_filtered.T)
htseq_genes = all_counts_CPM.T.loc[Htseq_table_filtered.index].index
htseq_counts_CPM = all_counts_CPM.T.loc[htseq_genes]
fusion_counts_CPM = all_counts_CPM.T.drop(htseq_genes)

# Split into train, test and validation sets
print("splitting into groups")
train_test_set, validation_set, train_test_info, \
validation_info = train_test_split(htseq_counts_CPM.T, sample_info.Binary_Group,
                                   test_size = 0.33, stratify = sample_info.Binary_Group, random_state = 10)

# Slight error somewhere in the code, this fixes it.
train_test_info = sample_info.loc[train_test_set.index]

# Select only genes expressed in at least 1% of the samples.
fusion_counts_filtered = functions_bin.select_feature_expression(fusion_counts_CPM, 1, 0.01)
fusion_counts_filtered = fusion_counts_CPM.loc[fusion_counts_filtered[0].index]

# Set aside the samples used in training/testing
train_test_info = sample_info.loc[train_test_set.index]
fusion_train_test = fusion_counts_filtered.T.loc[train_test_info.index]
htseq_train_test = htseq_counts_CPM.T.loc[train_test_info.index]



# perform optimal feature selection using lasso loops
print("Selecting fusion genes only Lasso features:")
lassos_fusion = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
fusion_feature_table, train_set, test_set, fusion_featured_selected, fusion_lasso_model, test_scores_fusion \
= functions_bin.optimal_lasso_selection(lassos_fusion, fusion_train_test.T, train_test_info, 0.4, 300000)


print("Selecting htseq genes only Lasso features:")
lassos_htseq = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
lassos_htseq = [0.1, 0.09]
lasso_htseq_feature_table, train_set, test_set, htseq_featured_selected, htseq_lasso_model, \
test_scores_htseq = functions_bin.optimal_lasso_selection(lassos_htseq, htseq_train_test.T, train_test_info, 0.4, 300000)

# Create combined dataframe
combined_features = lasso_htseq_feature_table.append(fusion_feature_table)

sample_info_test = sample_info.loc[test_set.index]
# Find best modeling settings for htseq genes + fusion genes
sample_info = pd.read_csv("sampleinfo.csv", sep=",")
sample_info = sample_info.set_index("SampleID")
sample_info = functions_bin.binarize_sampleinfo(sample_info)
sample_info = sample_info.loc[Htseq_table_filtered.T.index]

best_settings_gridsearch_combined_rfc, model_combined = functions_bin.grid_search_rfc(combined_features, sample_info, train_set.index, param_grid_rfc, rfc)
best_settings_gridsearch_htseq_rfc, model_htseq = functions_bin.grid_search_rfc(lasso_htseq_feature_table, sample_info, train_set.index, param_grid_rfc, rfc)

# Generate ROC curves and confusion matrixes for fusion and
combined_features_all = all_counts_CPM.T.loc[combined_features.index]
lasso_features_all = all_counts_CPM.T.loc[htseq_featured_selected.index]

print("performing analysis on dataset htseq + fusion genes")
functions_bin.plot_roc_accuracy_full(combined_features_all, sample_info, best_settings_gridsearch_combined_rfc, train_set, test_set, validation_set, "htseq_fusion_ROC.png")
print("performing analysis on dataset htseq genes")
functions_bin.plot_roc_accuracy_full(lasso_features_all, sample_info, best_settings_gridsearch_htseq_rfc, train_set, test_set, validation_set, "Htseq_ROC.png")

# Generate feature importance bar plots.

bootstrap1, max_features1, n_estimators1, min_samples_leaf1, min_samples_split1, max_depth1 = functions_bin.optimal_settings(best_settings_gridsearch_combined)
bootstrap2, max_features2, n_estimators2, min_samples_leaf2, min_samples_split2, max_depth2 = functions_bin.optimal_settings(best_settings_gridsearch_htseq)

rfc_combined = RandomForestClassifier(bootstrap=bootstrap1, max_features=max_features1, n_estimators=n_estimators1,
                                 min_samples_leaf=min_samples_leaf1, min_samples_split=min_samples_split1,
                                 max_depth=max_depth1, random_state=5, n_jobs=-1)

rfc_htseq = RandomForestClassifier(bootstrap=bootstrap2, max_features=max_features2, n_estimators=n_estimators2,
                                 min_samples_leaf=min_samples_leaf2, min_samples_split=min_samples_split2,
                                 max_depth=max_depth2, random_state=5, n_jobs=-1)


rfc_combined.fit(combined_features.T.loc[train_set.index], sample_info.loc[train_set.index].Binary_Group)
rfc_htseq.fit(lasso_features_all.T.loc[train_set.index], sample_info.loc[train_set.index].Binary_Group)

combined_importances = rfc_combined.feature_importances_
htseq_importances = model_htseq.feature_importances_
xlabels_combined = combined_features.T.index
xlabels_htseq = lasso_htseq_feature_table.T.index

functions_bin.plot_feature_importance(combined_importances, xlabels_combined, "feature_importance_combined.png")
functions_bin.plot_feature_importance(htseq_importances, xlabels_htseq, "feature_importance_htseq.png")

# SVM gridsearch fusion+htseq
SVM_combined = functions_bin.SVM_gridsearch(combined_features_all.loc[train_set.index], sample_info.loc[train_set.index]
                             , combined_features_all.loc[test_set.index], sample_info.loc[test_set.index]
                             , combined_features_all.loc[val_set.index], sample_info.loc[val_set.index],
                             param_grid_svc, scores)
SVM_htseq = functions_bin.SVM_gridsearch(lasso_features_all.loc[train_set.index], sample_info.loc[train_set.index]
                             , lasso_features_all.loc[test_set.index], sample_info.loc[test_set.index]
                             , lasso_features_all.loc[val_set.index], sample_info.loc[val_set.index],
                             param_grid_svc, scores)


