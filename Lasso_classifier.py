import pandas as pd
import seaborn as sns; sns.set(color_codes=True)
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


sample_info = pd.read_csv("sampleinfo.csv", sep=",")
sample_info = sample_info.set_index("SampleID")

# Load datafiles, htseq count = standard method, all counts includes fusion genes detected.
htseq_count_table = pd.read_csv("htseq_table.csv", sep=",", index_col=0)
all_counts_table = pd.read_csv("htseq_rawfusion_combined.csv", sep=",", index_col=0)



# Load sampleinfo and set index
sample_info = pd.read_csv("all_classes.csv", sep=",")
sample_info = sample_info.set_index("SampleID")

# Add binary group for classification
try:
    binary_y = lb.fit_transform(sample_info.Group)
    sample_info["Binary_Group"] = binary_y
except ValueError:
    pass

# Remove samples with too low read count.
print("checking validation cohort for read count:")
Htseq_table_filtered, sample_info, dropped_samples = functions_bin.min_reads_filter(htseq_count_table, sample_info)

# Make sure all the files in use only contain the samples we'll be using.
all_counts_filtered = all_counts_table.T.loc[Htseq_table_filtered.index]
all_counts_CPM = functions_bin.normalize_counttable(all_counts_filtered)
htseq_genes = all_counts_CPM.index.isin(Htseq_table_filtered.index)
htseq_counts_CPM = all_counts_CPM.loc[htseq_genes]
fusion_counts_CPM = all_counts_CPM.loc[~htseq_genes]

# Split into train, test and validation sets
train_test_set, validation_set, train_test_info, \
validation_info = train_test_split(htseq_counts_CPM.T, sample_info.Binary_Group,
                                   test_size = 0.33, stratify = sample_info.Binary_Group, random_state = 10)

# Slight error somewhere in the code, this fixes it.
train_test_info = sample_info.loc[train_test_info.index]

# Set apart a train/test split.
htseq_train_test = htseq_counts_CPM.loc[train_test_info.index]

print("Selecting fusion genes only Lasso features:")
lassos_fusion = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
fusion_feature_table, train_set, test_set, fusion_featured_selected, fusion_lasso_model, \
test_scores_fusion = functions_bin.optimal_lasso_selection(lassos_fusion, fusion_counts_CPM.T, train_test_info, 0.4, 300000)

# select highest scoring alpha for htseq files only, and extract selected genes
print("Selecting htseq genes only Lasso features:")
lassos_htseq = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
lasso_htseq_feature_table, train_set, test_set, htseq_featured_selected, htseq_lasso_model, \
test_scores_htseq = functions_bin.optimal_lasso_selection(lassos_htseq, htseq_train_test.T, train_test_info, 0.4, 300000)

# Create combined dataframe
combined_features = lasso_htseq_feature_table.append(fusion_feature_table)

sample_info_test = sample_info.loc[test_set.index]
# Find best modeling settings for htseq genes
best_settings_gridsearch_combined = functions_bin.grid_search_rfc(combined_features, sample_info_test,
                                                         train_set.index, param_grid_rfc, rfc)
# Find best modeling settings for htseq + fusion genes
best_settings_gridsearch_htseq = functions_bin.grid_search_rfc(lasso_htseq_feature_table, sample_info_test,
                                                         train_set.index, param_grid_rfc, rfc)

functions_bin.plot_roc_accuracy_full(combined_features, sample_info, best_settings_gridsearch_combined, train_set, test_set, validation_set, "htseq_fusion_ROC.png")
functions_bin.plot_roc_accuracy_full(lasso_htseq_feature_table, sample_info, best_settings_gridsearch_htseq, train_set, test_set, validation_set, "Htseq_ROC.png")

