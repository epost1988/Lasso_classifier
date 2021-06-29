import functions_bin as functions_bin
import data_collection as data_collection
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load variables
rfc = RandomForestClassifier(random_state=5)
lb = LabelEncoder()

# Collect benchmarking sampleinfo and count table.
benchmark_counts = pd.read_csv("counts_benchmark.csv", sep=",", index_col=0)
benchmark_sampleinfo = pd.read_csv("sampleinfo_benchmark.csv", sep=",", index_col=0)

# Add binary group for lasso regression:
binary_groups_benchmark = lb.fit_transform(benchmark_sampleinfo.group)
benchmark_sampleinfo.insert(1, "Binary_Group", binary_groups_benchmark, True)

# Normalize Data
CPM_benchmark_counts = functions_bin.normalize_counttable(benchmark_counts)

# Split samples as performed by N Sol
x_train_bench, y_train_bench, x_test_bench, y_test_bench, x_val_bench, y_val_bench = data_collection.loading_NRG_tables(CPM_benchmark_counts, benchmark_sampleinfo)

# search for optimal number of features.
alphas = np.arange(0.00001, 0.0001, 0.00001)
Benchmark_lasso_features = functions_bin.optimal_lasso_selection_NRG(alphas, x_train_bench, y_train_bench.Binary_Group,
                                                                     x_test_bench, y_test_bench.Binary_Group, 300000)

param_grid = {
    'bootstrap': [True],
    'max_depth': [8, 10, 13, 16, 20],
    'max_features': [2, 4, 6, 8],
    'min_samples_leaf': [2, 3, 4],
    'min_samples_split': [2, 4, 6, 8],
    'n_estimators': [1400, 1600]
}

# Filter lasso genes result from CPM table to respective groups.
lasso_CPM = CPM_benchmark_counts.T.loc[Benchmark_lasso_features.index]
x_train_bench_lasso = x_train_bench.T.loc[Benchmark_lasso_features.index]
x_test_bench_lasso = x_train_bench.T.loc[Benchmark_lasso_features.index]
x_val_bench_lasso = x_val_bench.T.loc[Benchmark_lasso_features.index]

# Gridsearch for optimal settings
benchmark_best_settings = functions_bin.grid_search_rfc_benchmark(x_train_bench_lasso, y_train_bench, param_grid, rfc)

print("Performing benchmark dataset LOOCV and ROC curve metrics.")
functions_bin.plot_roc_accuracy_full(lasso_CPM, benchmark_sampleinfo, benchmark_best_settings, x_train_bench,
                                     x_test_bench, x_val_bench, "Test_benchmark.png")
