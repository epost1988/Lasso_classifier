import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# Remove samples with total expression < 100000
def min_reads_filter(dataframe, sampleinfo):
    dropped_samples = []
    dropped_table = dataframe
    dropped_info = sampleinfo.T
    count = 0
    sum_reads = dataframe.sum()
    for i in sum_reads:
        if i>= 100000:
            pass
        else:
            print("dropped sample " + dataframe.T.index[count] + " due to low read count: " + str(i))
            dropped_table = dropped_table.drop(columns=[dataframe.T.index[count]])
            dropped_info = dropped_info.drop(columns=[dataframe.T.index[count]])
            dropped_samples.append(dataframe.T.index[count])
        count +=1
    dropped_info = dropped_info.T
    return(dropped_table, dropped_info, dropped_samples)

# Normalize data to CPM and sqrt it
def normalize_counttable(data):
    Total_counts = data.sum(axis=0)
    data_CPM = data.loc[:,Total_counts.index].div(Total_counts)*1000000
    normalized_table = data_CPM.apply(np.sqrt).fillna(0)
    return(normalized_table)


# Extract optimal settings from rfc
def optimal_settings(output_feature_selection):
    bootstrap = output_feature_selection["bootstrap"]
    max_features = output_feature_selection["max_features"]
    n_estimators = output_feature_selection["n_estimators"]
    min_samples_leaf = output_feature_selection["min_samples_leaf"]
    min_samples_split = output_feature_selection["min_samples_split"]
    max_depth = output_feature_selection["max_depth"]
    return (bootstrap, max_features, n_estimators, min_samples_leaf, min_samples_split, max_depth)

# Gridsearch for benchmark dataset
def grid_search_rfc_benchmark(x_train, y_train, param_grid, model):
    CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    CV_rfc.fit(x_train, y_train.Binary_Group)
    return (CV_rfc.best_params_)

# Lasso feature selection for NRG dataset
def optimal_lasso_selection_NRG(set_alphas, X_train, Y_train, X_test, Y_test, iterations):
    optimal_alpha = -10
    optimal_features = 0
    test_scores = []
    lasso_output = pd.DataFrame()
    features = pd.DataFrame()
    print("Performing lasso feature selection with alphas " + str(set_alphas))
    for i in set_alphas:
        lasso = Lasso(alpha=i, max_iter=iterations)
        lasso.fit(X_train, Y_train)
        train_score = lasso.score(X_train, Y_train)
        test_score = lasso.score(X_test, Y_test)
        coeff_used = np.sum(lasso.coef_ != 0)
        test_scores.append(test_score)
        if test_score >= optimal_alpha:
            optimal_alpha = test_score
            optimal_features = coeff_used
            lasso_output = pd.DataFrame(lasso.coef_ != 0, index=X_train.T.index, columns=["Lasso"])
            features = lasso_output[lasso_output.Lasso == True]

    print("the best scoring test alpha selected was", optimal_alpha)
    print("number of features selected:", optimal_features)
    selected_optimal_features = X_train.T.loc[features.index]
    return (selected_optimal_features)

# ROC plotting. Includes LOOCV for training validation.
def plot_roc_accuracy_full(dataset, sample_info, best_settings, train_set, test_set, validation_set, filename):
    bootstrap1, max_features1, n_estimators1, min_samples_leaf1, min_samples_split1, max_depth1 = optimal_settings(
        best_settings)
    X_train = dataset.T.loc[train_set.index]
    X_test = dataset.T.loc[test_set.index]
    y_train = sample_info.loc[train_set.index]
    y_test = sample_info.loc[test_set.index]
    x_validation = dataset.T.loc[validation_set.index]
    y_validation = sample_info.loc[validation_set.index]

    rfc = RandomForestClassifier(bootstrap=bootstrap1, max_features=max_features1, n_estimators=n_estimators1,
                                 min_samples_leaf=min_samples_leaf1, min_samples_split=min_samples_split1,
                                 max_depth=max_depth1, random_state=5, n_jobs=-1)

    rfc.fit(X_train, y_train.Binary_Group)
    train_proba = cross_val_predict(rfc, X_train, y_train.Binary_Group, cv=KFold(X_train.shape[0]),
                                    method="predict_proba")
    train_proba = train_proba[:, 1]
    fpr, tpr, threshold = roc_curve(y_train.Binary_Group, train_proba)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC Training = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    proba = rfc.predict_proba(X_test)
    proba = proba[:, 1]
    fpr2, tpr2, threshold = roc_curve(y_test.Binary_Group, proba)
    roc_auc = metrics.auc(fpr2, tpr2)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr2, tpr2, 'r', label='AUC Test = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    validation_proba = rfc.predict_proba(x_validation)
    validation_proba = validation_proba[:, 1]
    fpr3, tpr3, threshold = roc_curve(y_validation.Binary_Group, validation_proba)
    roc_auc = metrics.auc(fpr3, tpr3)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr3, tpr3, 'g', label='AUC validation = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(filename)

    train_prd = cross_val_predict(rfc, X_train, y_train.Binary_Group, cv=KFold(X_train.shape[0]))
    cm, Accuracy = functions.build_cm(y_train.Binary_Group, train_prd)
    print("train set:")
    print(cm)
    print("Accuracy:", Accuracy)

    test_prd = rfc.predict(X_test)
    cm, Accuracy = functions.build_cm(y_test.Binary_Group, test_prd)
    print("test set:")
    print(cm)
    print("Accuracy:", Accuracy)

    validation_prd = rfc.predict(x_validation)
    cm, Accuracy = functions.build_cm(y_validation.Binary_Group, validation_prd)
    print("validation set matrix/accuracy:")
    print(cm)
    print("Accuracy:", Accuracy)


