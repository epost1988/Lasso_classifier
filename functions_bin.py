import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add binary group to sampleinfo file.
def binarize_sampleinfo(sample_info):
    lb = LabelEncoder()
    try:
        binary_step1 = lb.fit_transform(sample_info)
        binary_step2 = sample_info
        binary_step2.insert(1, "Binary_Group", binary_step1, True)
        sample_info = binary_step2.loc[sample_info.index]
    except ValueError:
        pass
    return(sample_info)


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

# Build confusion matrix:
def build_cm(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return(cm,accuracy)

# Normalize data to CPM and sqrt it
def normalize_counttable(data):
    Total_counts = data.sum(axis=0)
    data_CPM = data.loc[:,Total_counts.index].div(Total_counts)*1000000
    normalized_table = data_CPM.T.apply(np.sqrt).fillna(0)
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

# Gridsearch random forest
def grid_search_rfc(dataset, sampleinfo, train_index, param_grid, model):
    CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5, verbose=1, n_jobs = -1)
    x_train = dataset.T.loc[train_index]
    y_train = sampleinfo.loc[train_index]
    CV_rfc.fit(x_train, y_train.Binary_Group)
    return(CV_rfc.best_params_, CV_rfc)

# Gridsearch for benchmark dataset
def grid_search_rfc_benchmark(x_train, y_train, param_grid, model):
    CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    CV_rfc.fit(x_train, y_train.Binary_Group)
    return (CV_rfc.best_params_)

# Loop for optimal lasso generation.
def optimal_lasso_selection(set_alphas, CPM_table, group_info, split_size, iterations):
    optimal_alpha = 0
    optimal_features = 0
    test_scores = []
    lasso_output = pd.DataFrame()
    features = pd.DataFrame()
    X_train, X_test, Y_train, Y_test = train_test_split(CPM_table.T, group_info.Binary_Group, test_size = split_size, stratify = group_info.Binary_Group, random_state = 5)
    for i in set_alphas:
        print("Performing lasso feature selection with alpha " + str(i))
        lasso = Lasso(alpha=i, max_iter = iterations)
        lasso.fit(X_train, Y_train)
        train_score = lasso.score(X_train, Y_train)
        test_score = lasso.score(X_test, Y_test)
        coeff_used = np.sum(lasso.coef_!=0)
        test_scores.append(test_score)
        if test_score >= optimal_alpha:
            optimal_alpha = test_score
            optimal_alpha_value = i
            optimal_features = coeff_used
            lasso_output = pd.DataFrame(lasso.coef_!=0, index=CPM_table.index, columns=["Lasso"])
            features = lasso_output[lasso_output.Lasso == True]
    
    print("the best scoring test alpha selected was", optimal_alpha_value)
    print("test score acquired =", optimal_alpha)
    print("number of features selected:", optimal_features)
    selected_optimal_features = CPM_table.loc[features.index]
    return(selected_optimal_features, X_train, X_test, features, lasso, test_scores)

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
    cm, Accuracy = build_cm(y_train.Binary_Group, train_prd)
    print("train set:")
    print(cm)
    print("Accuracy:", Accuracy)

    test_prd = rfc.predict(X_test)
    cm, Accuracy = build_cm(y_test.Binary_Group, test_prd)
    print("test set:")
    print(cm)
    print("Accuracy:", Accuracy)

    validation_prd = rfc.predict(x_validation)
    cm, Accuracy = build_cm(y_validation.Binary_Group, validation_prd)
    print("validation set matrix/accuracy:")
    print(cm)
    print("Accuracy:", Accuracy)


# SVM gridsearch:
def SVC_plot_ROC(model, X_train, X_test, X_val, y_train, y_test, y_val))
    try:
        train_proba = cross_val_predict(model.best_estimator_, X_train, y_train.Binary_Group, cv = KFold(X_train.shape[0]), method="predict_proba")
        train_proba = train_proba[:,1]
        train_prd = cross_val_predict(model.best_estimator_, X_train, y_train.Binary_Group, cv = KFold(X_train.shape[0]))
        fpr, tpr, threshold = roc_curve(y_train.Binary_Group, train_proba)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC Training = %0.3f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
    
        proba = model_htseq.predict_proba(X_test)
        proba = proba[:,1]
        fpr2, tpr2, threshold = roc_curve(y_test.Binary_Group, proba)
        roc_auc = metrics.auc(fpr2, tpr2)
    
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr2, tpr2, 'r', label = 'AUC Test = %0.3f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        
        validation_proba = model_htseq.predict_proba(X_val)
        validation_proba = validation_proba[:,1]
        fpr3, tpr3, threshold = roc_curve(y_val.Binary_Group, validation_proba)
        roc_auc = metrics.auc(fpr3, tpr3)
        
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr3, tpr3, 'g', label = 'AUC validation = %0.3f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('SVM_ROC-Curve.png')
        plt.show()
        print()
        #train_prd = model_htseq_fusion.predict(X_train)
        cm, Accuracy = functions.build_cm(y_train.Binary_Group, train_prd)
        print("train set matrix/accuracy:")
        print(cm)
        print("Accuracy:", Accuracy)
        print()
        test_prd = model_htseq.predict(X_test)
        cm, Accuracy = functions.build_cm(y_test.Binary_Group, test_prd)
        print("test set matrix/accuracy:")
        print(cm)
        print("Accuracy:",Accuracy)
        print()
        validation_prd = model_htseq.predict(X_val)
        cm, Accuracy = functions.build_cm(y_val.Binary_Group, validation_prd)
        print("validation set matrix/accuracy:")
        print(cm)
        print("Accuracy:",Accuracy)
    
    except KeyError:
        pass
# Plot feature importance RFC model
def plot_feature_importance(importances, xlabels, figname):
    fig = pyplot.figure(figsize=(15,8))
    x_axis_range = range(len(importances))
    plt.bar([x for x in range(len(importances))], importances)
    plt.xticks(x_axis_range, xlabels, rotation='vertical')
    plt.rcParams["figure.figsize"] = (15, 15)
    plt.savefig(str(figname))


# Remove uninformative features for fusion genes
def select_feature_expression(data, n, percentage):
    print("selecting features with minimal CPM count:", n)
    print("and present in", percentage, "% of samples")
    datafile = data.T
    #datafile = data
    output = []
    for name in datafile:
        output.append(len(datafile[datafile[name]>n]))
    Filtered = pd.DataFrame(output)
    Filtered = Filtered.set_index(data.index)
    Filtered = Filtered.loc[Filtered[0]>(len(data.columns)*percentage)]
    return (Filtered, output)



