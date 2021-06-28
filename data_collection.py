import fusion_functions as functions
import pandas as pd

# Function that generates train/test/validation splits based on data from M Best/G in het Veld data
def loading_NRG_tables(counttable, sampleinfotable):
    x_train = counttable.T[sampleinfotable.isTraining == 1]
    y_train = sampleinfotable[sampleinfotable.isTraining == 1]
    x_test = counttable.T[sampleinfotable.isEvaluation == 1]
    y_test = sampleinfotable[sampleinfotable.isEvaluation == 1]
    x_val = counttable.T[sampleinfotable.isValidation == 1]
    y_val = sampleinfotable[sampleinfotable.isValidation == 1]
    return (x_train, y_train, x_test, y_test, x_val, y_val)


# Load validation set
sample_info = pd.read_csv("all_classes.csv", sep=",")
sample_info = sample_info.set_index("SampleID")

all_fusion, CPM_all_table, seq_info_all = functions.collect_output(
    "/media/tepdance/Disk2/Edward_testing/All_output_in_one/", sample_info)

# Load HTseq files

