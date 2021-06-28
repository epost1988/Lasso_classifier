# As the raw htseq files were not uploaded, the following code was used to load in from raw htseq files.
# The dataset contained some issues:    wrong names for files
#                                       Wrong file name extensions
#                                       reseq files
# In order to cope with this, these lines below were hard coded.
# Input requires a file containing the names of the htseq files the user would like to include
# A pathway to where they are located must also be provided
#
import glob
import pandas as pd

# collect sample names
filenames = pd.DataFrame(all_fusion.T.index)
filenames.to_csv("all_files_search.csv", sep=",")

length_foldername = len("Htseq_files/")
length_file_extension = len(".htseq.ssv")

# read sample names and search folder
# REMARK: search folder is currently hard coded, needs work
filenames = pd.read_csv("all_files_search.csv", index_col=0)
filenames_list = filenames.values
file_locations = []
missing_sample_names = []
for i in filenames_list:
    filename = (i[0])
    if (glob("Htseq_files/" + filename + ".bam.htseq*")) != []:
        file_locations.append(glob("Htseq_files/" + filename + ".bam.htseq*"))
    elif (glob("Htseq_files/" + filename + ".htseq*")) != []:
        file_locations.append(glob("Htseq_files/" + filename + ".htseq*"))
    else:
        missing_sample_names.append(i)

# acquire lengths to remove to aqcuire filename, does not account for samples ending with .bam


for f in file_locations:
    file_path = f[0]
    # Remove folder name and extension, taking the files containing .bam in account
    if "bam" in file_path:
        filename = file_path[length_foldername:-(length_file_extension + len(".bam"))]
    else:
        filename = file_path[length_foldername:-length_file_extension]
    try:
        Htseq_table[filename] = pd.read_csv(file_path, delimiter='\t', header=None, names=[filename])
    except NameError:
        Htseq_table = pd.read_csv(file_path, delimiter='\t', header=None, index_col=[0], names=[filename])
print("Files searched:", len(filenames_list))
print("Files found: ", len(Htseq_table.T))
print("Missing files:")
for i in missing_sample_names:
    print(i[0])

sample1 = ["Vumc-NSCLC-109"]
sample2 = ["Vumc-NSCLC-441-copy-TR2238"]
sample3 = ["Vumc-HD-4-TR1150-reseq"]
missing1 = pd.read_csv("/media/tepdance/Disk2/Edward_testing/Htseq_files/Vumc-NSCLC-109-1.bam.htseq.ssv", sep="\t",
                       index_col=0, names=sample1)
missing2 = pd.read_csv("/media/tepdance/Disk2/Edward_testing/Htseq_files/Vumc-NSCLC-441-TR2238.htseq.ssv", sep="\t",
                       index_col=0, names=sample2)
missing3 = pd.read_csv("/media/tepdance/Disk2/Edward_testing/Htseq_files/Vumc-HD-4-TR1150.htseq.ssv", sep="\t",
                       index_col=0, names=sample3)

missing_samples1 = missing1.join(missing2)
missing_samples = missing_samples1.join(missing3)

Htseq_table_final = Htseq_table.join(missing_samples)

# Remove unwanted rows
Htseq_table_final.drop(Htseq_table_final.tail(5).index, inplace=True)