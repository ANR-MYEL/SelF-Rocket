
import os
import csv
import numpy as np
import argparse
from tqdm import tqdm
from self_rocket import SelFRocket
from tsml.datasets import load_from_ts_file
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser()
parser.add_argument("-df", "--inputDataFolder", type=str, required=True)
parser.add_argument("-k", "--k_fold", type=int, required=False, default=2)
parser.add_argument("-r", "--num_resamples", type=int, required=False, default=30)
parser.add_argument("-nr", "--num_runs", type=int, required=False, default=10)

arguments = parser.parse_args()

if __name__ == '__main__':
    """
        Read the options as variables
    """
    data_path = arguments.inputDataFolder
    num_kfold = arguments.k_fold
    num_resamples = arguments.num_resamples
    num_runs = arguments.num_runs

    datasets = np.loadtxt("list_UCR_datasets.txt",dtype="str")
    output_path = os.getcwd() + "/results/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    all_datasets_perf = []
    accuracy_tab_rsmpl = []
    ir_po_tab_rsmpl = []
    pooling_names = ["PPV","GMP","MPV","MIPV","LSPV","PPV_DIFF","GMP_DIFF","MPV_DIFF",
                        "MIPV_DIFF","LSPV_DIFF","PPV_MIX","GMP_MIX","MPV_MIX","MIPV_MIX","LSPV_MIX"]
    for i in range(len(datasets)):
        dataset_perf_SR = []
        dst = datasets[i]
        dataset_ir_po_SR = []
        for j in tqdm(range(num_resamples)):
            X_train, y_train = load_from_ts_file(data_path+dst+'/'+dst+str(j)+'_TRAIN.ts')
            X_test, y_test = load_from_ts_file(data_path+dst+'/'+dst+str(j)+'_TEST.ts')
            model = SelFRocket(num_runs,num_kfold)
            model.fit(X_train,y_train)
            y_sr = model.predict(X_test)
            accuracy_SR = accuracy_score(y_test,y_sr)
            dataset_perf_SR.append(accuracy_SR)
            dataset_ir_po_SR.append(pooling_names[model.selected_comb])
        output_acc=os.path.join(output_path,"Perf_rsmpl_SR_UCR112_k{}_nr{}.csv".format(num_kfold,num_runs))
        output_main_review=os.path.join(output_path,"Mean_perf_SR_UCR112_k{}_nr{}.csv".format(num_kfold,num_runs))
        output_ir_po_by_rsmpl=os.path.join(output_path,"IR_PO_rsmpl_SR_UCR112_k{}_nr{}.csv".format(num_kfold,num_runs))
        all_datasets_perf.append([dst,np.mean(dataset_perf_SR)])
        accuracy_tab_rsmpl.append([dst]+dataset_perf_SR)
        ir_po_tab_rsmpl.append([dst]+dataset_ir_po_SR)
        
        with open(output_acc, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["dataset"]+["{}".format(rsmplind) for rsmplind in range(num_resamples)])
            writer.writerows(accuracy_tab_rsmpl)  

        with open(output_ir_po_by_rsmpl, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["dataset"]+["{}".format(rsmplind) for rsmplind in range(num_resamples)])
            writer.writerows(ir_po_tab_rsmpl) 
 
        with open(output_main_review, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["dataset","MEAN Accuracy SelF-Rocket"])
            writer.writerows(all_datasets_perf) 
 
    # Tell that the program ended
    print("Regular end of the execution")