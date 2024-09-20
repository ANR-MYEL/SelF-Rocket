#====================================================================================================
#===
#=== Mouhamadou Mansour Lo, Gildas Morvan, Mathieu Rossi, Fabrice Morganti, David Mercier
#===
#=== Time series classification with random convolution kernels based transforms: pooling operators and input representations matter
#===
#=== https://arxiv.org/pdf/2409.01115
#===
#=== Source of SelF-Rocket.
#===
#=== v1.0.0 - 2024/09/19 - Included the dataset name selection option
#===                       Changed the name of various existing options
#===                       Removed the rdt otion
#===
#=== 
#=== 
#====================================================================================================

import argparse
import os
import time
import random
import csv

import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from tsml.datasets import load_from_ts_file
from sklearn.model_selection import StratifiedKFold


from features_generator import fit,transform
import warnings
# TODO : Remove this filter to have better error outputs
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-fpk", "--features_per_kernel", type=int, required=False, default=5)
parser.add_argument("-fpc", "--features_per_classifier", type=int, required=False, default=2500)
parser.add_argument("-df", "--inputDataFolder", type=str, required=True)
parser.add_argument("-dn", "--inputDatasetName", type=str, required=True)
parser.add_argument("-k", "--k_fold", type=int, required=False, default=2)
parser.add_argument("-r", "--num_resamples", type=int, required=False, default=30)

arguments = parser.parse_args()

def generate_FV_train(data_path,dst,num_kfold,num_features_pk):

    Xt, y = load_from_ts_file(data_path+dst+'/'+dst+str(j)+'_TRAIN.ts')
    Xt = np.squeeze(Xt)
    X_diff = np.diff(Xt, 1)
    parameters1 = fit(Xt)
    parameters2 = fit(X_diff)
    X_transform = transform(Xt,X_diff,parameters1,parameters2,n_features_per_kernel=num_features_pk)
    scaler = preprocessing.StandardScaler().fit(X_transform)
    X_transform = scaler.transform(X_transform)
    skf = StratifiedKFold(n_splits=num_kfold)
    k_fold = skf.split(X_transform, y)

    ppv = X_transform[:,0:9996]
    lspv = X_transform[:,9996:19992]
    mpv = X_transform[:,19992:29988]
    mipv = X_transform[:,29988:39984]
    gmp = X_transform[:,39984:49980]
    ppv_diff = X_transform[:,49980:59976]
    lspv_diff = X_transform[:,59976:69972]
    mpv_diff = X_transform[:,69972:79968]
    mipv_diff = X_transform[:,79968:89964]
    gmp_diff = X_transform[:,89964:99960]
    ppv_mix = np.concatenate((ppv, ppv_diff), axis=1)
    lspv_mix = np.concatenate((lspv, lspv_diff), axis=1)
    mpv_mix = np.concatenate((mpv, mpv_diff), axis=1)
    mipv_mix = np.concatenate((mipv, mipv_diff), axis=1)
    gmp_mix = np.concatenate((gmp, gmp_diff), axis=1)
    pooling_op = [ppv,gmp,mpv,mipv,lspv,ppv_diff,gmp_diff,mpv_diff,
                mipv_diff,lspv_diff,ppv_mix,gmp_mix,mpv_mix,mipv_mix,lspv_mix]
    return y,pooling_op,k_fold,scaler,parameters1,parameters2

def generate_FV_test(data_path,dst,num_features_pk,scaler,parameters1,parameters2):

    Xt, y = load_from_ts_file(data_path+dst+'/'+dst+str(j)+'_TEST.ts')
    Xt = np.squeeze(Xt)
    X_diff = np.diff(Xt, 1)
    X_transform = transform(Xt,X_diff,parameters1,parameters2,n_features_per_kernel=num_features_pk)
    X_transform = scaler.transform(X_transform)

    ppv = X_transform[:,0:9996]
    lspv = X_transform[:,9996:19992]
    mpv = X_transform[:,19992:29988]
    mipv = X_transform[:,29988:39984]
    gmp = X_transform[:,39984:49980]
    ppv_diff = X_transform[:,49980:59976]
    lspv_diff = X_transform[:,59976:69972]
    mpv_diff = X_transform[:,69972:79968]
    mipv_diff = X_transform[:,79968:89964]
    gmp_diff = X_transform[:,89964:99960]
    ppv_mix = np.concatenate((ppv, ppv_diff), axis=1)
    lspv_mix = np.concatenate((lspv, lspv_diff), axis=1)
    mpv_mix = np.concatenate((mpv, mpv_diff), axis=1)
    mipv_mix = np.concatenate((mipv, mipv_diff), axis=1)
    gmp_mix = np.concatenate((gmp, gmp_diff), axis=1)
    pooling_op = [ppv,gmp,mpv,mipv,lspv,ppv_diff,gmp_diff,mpv_diff,
                mipv_diff,lspv_diff,ppv_mix,gmp_mix,mpv_mix,mipv_mix,lspv_mix]

    return y,pooling_op
    
if __name__ == '__main__':
    """
        Read the options as variables
    """
    data_path = arguments.inputDataFolder
    num_kfold = arguments.k_fold
    num_features_pk = arguments.features_per_kernel
    num_features_pc = arguments.features_per_classifier
    num_resamples = arguments.num_resamples
    dataset = arguments.inputDatasetName
    """
        Identify the dataset from the options.
    """
    # Check that the provided dataset name is valid (a directory having this 
    # name has to be present in the directory whose path is "inputDataFolder").
    # TODO : Check that the directory exists in $inputDataFolder
    # Create the list of dataset for which computations will be made.
    # Currently only one, but in the future, inputDatasetName might be a name list
    datasets = [dataset]

    # Declare the directory where results are written
    output_path = os.getcwd() + "/results/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    

    all_datasets_acc  = []
    all_datasets_time  = []
    matr_acc = np.zeros((len(datasets), num_kfold * num_resamples * 15), dtype=np.float32)

    for i in range(len(datasets)):

        dst = datasets[i]
        accuracy_tab = []
        crea_time = []
        classif_train_time = []
        classif_test_time = []
        select_time = []

        for j in tqdm(range(num_resamples)):

            start = time.time()
            y_train,pooling_op,k_fold,scaler,parameters1,parameters2 = generate_FV_train(data_path,dst,num_kfold,num_features_pk)
            crea_time.append(time.time()-start)
            y_test,pooling_op_t = generate_FV_test(data_path,dst,num_features_pk,scaler,parameters1,parameters2)
            
            pooling_names = ["PPV","GMP","MPV","MIPV","LSPV","PPV_DIFF","GMP_DIFF","MPV_DIFF",
                             "MIPV_DIFF","LSPV_DIFF","PPV_MIX","GMP_MIX","MPV_MIX","MIPV_MIX","LSPV_MIX"]
            times = ["TIME_CREATION_FEATURES","TIME_PO_SELECTION","TIME_TRAIN_CLASSIFIER","TIME_TEST_CLASSIFIER"]
            
            vect_acc_kf = np.zeros((num_kfold * 15), dtype=np.float32)

            start = time.time()
            for l,(train_index, test_index) in enumerate(k_fold):
                for k in range(len(pooling_op)):
                    feature_idx = random.sample(range(0,pooling_op[k].shape[1]),num_features_pc)
                    features = pooling_op[k][train_index][:,feature_idx]
                    y_train_kfold = y_train[train_index]
                    features_t  = pooling_op[k][test_index][:,feature_idx]
                    y_test_kfold = y_train[test_index]
                    classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
                    classifier.fit(features, y_train_kfold)
                    y_pred = classifier.predict(features_t)
                    accuracy = accuracy_score(y_test_kfold, y_pred)
                    matr_acc[i,num_kfold*num_resamples*k+num_kfold*j+l] = accuracy
                    vect_acc_kf[num_kfold*k + l] = accuracy
            select_time.append(time.time()-start)

        accuracy_tab = [np.mean(matr_acc[i,num_kfold*num_resamples*v:num_kfold*num_resamples*v+num_kfold*num_resamples]) for v in range(15)]
        ind_max = accuracy_tab.index(max(accuracy_tab))
        tab_rs_acc = []
        for j in tqdm(range(num_resamples)):
            
            y_train,pooling_op,k_fold,scaler,parameters1,parameters2 = generate_FV_train(data_path,dst,num_kfold,num_features_pk)
            y_test,pooling_op_t = generate_FV_test(data_path,dst,num_features_pk,scaler,parameters1,parameters2)

            start = time.time()
            classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
            features_train = pooling_op[ind_max]
            classifier.fit(features_train, y_train)
            classif_train_time.append(time.time()-start)
            start = time.time()
            features_test = pooling_op_t[ind_max]
            y_pred_fin = classifier.predict(features_test)
            classif_test_time.append(time.time()-start)
            accuracy_fin = accuracy_score(y_test, y_pred_fin)
            tab_rs_acc.append(accuracy_fin)

        accuracy_tab.insert(0,np.mean(tab_rs_acc))
        pooling_names.insert(0,"SelF-Rocket performance")
        time_tab = [np.mean(crea_time),np.mean(select_time),np.mean(classif_train_time),np.mean(classif_test_time)]
        all_datasets_acc.append([dst] + accuracy_tab)
        all_datasets_time.append([dst] + time_tab)

        output_acc=os.path.join(output_path,"accuracy_self_rocket_k{}_f{}_dsn{}.csv".format(num_kfold,num_features_pc,dst))
        output_mat=os.path.join(output_path,"matr_acc_self_rocket_k{}_f{}_dsn{}.npy".format(num_kfold,num_features_pc,dst))
        output_time=os.path.join(output_path,"time_self_rocket_k{}_f{}_dsn{}.csv".format(num_kfold,num_features_pc,dst))
        
        with open(output_acc, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset'] + pooling_names)
            writer.writerows(all_datasets_acc)
        
        with open(output_time, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset'] + times)
            writer.writerows(all_datasets_time)

        with open(output_mat, 'wb') as f:
            np.save(f,matr_acc)   

    # Tell that the program ended
    print("Regular end of the execution")
