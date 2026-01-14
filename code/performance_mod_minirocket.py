#====================================================================================================#
#===                                                                                              ===#
#=== Mouhamadou Mansour Lo, Gildas Morvan, Mathieu Rossi, Fabrice Morganti, David Mercier         ===#
#===                                                                                              ===#
#=== Time series classification with random convolution kernels:                                  ===#
#=== pooling operators and input representations matter                                           ===#
#===                                                                                              ===#
#=== https://arxiv.org/pdf/2409.01115                                                             ===#
#===                                                                                              ===#
#=== Source of SelF-Rocket.                                                                       ===#
#===                                                                                              ===#
#=== Section 03 Experiments                                                                       ===#
#===                                                                                              ===#
#===                                                                                              ===#
#===                                                                                              ===#
#===                                                                                              ===#
#===                                                                                              ===#
#====================================================================================================#

from features_generator import fit,transform
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
import csv
from tsml.datasets import load_from_ts_file

def model_perf_minirocket(datasets):
    all_datasets = []
    matr_acc = np.zeros((len(datasets), 15 * 30), dtype=np.float32)
    nb_kernels = 10000

    for i in range(len(datasets)):
        dst = datasets[i]
        accuracy_tab = []
        for j in tqdm(range(30)):

            X_train, y_train = load_from_ts_file("../datasets_UCR_resamp_tsv/"+dst+'/'+dst+str(j)+'_TRAIN.ts')
            X_test, y_test = load_from_ts_file("../datasets_UCR_resamp_tsv/"+dst+'/'+dst+str(j)+'_TEST.ts')
            X_train = np.squeeze(X_train)
            X_test = np.squeeze(X_test)
            X_train_diff = np.diff(X_train, 1)
            X_test_diff = np.diff(X_test, 1)

            parameters1 = fit(X_train,num_features=nb_kernels)
            parameters2 = fit(X_train_diff,num_features=nb_kernels)

            X_training_transform = transform(X_train,X_train_diff,parameters1,parameters2,n_features_per_kernel=5)
            
            scaler = preprocessing.StandardScaler().fit(X_training_transform)
            X_training_transform = scaler.transform(X_training_transform)
            
            X_test_transform = transform(X_test,X_test_diff,parameters1,parameters2,n_features_per_kernel=5)
            X_test_transform = scaler.transform(X_test_transform)

            nb_features_trns = (nb_kernels // 84) * 84

            ppv = X_training_transform[:,0:nb_features_trns]
            lspv = X_training_transform[:,nb_features_trns:nb_features_trns*2]
            mpv = X_training_transform[:,nb_features_trns*2:nb_features_trns*3]
            mipv = X_training_transform[:,nb_features_trns*3:nb_features_trns*4]
            zc = X_training_transform[:,nb_features_trns*4:nb_features_trns*5]
            ppv_diff = X_training_transform[:,nb_features_trns*5:nb_features_trns*6]
            lspv_diff = X_training_transform[:,nb_features_trns*6:nb_features_trns*7]
            mpv_diff = X_training_transform[:,nb_features_trns*7:nb_features_trns*8]
            mipv_diff = X_training_transform[:,nb_features_trns*8:nb_features_trns*9]
            zc_diff = X_training_transform[:,nb_features_trns*9:nb_features_trns*10]
            ppv_mix = np.concatenate((ppv, ppv_diff), axis=1)
            lspv_mix = np.concatenate((lspv, lspv_diff), axis=1)
            mpv_mix = np.concatenate((mpv, mpv_diff), axis=1)
            mipv_mix = np.concatenate((mipv, mipv_diff), axis=1)
            zc_mix = np.concatenate((zc, zc_diff), axis=1)

            ppv_t = X_test_transform[:,0:nb_features_trns]
            lspv_t = X_test_transform[:,nb_features_trns:nb_features_trns*2]
            mpv_t = X_test_transform[:,nb_features_trns*2:nb_features_trns*3]
            mipv_t = X_test_transform[:,nb_features_trns*3:nb_features_trns*4]
            zc_t = X_test_transform[:,nb_features_trns*4:nb_features_trns*5]
            ppv_diff_t = X_test_transform[:,nb_features_trns*5:nb_features_trns*6]
            lspv_diff_t = X_test_transform[:,nb_features_trns*6:nb_features_trns*7]
            mpv_diff_t = X_test_transform[:,nb_features_trns*7:nb_features_trns*8]
            mipv_diff_t = X_test_transform[:,nb_features_trns*8:nb_features_trns*9]
            zc_diff_t = X_test_transform[:,nb_features_trns*9:nb_features_trns*10]
            ppv_mix_t = np.concatenate((ppv_t, ppv_diff_t), axis=1)
            lspv_mix_t = np.concatenate((lspv_t, lspv_diff_t), axis=1)
            mpv_mix_t = np.concatenate((mpv_t, mpv_diff_t), axis=1)
            mipv_mix_t = np.concatenate((mipv_t, mipv_diff_t), axis=1)
            zc_mix_t = np.concatenate((zc_t, zc_diff_t), axis=1)
            
            pooling_op = [ppv,zc,mpv,mipv,lspv,ppv_diff,zc_diff,mpv_diff,
                          mipv_diff,lspv_diff,ppv_mix,zc_mix,mpv_mix,mipv_mix,lspv_mix]
            pooling_op_t = [ppv_t,zc_t,mpv_t,mipv_t,lspv_t,ppv_diff_t,zc_diff_t,mpv_diff_t,
                          mipv_diff_t,lspv_diff_t,ppv_mix_t,zc_mix_t,mpv_mix_t,mipv_mix_t,lspv_mix_t]
            pooling_names = ["PPV","ZC","MPV","MIPV","LSPV","PPV_DIFF","ZC_DIFF","MPV_DIFF",
                             "MIPV_DIFF","LSPV_DIFF","PPV_MIX","ZC_MIX","MPV_MIX","MIPV_MIX","LSPV_MIX"]

            for k in range(len(pooling_op)):
                features = pooling_op[k]
                features_t = pooling_op_t[k]
                classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
                classifier.fit(features, y_train)
                y_pred = classifier.predict(features_t)   
                accuracy = accuracy_score(y_test, y_pred)
                matr_acc[i,30*k+j] = accuracy

        accuracy_tab = [np.mean(matr_acc[i,30*v:30*v+30]) for v in range(15)]
        all_datasets.append([dst] + accuracy_tab)
        with open("results_accuracy_mod_MINIROCKET_UCR.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset'] + pooling_names)
            writer.writerows(all_datasets)

        with open('matr_acc.npy', 'wb') as f:
            np.save(f,matr_acc)   

if __name__ == '__main__':
    datasets_UCR_Bench = np.loadtxt("list_UCR_datasets.txt",dtype="str")
    model_perf_minirocket(datasets_UCR_Bench)


    