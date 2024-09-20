#====================================================================================================
#===
#=== Mouhamadou Mansour Lo, Gildas Morvan, Mathieu Rossi, Fabrice Morganti, David Mercier
#===
#=== Time series classification with random convolution kernels based transforms: pooling operators and input representations matter
#===
#=== https://arxiv.org/pdf/2409.01115
#===
#=== section 03 Experiments
#=== 
#=== 
#====================================================================================================

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

            parameters1 = fit(X_train)
            parameters2 = fit(X_train_diff)

            X_training_transform = transform(X_train,X_train_diff,parameters1,parameters2,n_features_per_kernel=5)
            
            scaler = preprocessing.StandardScaler().fit(X_training_transform)
            X_training_transform = scaler.transform(X_training_transform)
            
            X_test_transform = transform(X_test,X_test_diff,parameters1,parameters2,n_features_per_kernel=5)
            X_test_transform = scaler.transform(X_test_transform)

            ppv = X_training_transform[:,0:9996]
            lspv = X_training_transform[:,9996:19992]
            mpv = X_training_transform[:,19992:29988]
            mipv = X_training_transform[:,29988:39984]
            gmp = X_training_transform[:,39984:49980]
            ppv_diff = X_training_transform[:,49980:59976]
            lspv_diff = X_training_transform[:,59976:69972]
            mpv_diff = X_training_transform[:,69972:79968]
            mipv_diff = X_training_transform[:,79968:89964]
            gmp_diff = X_training_transform[:,89964:99960]
            ppv_mix = np.concatenate((ppv, ppv_diff), axis=1)
            lspv_mix = np.concatenate((lspv, lspv_diff), axis=1)
            mpv_mix = np.concatenate((mpv, mpv_diff), axis=1)
            mipv_mix = np.concatenate((mipv, mipv_diff), axis=1)
            gmp_mix = np.concatenate((gmp, gmp_diff), axis=1)

            ppv_t = X_test_transform[:,0:9996]
            lspv_t = X_test_transform[:,9996:19992]
            mpv_t = X_test_transform[:,19992:29988]
            mipv_t = X_test_transform[:,29988:39984]
            gmp_t = X_test_transform[:,39984:49980]
            ppv_diff_t = X_test_transform[:,49980:59976]
            lspv_diff_t = X_test_transform[:,59976:69972]
            mpv_diff_t = X_test_transform[:,69972:79968]
            mipv_diff_t = X_test_transform[:,79968:89964]
            gmp_diff_t = X_test_transform[:,89964:99960]
            ppv_mix_t = np.concatenate((ppv_t, ppv_diff_t), axis=1)
            lspv_mix_t = np.concatenate((lspv_t, lspv_diff_t), axis=1)
            mpv_mix_t = np.concatenate((mpv_t, mpv_diff_t), axis=1)
            mipv_mix_t = np.concatenate((mipv_t, mipv_diff_t), axis=1)
            gmp_mix_t = np.concatenate((gmp_t, gmp_diff_t), axis=1)
            
            pooling_op = [ppv,gmp,mpv,mipv,lspv,ppv_diff,gmp_diff,mpv_diff,
                          mipv_diff,lspv_diff,ppv_mix,gmp_mix,mpv_mix,mipv_mix,lspv_mix]
            pooling_op_t = [ppv_t,gmp_t,mpv_t,mipv_t,lspv_t,ppv_diff_t,gmp_diff_t,mpv_diff_t,
                          mipv_diff_t,lspv_diff_t,ppv_mix_t,gmp_mix_t,mpv_mix_t,mipv_mix_t,lspv_mix_t]
            pooling_names = ["PPV","GMP","MPV","MIPV","LSPV","PPV_DIFF","GMP_DIFF","MPV_DIFF",
                             "MIPV_DIFF","LSPV_DIFF","PPV_MIX","GMP_MIX","MPV_MIX","MIPV_MIX","LSPV_MIX"]

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


    