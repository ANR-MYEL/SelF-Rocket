#====================================================================================================
#===
#=== Mouhamadou Mansour Lo, Gildas Morvan, Mathieu Rossi, Fabrice Morganti, David Mercier
#===
#=== Time series classification with random convolution kernels based transforms: pooling operators and input representations matter
#===
#=== https://arxiv.org/pdf/2409.01115
#===
#=== Source of Hydra-SelFRocket.
#===
#=== v3.0.0 - 2024/12/10 - Corrected version of Hydra-SelFRocket
#===                       
#===                   
#===
#=== 
#=== 
#====================================================================================================


import random
import numpy as np
import torch

from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit,RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from hydra import Hydra,SparseScaler
from features_generator import fit,transform
from hm_voting_system import highest_median_voting
import warnings
# TODO : Remove this filter to have better error outputs
warnings.filterwarnings("ignore")

def from_vect_to_ballots(pooling_names,vect_acc,vect_names):
    num_elect = int(len(vect_acc)/len(pooling_names))
    ballots_vect_acc = [vect_acc[i*15:i*15+15] for i in range(num_elect)]
    ballots_vect_names = [vect_names[i*15:i*15+15] for i in range(num_elect)]
    bllts = []
    for i in range(num_elect):
        keys = ballots_vect_names[i]
        values = ballots_vect_acc[i]
        res = dict(map(lambda i,j : (i,j) , keys,values))
        bllts.append(res)
    return bllts

def is_max_value(liste,comb_acc,top_value):
    rt_value = None
    sorted_list = sorted(liste,reverse=True)
    rt_value = 1 if comb_acc>= sorted_list[top_value-1] else 0
    return rt_value

def voting_system_threshold(pooling_names,vect_acc,idx_chosen_comb,top,prct):
    idx = None
    num_elect = int(len(vect_acc)/len(pooling_names))
    ballots_vect_acc = [vect_acc[i*15:i*15+15] for i in range(num_elect)]
    count_max = [is_max_value(ballots_vect_acc[j],
                ballots_vect_acc[j][idx_chosen_comb],top) for j in range(num_elect)]
    idx = idx_chosen_comb if np.mean(count_max)*100 >= prct else 10
    return idx


class HydraSelFRocket:

    def __init__(
        self,
        num_runs = 10,
        num_kfold = 2
    ): 
        self.name = "Hydra-SelFRocket"
        self.num_features_pk = 5
        self.num_features_pc =  5000
        self.szmxdst = 500
        self.num_runs = num_runs
        self.num_kfold = num_kfold
        self.topvt = 5
        self.vot_threshold = 90
        self.classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
        self.scaler_sr = StandardScaler()
        self.scaler_h = SparseScaler()
        self.parameters1 = None
        self.parameters2 = None
        self.hydra = None
        self.selected_comb = -1

    def transform_h(self,X,train=True):
        X = np.squeeze(X)
        X_resh = np.float32(X)
        X_resh = X_resh.reshape((X.shape[0],1,X.shape[1]))
        X_torch = torch.from_numpy(X_resh)
        if train == True:
            self.hydra = Hydra(X_torch.shape[-1])
        X_transform_h = self.hydra.batch(X_torch)
        return X_transform_h

    def transform_sr(self,X,train=True):
        X = np.squeeze(X)
        X_diff = np.diff(X,1)
        if train == True:
            self.parameters1 = fit(X)
            self.parameters2 = fit(X_diff)
        X_transform = transform(X,X_diff,self.parameters1,self.parameters2,
                                n_features_per_kernel=self.num_features_pk)
        if train == True:
            self.scaler_sr.fit(X_transform)
        X_transform = self.scaler_sr.transform(X_transform)
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
        return pooling_op,X_transform
    
    def features_selection(self,X_training_transform,y_train,pooling_op,pooling_names):
        if len(y_train) >= self.szmxdst:
            skf = StratifiedShuffleSplit(n_splits=self.num_kfold*self.num_runs,
                                         train_size=int(self.szmxdst/2),test_size=int(self.szmxdst/2))
        else:
            skf = RepeatedStratifiedKFold(n_splits=self.num_kfold,n_repeats=self.num_runs)
        k_fold = skf.split(X_training_transform,y_train)
        compt = 0
        vect_name_kf =  [None]*(self.num_runs*self.num_kfold*15)
        vect_acc_hmvs  = np.zeros((self.num_runs*self.num_kfold*15), dtype=np.float32)
        for l,(train_index, test_index) in enumerate(k_fold):
            for k in range(len(pooling_op)):
                feature_idx = random.sample(range(0,pooling_op[k].shape[1]),self.num_features_pc)
                features = pooling_op[k][train_index][:,feature_idx]
                y_train_kfold = y_train[train_index]
                features_t  = pooling_op[k][test_index][:,feature_idx]
                y_test_kfold = y_train[test_index]
                classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
                classifier.fit(features, y_train_kfold)
                y_pred = classifier.predict(features_t)
                accuracy = accuracy_score(y_test_kfold, y_pred)
                vect_acc_hmvs[compt*15 + k] = accuracy
                vect_name_kf[compt*15 + k] = pooling_names[k]
            compt += 1
        ballots_hmvs = from_vect_to_ballots(pooling_names,vect_acc_hmvs,vect_name_kf)
        po_max_hmvs = highest_median_voting(ballots_hmvs, pooling_names)
        return po_max_hmvs,vect_acc_hmvs
    
    def fit(self,X_train,y_train):
        pooling_names = ["PPV","GMP","MPV","MIPV","LSPV","PPV_DIFF","GMP_DIFF","MPV_DIFF",
                        "MIPV_DIFF","LSPV_DIFF","PPV_MIX","GMP_MIX","MPV_MIX","MIPV_MIX","LSPV_MIX"]
        pooling_names_to_num = dict(map(lambda i,j : (i,j) , pooling_names,range(0,15)))
        pooling_op,X_training_transform = self.transform_sr(X_train)
        po_max_hmvs,vect_acc_hmvs = self.features_selection(X_training_transform,y_train,pooling_op,pooling_names)
        ind_max_hmvs = pooling_names_to_num[po_max_hmvs]
        idx_fin = voting_system_threshold(pooling_names,vect_acc_hmvs,ind_max_hmvs,self.topvt,self.vot_threshold)
        self.selected_comb = idx_fin
        features_train_sr = pooling_op[idx_fin]
        X_training_transform_h = self.transform_h(X_train)
        features_train = torch.cat((X_training_transform_h,torch.from_numpy(features_train_sr)),1)
        features_train = self.scaler_h.fit_transform(features_train)
        self.classifier.fit(features_train,y_train)

    def predict(self,X_test):
        pooling_op_t = self.transform_sr(X_test,False)[0]
        features_test_sr = pooling_op_t[self.selected_comb]
        X_test_transform_h = self.transform_h(X_test,False)
        features_test = torch.cat((X_test_transform_h,torch.from_numpy(features_test_sr)),1)
        features_test = self.scaler_h.transform(features_test)
        yhat = self.classifier.predict(features_test)
        return yhat




