#====================================================================================================#
#===                                                                                              ===#
#=== Mouhamadou Mansour Lo, Gildas Morvan, Mathieu Rossi, Fabrice Morganti, David Mercier         ===#
#===                                                                                              ===#
#=== Time series classification with random convolution kernels based transforms:                 ===#
#=== pooling operators and input representations matter                                           ===#
#===                                                                                              ===#
#=== https://arxiv.org/pdf/2409.01115                                                             ===#
#===                                                                                              ===#
#=== Source of SelF-Rocket.                                                                       ===#
#===                                                                                              ===#
#=== v2.1.0 - 01/27/2025 - Version of SelF-Rocket                                                 ===#
#===                                                                                              ===#
#===                                                                                              ===#
#===                                                                                              ===#
#===                                                                                              ===#
#===                                                                                              ===#
#====================================================================================================#


import random
import numpy as np

from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit,RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from features_generator import fit,transform
import warnings
# TODO : Remove this filter to have better error outputs
warnings.filterwarnings("ignore")

def highest_median_voting(ballots, candidates):
    scores = {candidate: [] for candidate in candidates}
    medians = {}
    
    for ballot in ballots:
        for candidate, score in ballot.items():
            if candidate in candidates:
                scores[candidate].append(score)

    for candidate, candidate_scores in scores.items():
        if candidate_scores:
            medians[candidate] = np.median(candidate_scores)
        else:
            medians[candidate] = 0 
            
    winner = max(medians, key=medians.get)
    return winner

def from_vect_to_ballots(pooling_names,vect_acc,vect_names):
    nb_comb = len(pooling_names)
    nb_vot = int(len(vect_acc)/nb_comb)
    ballots_vect_acc = [vect_acc[i*nb_comb:i*nb_comb+nb_comb]
                         for i in range(nb_vot)]
    ballots_vect_names = [vect_names[i*nb_comb:i*nb_comb+nb_comb]
                         for i in range(nb_vot)]
    bllts = []
    for i in range(nb_vot):
        keys = ballots_vect_names[i]
        values = ballots_vect_acc[i]
        res = dict(map(lambda i,j : (i,j) , keys,values))
        bllts.append(res)
    return bllts

def is_in_top_values(liste,comb_acc,top):
    rt_value = None
    sorted_list = sorted(liste,reverse=True)
    rt_value = 1 if comb_acc>= sorted_list[top-1] else 0
    return rt_value

def voting_system_threshold(pooling_names,vect_acc,idx_chosen_comb,idx_default,top,prct):
    idx = None
    nb_comb = len(pooling_names)
    nb_vot = int(len(vect_acc)/nb_comb)
    ballots_vect_acc = [vect_acc[i*nb_comb:i*nb_comb+nb_comb]
                         for i in range(nb_vot)]
    count_max = [is_in_top_values(ballots_vect_acc[j],
                ballots_vect_acc[j][idx_chosen_comb],top) for j in range(nb_vot)]
    idx = idx_chosen_comb if np.mean(count_max)*100 >= prct else idx_default
    return idx


class SelFRocket:

    def __init__(
        self,
        num_runs = 10,
        num_features_pc = 5000,
        only_MIX = False
    ): 
        self.name = "SelF-Rocket"
        self.num_features_pc =  num_features_pc
        self.num_runs = num_runs
        self.only_MIX = only_MIX
        self.mxszdst = 500
        self.num_kfold = 2
        self.num_features_pk = 5
        self.topvt = 5
        self.vot_threshold = 90
        self.classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
        self.scaler = StandardScaler()
        self.parameters1 = None
        self.parameters2 = None
        self.selected_comb_po = None
        self.selected_comb_num = -1

    def transform_sr(self,X,train=True):
        X = np.squeeze(X)
        X_diff = np.diff(X,1)
        if train == True:
            self.parameters1 = fit(X)
            self.parameters2 = fit(X_diff)
        X_transform = transform(X,X_diff,self.parameters1,self.parameters2,
                                n_features_per_kernel=self.num_features_pk)
        if train == True:
            self.scaler.fit(X_transform)
        X_transform = self.scaler.transform(X_transform)
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
        if len(y_train) >= self.mxszdst:
            skf = StratifiedShuffleSplit(n_splits=self.num_kfold*self.num_runs,
                                         train_size=int(self.mxszdst/2),test_size=int(self.mxszdst/2))
        else:
            skf = RepeatedStratifiedKFold(n_splits=self.num_kfold,n_repeats=self.num_runs)
        k_fold = skf.split(X_training_transform,y_train)
        nb_comb = len(pooling_names)
        vect_name_hmvs =  [None]*(self.num_runs*self.num_kfold*nb_comb)
        vect_acc_hmvs  = np.zeros((self.num_runs*self.num_kfold*nb_comb), dtype=np.float32)
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
                vect_acc_hmvs[l*nb_comb + k] = accuracy
                vect_name_hmvs[l*nb_comb + k] = pooling_names[k]
        return vect_acc_hmvs,vect_name_hmvs
    
    def fit(self,X_train,y_train):
        pooling_names = ["PPV","GMP","MPV","MIPV","LSPV","PPV_DIFF","GMP_DIFF","MPV_DIFF",
                        "MIPV_DIFF","LSPV_DIFF","PPV_MIX","GMP_MIX","MPV_MIX","MIPV_MIX","LSPV_MIX"]          
        pooling_op,X_training_transform = self.transform_sr(X_train)
        if self.only_MIX == True:
            idx_only_MIX = [10,11,12,13,14]
            pooling_names = [pooling_names[i] for i in idx_only_MIX]
            pooling_op = [pooling_op[i] for i in idx_only_MIX]
            self.topvt = 2
        pooling_names_to_num = dict(map(lambda i,j : (i,j) , pooling_names,range(0,len(pooling_names)))) 
        vect_acc_hmvs,vect_name_hmvs = self.features_selection(X_training_transform,y_train,
                                                               pooling_op,pooling_names)
        ballots_hmvs = from_vect_to_ballots(pooling_names,vect_acc_hmvs,vect_name_hmvs)
        po_max_hmvs = highest_median_voting(ballots_hmvs, pooling_names)
        ind_max_hmvs = pooling_names_to_num[po_max_hmvs]
        ind_def = pooling_names_to_num["PPV_MIX"]
        idx_fin = voting_system_threshold(pooling_names,vect_acc_hmvs,ind_max_hmvs,ind_def,
                                          self.topvt,self.vot_threshold)
        self.selected_comb_num = idx_fin
        self.selected_comb_po = pooling_names[idx_fin]
        features_train = pooling_op[idx_fin]
        self.classifier.fit(features_train,y_train)

    def predict(self,X_test):
        pooling_op_t = self.transform_sr(X_test,False)[0]
        if self.only_MIX == True:
            idx_only_MIX = [10,11,12,13,14]
            pooling_op_t = [pooling_op_t[i] for i in idx_only_MIX]
        features_test = pooling_op_t[self.selected_comb_num]
        yhat = self.classifier.predict(features_test)
        return yhat