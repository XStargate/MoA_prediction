import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing

from config import Config

def preprocessor(train, test, gs, cs):
    
    cfg = Config()
    
    # PCA
    
    n_gs = cfg.ncomp_g # No of PCA comps to include
    n_cs = cfg.ncomp_c # No of PCA comps to include

    # c-mean, g-mean
    
    train_c_mean = train[:,cs].mean(axis=1)
    test_c_mean = test[:,cs].mean(axis=1)
    train_g_mean = train[:,gs].mean(axis=1)
    test_g_mean = test[:,gs].mean(axis=1)
    
    # Scale train data
    scaler = preprocessing.StandardScaler()

    train = scaler.fit_transform(train)

    # Scale Test data
    test = scaler.transform(test)
    
    # PCA
    pca_cs = PCA(n_components = n_cs)
    pca_gs = PCA(n_components = n_gs)
    
    train_pca_gs = pca_gs.fit_transform(train[:,gs])
    train_pca_cs = pca_cs.fit_transform(train[:,cs])
    test_pca_gs = pca_gs.transform(test[:,gs])
    test_pca_cs = pca_cs.transform(test[:,cs])
    
    # Append Features
    
    train = np.concatenate((train,train_pca_gs,train_pca_cs,train_c_mean[:,np.newaxis]
                            ,train_g_mean[:,np.newaxis]),axis=1)
    test = np.concatenate((test,test_pca_gs,test_pca_cs,test_c_mean[:,np.newaxis],
                           test_g_mean[:,np.newaxis]),axis=1)
    
    return train, test

def preprocessor_2(train, test):
    scaler = preprocessing.StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    
    return train, test

def process_data(data):
    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
    return data