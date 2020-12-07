import os
import pandas as pd
from pickle import dump, load
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from config import Config

from pdb import set_trace

def _pca(train_features, test_features, runty, save_path, load_path, ncomp_g=463, ncomp_c=60):

    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    ### PCA on GENES
    pca_g = PCA(n_components=ncomp_g, random_state=42)
    data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
    if (runty == 'traineval'):
        gpca = (pca_g.fit(data[GENES]))
        dump(gpca, open(os.path.join(save_path, 'gpca.pkl'), 'wb'))
    elif (runty == 'eval'):
        gpca = load(open(os.path.join(load_path, 'gpca.pkl'), 'rb'))
    
    data2 = gpca.transform(data[GENES])
    
    train2 = data2[: train_features.shape[0]]
    test2 = data2[-test_features.shape[0]:]
    
    train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(ncomp_g)])
    test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(ncomp_g)])
    
    # drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)
    
    ### PCA on CELLS
    pca_c = PCA(n_components=ncomp_c, random_state=42)
    data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
    if (runty == 'traineval'):
        cpca = (pca_c.fit(data[CELLS]))
        dump(cpca, open(os.path.join(save_path, 'cpca.pkl'), 'wb'))
    elif (runty == 'eval'):
        cpca = load(open(os.path.join(load_path, 'cpca.pkl'), 'rb'))
    
    data2 = cpca.transform(data[CELLS])
    
    train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(ncomp_c)])
    test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(ncomp_c)])

    # drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)
        
    return train_features, test_features


def _pca_select (train_features, test_features, runty, 
                 save_path, load_path, variance_threshold=0.8):

    cfg = Config()
    
    train_features, test_features = _pca(
        train_features=train_features, test_features=test_features, runty=runty,
        save_path=save_path, load_path=load_path, ncomp_g=cfg.ncomp_g, 
        ncomp_c=cfg.ncomp_c)
    
    var_thresh = VarianceThreshold(variance_threshold)
    data = train_features.append(test_features)
    if (runty == 'traineval'):
        va_th = (var_thresh.fit(data.iloc[:, 4:]))
        dump(va_th, open(os.path.join(save_path, 'var_th.pkl'), 'wb'))
    elif (runty == 'eval'):
        va_th = load(open(os.path.join(load_path, 'var_th.pkl'), 'rb'))
    
    data_transformed = va_th.transform(data.iloc[:, 4:])

    train_features_transformed = data_transformed[ : train_features.shape[0]]
    test_features_transformed = data_transformed[-test_features.shape[0] : ]
    
    train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                  columns=['sig_id','cp_type','cp_time','cp_dose'])
    train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)
    test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                columns=['sig_id','cp_type','cp_time','cp_dose'])
    test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)
        
    return train_features, test_features

def process(train_features, test_features, train_targets_scored, 
            train_targets_nonscored, runty, save_path, load_path):
            
    train_features, test_features = _pca_select(train_features=train_features,
                                                test_features=test_features,
                                                runty=runty, save_path=save_path, 
                                                load_path=load_path)

    train = train_features.merge(train_targets_scored, on='sig_id')
    train = train.merge(train_targets_nonscored, on='sig_id')
    train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

    targets_scored = train[train_targets_scored.columns]
    targets_nonscored = train[train_targets_nonscored.columns]

    train = train.drop('cp_type', axis=1)
    test = test.drop('cp_type', axis=1)

    return train, test, targets_scored, targets_nonscored

def process_input2(train_features, test_features, preds):
    train = train_features[train_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

    return train[preds], test[preds]