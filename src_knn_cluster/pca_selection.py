import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from config import Config

from pdb import set_trace

def _pca(train_features, test_features, runty, ncomp_g=600, ncomp_c=50, test_features_private=None):

    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    ### PCA on GENES
    data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
    # data2 = (PCA(n_components=ncomp_g, random_state=42).fit_transform(data[GENES]))
    pca_g = PCA(n_components=ncomp_g, random_state=42)
    pca_g.fit(data[GENES])
    
    data2 = pca_g.transform(data[GENES])
    
    train2 = data2[: train_features.shape[0]]
    test2 = data2[-test_features.shape[0]:]
    
    train_gpca = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(ncomp_g)])
    test_gpca = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(ncomp_g)])
    
    # drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
    train_features = pd.concat((train_features, train_gpca), axis=1)
    test_features = pd.concat((test_features, test_gpca), axis=1)
    
    data_p = pd.concat([pd.DataFrame(train_features[GENES]), 
                        pd.DataFrame(test_features_private[GENES])])
    data2_p = pca_g.transform(data_p[GENES])
        
    train2_p = data2_p[: train_features.shape[0]]
    test2_p = data2_p[-test_features_private.shape[0]:]
        
    train_gpca_p = pd.DataFrame(train2_p, columns=[f'pca_G-{i}' for i in range(ncomp_g)])
    test_gpca_p = pd.DataFrame(test2_p, columns=[f'pca_G-{i}' for i in range(ncomp_g)])
        
    # train_features = pd.concat((train_features, train2_p), axis=1)
    test_features_private = pd.concat((test_features_private, test_gpca_p), axis=1)
    
    ### PCA on CELLS
    data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
    # data2 = (PCA(n_components=ncomp_c, random_state=42).fit_transform(data[CELLS]))
    pca_c = PCA(n_components=ncomp_c, random_state=42)
    pca_c.fit(data[CELLS])
    
    data2 = pca_c.transform(data[CELLS])
    
    train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

    train_cpca = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(ncomp_c)])
    test_cpca = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(ncomp_c)])

    # drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
    train_features = pd.concat((train_features, train_cpca), axis=1)
    test_features = pd.concat((test_features, test_cpca), axis=1)
    
    train_pca = pd.concat((train_gpca, train_cpca), axis=1)
    test_pca = pd.concat((test_gpca, test_cpca), axis=1)
        
    data_p = pd.concat([pd.DataFrame(train_features[CELLS]), 
                        pd.DataFrame(test_features_private[CELLS])])
    data2_p = pca_c.transform(data_p[CELLS])
        
    train2_p = data2_p[: train_features.shape[0]]
    test2_p = data2_p[-test_features_private.shape[0]:]
        
    train_cpca_p = pd.DataFrame(train2_p, columns=[f'pca_C-{i}' for i in range(ncomp_c)])
    test_cpca_p = pd.DataFrame(test2_p, columns=[f'pca_C-{i}' for i in range(ncomp_c)])
        
    # train_features = pd.concat((train_features, train2_p), axis=1)
    test_features_private = pd.concat((test_features_private, test_cpca_p), axis=1)
        
    train_pca_p = pd.concat((train_gpca_p, train_cpca_p), axis=1)
    test_pca_p = pd.concat((test_gpca_p, test_cpca_p), axis=1)
        
    return train_features, test_features, test_features_private,   \
        train_pca, test_pca, test_pca_p


def _pca_select (train_features, test_features, test_features_private, variance_threshold=0.85):

    cfg = Config()
    
    c_n = [f for f in list(train_features.columns) if f not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
    data = train_features.append(test_features)
    mask = (data[c_n].var() >= 0.85).values
    # mask = (train_features[c_n].var() >= variance_threshold).values
    tmp = train_features[c_n].loc[:, mask]
    train_features = pd.concat([train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)
    tmp = test_features[c_n].loc[:, mask]
    test_features = pd.concat([test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)
    tmp = test_features_private[c_n].loc[:, mask]
    test_features_private = pd.concat([test_features_private[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)
    
    return train_features, test_features, test_features_private

def process(train_features, test_features, train_targets_scored, train_targets_nonscored, 
            test_features_private, runty):
    
    if (runty == 'traineval'):
        assert test_features_private is None,  \
            "Error: test_features_private should be None when run type is 'traineval'"
    elif (runty == 'eval'):
        assert test_features_private is not None,  \
            "Error: test_features_private should not be None when run type is 'eval'"

    train_features, test_features = _pca_select(train_features=train_features,
                                                test_features=test_features,
                                                runty=runty,
                                                test_features_private=test_features_private)

    # train_features = train_features.merge(train_targets_nonscored, on='sig_id')
    train = train_features.merge(train_targets_scored, on='sig_id')
    train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

    targets_scored = train[train_targets_scored.columns]
    # targets_nonscored = train[train_targets_nonscored.columns]

    train = train.drop('cp_type', axis=1)
    test = test.drop('cp_type', axis=1)

    return train, test, targets_scored
