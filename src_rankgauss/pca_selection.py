import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from config import Config

from pdb import set_trace

def _pca(train_features, test_features, runty, ncomp_g=463, ncomp_c=60, test_features_private=None):

    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    ### PCA on GENES
    data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
    # data2 = (PCA(n_components=ncomp_g, random_state=42).fit_transform(data[GENES]))
    pca_func = PCA(n_components=ncomp_g, random_state=42)
    pca_func.fit(data[GENES])
    data2 = pca_func.transform(data[GENES])
    
    train2 = data2[: train_features.shape[0]]
    test2 = data2[-test_features.shape[0]:]
    
    train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(ncomp_g)])
    test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(ncomp_g)])
    
    # drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)
    
    if (runty == 'eval'):
        data_p = pd.concat([pd.DataFrame(train_features[GENES]), 
                            pd.DataFrame(test_features_private[GENES])])
        data2_p = pca_func.transform(data_p[GENES])
        
        train2_p = data2_p[: train_features.shape[0]]
        test2_p = data2_p[-test_features_private.shape[0]:]
        
        train2_p = pd.DataFrame(train2_p, columns=[f'pca_G-{i}' for i in range(ncomp_g)])
        test2_p = pd.DataFrame(test2_p, columns=[f'pca_G-{i}' for i in range(ncomp_g)])
        
        # train_features = pd.concat((train_features, train2_p), axis=1)
        test_features_private = pd.concat((test_features_private, test2_p), axis=1)
    
    ### PCA on CELLS
    data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
    # data2 = (PCA(n_components=ncomp_c, random_state=42).fit_transform(data[CELLS]))
    pca_func = PCA(n_components=ncomp_c, random_state=42)
    pca_func.fit(data[CELLS])
    data2 = pca_func.transform(data[CELLS])
    
    train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(ncomp_c)])
    test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(ncomp_c)])

    # drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)
    
    if (runty == 'eval'):
        data_p = pd.concat([pd.DataFrame(train_features[CELLS]), 
                            pd.DataFrame(test_features_private[CELLS])])
        data2_p = pca_func.transform(data_p[CELLS])
        
        train2_p = data2_p[: train_features.shape[0]]
        test2_p = data2_p[-test_features_private.shape[0]:]
        
        train2_p = pd.DataFrame(train2_p, columns=[f'pca_C-{i}' for i in range(ncomp_c)])
        test2_p = pd.DataFrame(test2_p, columns=[f'pca_C-{i}' for i in range(ncomp_c)])
        
        # train_features = pd.concat((train_features, train2_p), axis=1)
        test_features_private = pd.concat((test_features_private, test2_p), axis=1)
        
    return train_features, test_features, test_features_private


def _pca_select (train_features, test_features, runty, test_features_private, variance_threshold=0.9):

    cfg = Config()
    
    var_thresh = VarianceThreshold(variance_threshold)
    train_features, test_features, test_features_private = _pca(
        train_features=train_features, test_features=test_features, runty=runty,
        ncomp_g=cfg.ncomp_g, ncomp_c=cfg.ncomp_c, test_features_private=test_features_private)
    
    data = train_features.append(test_features)
    # data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])
    var_thresh.fit(data.iloc[:, 4:])
    data_transformed = var_thresh.transform(data.iloc[:, 4:])

    train_features_transformed = data_transformed[ : train_features.shape[0]]
    test_features_transformed = data_transformed[-test_features.shape[0] : ]
    
    if (runty == 'eval'):
        data_p = train_features.append(test_features_private)
        data_transformed_p = var_thresh.transform(data_p.iloc[:, 4:])
        
        train_features_transformed_p = data_transformed_p[ : train_features.shape[0]]
        test_features_transformed_p = data_transformed_p[-test_features_private.shape[0] : ]

    train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                  columns=['sig_id','cp_type','cp_time','cp_dose'])

    if (runty == 'traineval'):
        train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)
        test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                    columns=['sig_id','cp_type','cp_time','cp_dose'])
        test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)
        
        return train_features, test_features
    
    elif (runty == 'eval'):
        train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed_p)], axis=1)
        test_features_private = pd.DataFrame(test_features_private[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                    columns=['sig_id','cp_type','cp_time','cp_dose'])
        test_features_private = pd.concat([test_features_private, pd.DataFrame(test_features_transformed_p)], axis=1)
        
        return train_features, test_features_private


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

    train = train_features.merge(train_targets_scored, on='sig_id')
    # train = train.merge(train_targets_nonscored, on='sig_id')
    train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

    targets_scored = train[train_targets_scored.columns]
    # targets_nonscored = train[train_targets_nonscored.columns]

    train = train.drop('cp_type', axis=1)
    test = test.drop('cp_type', axis=1)

    return train, test, targets_scored
