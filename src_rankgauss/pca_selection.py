import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

def _pca(train_features, test_features, ncomp_g=463, ncomp_c=60):

    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    ### PCA on GENES
    data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
    data2 = (PCA(n_components=ncomp_g, random_state=42).fit_transform(data[GENES]))
    train2 = data2[: train_features.shape[0]]
    test2 = data2[-test_features.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(ncomp_g)])
    test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(ncomp_g)])

    # drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)

    ### PCA on CELLS
    data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
    data2 = (PCA(n_components=ncomp_c, random_state=42).fit_transform(data[CELLS]))
    train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(ncomp_c)])
    test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(ncomp_c)])

    # drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)

    return train_features, test_features

def _pca_select (train_features, test_features, variance_threshold=0.9):

    var_thresh = VarianceThreshold(variance_threshold)
    train_features, test_features = _pca(train_features=train_features, test_features=test_features)
    data = train_features.append(test_features)
    data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

    train_features_transformed = data_transformed[ : train_features.shape[0]]
    test_features_transformed = data_transformed[-test_features.shape[0] : ]


    train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                  columns=['sig_id','cp_type','cp_time','cp_dose'])

    train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)


    test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                 columns=['sig_id','cp_type','cp_time','cp_dose'])

    test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

    return train_features, test_features


def process(train_features, test_features, train_targets_scored):
    train_features, test_features = _pca_select(train_features=train_features,
                                                test_features=test_features)

    train = train_features.merge(train_targets_scored, on='sig_id')
    train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

    target = train[train_targets_scored.columns]

    train = train.drop('cp_type', axis=1)
    test = test.drop('cp_type', axis=1)

    return train, test, target
