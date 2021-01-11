from sklearn.preprocessing import QuantileTransformer

def rankGauss(train_features, test_features):

    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    for col in (GENES + CELLS):

        transformer = QuantileTransformer(n_quantiles=100, random_state=0,
                                          output_distribution='normal')
        vec_len = len(train_features[col].values)
        vec_len_test = len(test_features[col].values)
        raw_vec = train_features[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)

        train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
        test_features[col] =  transformer.transform(
            test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]

    return train_features, test_features
