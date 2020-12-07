from sklearn.preprocessing import QuantileTransformer
import pandas as pd


def scale_norm(col):
    return (col - col.mean()) / col.std()


def scale_minmax(col):
    return (col - col.min()) / (col.max() - col.min())


def scaling(x_train, x_test, scale="rankgauss", n_quantiles=100, seed=42):

    genes_features = [column for column in x_train.columns if 'g-' in column]
    cells_features = [column for column in x_train.columns if 'c-' in column]
    num_features = [column for column in x_train.columns if column not in [
        "sig_id", "cp_type", "cp_time", "cp_dose", "drug_id"]]

    if scale == "norm":
        print("norm")
        x_train[num_features] = x_train[num_features].apply(scale_norm, axis=0)
        x_test[num_features] = x_test[num_features].apply(scale_norm, axis=0)

    elif scale == "minmax":
        print("minmax")
        x_train[num_features] = x_train[num_features].apply(scale_minmax, axis=0)
        x_test[num_features] = x_test[num_features].apply(scale_minmax, axis=0)

    elif scale == "rankgauss":
        # Rank Gauss
        print("Rank Gauss")
        for col in (genes_features + cells_features):

            transformer = QuantileTransformer(
                n_quantiles=n_quantiles, random_state=seed, output_distribution='normal')

            vec_len = len(x_train[col].values)
            vec_len_test = len(x_test[col].values)
            raw_vec = x_train[col].values.reshape(vec_len, 1)
            transformer.fit(raw_vec)

            x_train[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
            x_test[col] = transformer.transform(
                x_test[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]

    else:
        pass

    return x_train, x_test
