import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def feature_selection(x_train, x_test, feature_select="variancethreshold", variancethreshold_for_FS=0.8):

    if (feature_select == "variancethreshold"):
        print("VarianceThreshold")
        print("Shape of x_train before VarianceThreshold: {}".format(
            x_train.iloc[:, 4:].shape))
        variancethreshold = VarianceThreshold(variancethreshold_for_FS)
        train_num_selected = variancethreshold.fit_transform(x_train.iloc[:, 4:])
        cols = x_train.columns[4:]
        cols = cols[variancethreshold.get_support(indices=True)]
        train_num_selected = pd.DataFrame(train_num_selected, columns=cols)

        test_num_selected = variancethreshold.transform(x_test.iloc[:, 4:])
        test_num_selected = pd.DataFrame(test_num_selected, columns=cols)
        #   data_all.drop(columns=num_features, inplace=True)
        x_train = pd.concat(
            [x_train[["sig_id", "cp_type", "cp_time", "cp_dose"]], train_num_selected], axis=1)
        x_test = pd.concat(
            [x_test[["sig_id", "cp_type", "cp_time", "cp_dose"]], test_num_selected], axis=1)
        print("Shape of x_train after VarianceThreshold: {}".format(train_num_selected.shape))
    else:
        pass

    return x_train, x_test
