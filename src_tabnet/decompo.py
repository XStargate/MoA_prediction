import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import pickle as pk


def decompo_process(x_train, x_test, decompo="PCA", genes_variance=0.85, cells_variance=0.9, seed=42, pca_drop_orig=False, runty='traineval', path="./"):

    genes_features = [column for column in x_train.columns if 'g-' in column]
    cells_features = [column for column in x_train.columns if 'c-' in column]

    data_all = pd.concat([x_train, x_test], ignore_index=True)

    if decompo == "PCA":
        print("PCA")

        # set pca decomposition function
        pca_genes = PCA(n_components=genes_variance, random_state=seed)
        pca_cells = PCA(n_components=cells_variance, random_state=seed)

        if runty == 'traineval':
            pca_genes.fit(data_all[genes_features])
            pca_cells.fit(data_all[cells_features])
            pk.dump(pca_genes, open(path+"pca_genes.pkl", "wb"))
            pk.dump(pca_cells, open(path+"pca_cells.pkl", "wb"))
            print("Successfully saved pca_genes.pkl and pca_genes.pkl")

        elif runty == 'eval':
            pca_genes = pk.load(open(path+"pca_genes.pkl", 'rb'))
            pca_cells = pk.load(open(path+"pca_cells.pkl", 'rb'))
            print("Successfully loaded pca_genes.pkl and pca_genes.pkl")

        # transform on train set
        train_pca_genes = pca_genes.transform(x_train[genes_features])
        train_pca_cells = pca_cells.transform(x_train[cells_features])

        # transform on test set
        test_pca_genes = pca_genes.transform(x_test[genes_features])
        test_pca_cells = pca_cells.transform(x_test[cells_features])

        train_pca_genes = pd.DataFrame(train_pca_genes, columns=[
                                       f"pca_g-{i}" for i in range(train_pca_genes.shape[1])])
        train_pca_cells = pd.DataFrame(train_pca_cells, columns=[
                                       f"pca_c-{i}" for i in range(train_pca_cells.shape[1])])

        test_pca_genes = pd.DataFrame(test_pca_genes, columns=[
                                      f"pca_g-{i}" for i in range(test_pca_genes.shape[1])])
        test_pca_cells = pd.DataFrame(test_pca_cells, columns=[
                                      f"pca_c-{i}" for i in range(test_pca_cells.shape[1])])

        if pca_drop_orig:
            x_train = pd.concat([x_train[["sig_id", "cp_type", "cp_time", "cp_dose"]],
                                 train_pca_genes, train_pca_cells], axis=1)
            x_test = pd.concat([x_test[["sig_id", "cp_type", "cp_time", "cp_dose"]],
                                test_pca_genes, test_pca_cells], axis=1)
        else:
            x_train = pd.concat([x_train, train_pca_genes, train_pca_cells], axis=1)
            x_test = pd.concat([x_test, test_pca_genes, test_pca_cells], axis=1)

        print("Number of x_train PCA components in gen_features is: {}, explained variance is: {}".format(
            train_pca_genes.shape[1], genes_variance))
        print("Number of x_train PCA components in cell_features is: {}, explained variance is: {}".format(
            train_pca_cells.shape[1], cells_variance))
        print("Number of x_test PCA components in gen_features is: {}, explained variance is: {}".format(
            test_pca_genes.shape[1], genes_variance))
        print("Number of x_test PCA components in cell_features is: {}, explained variance is: {}".format(
            test_pca_cells.shape[1], cells_variance))
    else:
        pass

    return x_train, x_test


def decompo_process_kfold(train_df, valid_df, x_test,
                          decompo="PCA",
                          genes_variance=0.85,
                          cells_variance=0.9,
                          seed=42,
                          variancethreshold_for_FS=0.8,
                          iseed=0,
                          fold=0,
                          runty='traineval',
                          path="./"):

    genes_features = [column for column in train_df.columns if column.startswith('g-')]
    cells_features = [column for column in train_df.columns if column.startswith('c-')]

    data_all = pd.concat([train_df, valid_df, x_test], ignore_index=True)

    if decompo == "PCA":
        print(f"PCA on seed {iseed}, FOLD {fold+1}...")

        # set pca decomposition function
        pca_genes = PCA(n_components=genes_variance, random_state=seed)
        pca_cells = PCA(n_components=cells_variance, random_state=seed)

        if runty == 'traineval':
            pca_genes.fit(data_all[genes_features])
            pca_cells.fit(data_all[cells_features])
            pk.dump(pca_genes, open(path+f"pca_genes_seed{iseed}_FOLD{fold}.pkl", "wb"))
            pk.dump(pca_cells, open(path+f"pca_cells_seed{iseed}_FOLD{fold}.pkl", "wb"))
            print(
                f"Successfully saved pca_genes_seed{iseed}_FOLD{fold}.pkl and pca_cells_seed{iseed}_FOLD{fold}.pkl")

        elif runty == 'eval':
            pca_genes = pk.load(open(path+f"pca_genes_seed{iseed}_FOLD{fold}.pkl", 'rb'))
            pca_cells = pk.load(open(path+f"pca_cells_seed{iseed}_FOLD{fold}.pkl", 'rb'))
            print(
                f"Successfully loaded pca_genes_seed{iseed}_FOLD{fold}.pkl and pca_cells_seed{iseed}_FOLD{fold}.pkl")

        # transform on train set
        train_df_pca_genes = pca_genes.transform(train_df[genes_features])
        train_df_pca_cells = pca_cells.transform(train_df[cells_features])
        valid_df_pca_genes = pca_genes.transform(valid_df[genes_features])
        valid_df_pca_cells = pca_cells.transform(valid_df[cells_features])

        # transform on test set
        test_pca_genes = pca_genes.transform(x_test[genes_features])
        test_pca_cells = pca_cells.transform(x_test[cells_features])

        train_df_pca_genes = pd.DataFrame(train_df_pca_genes, columns=[
            f"pca_g-{i}" for i in range(train_df_pca_genes.shape[1])])
        train_df_pca_cells = pd.DataFrame(train_df_pca_cells, columns=[
            f"pca_c-{i}" for i in range(train_df_pca_cells.shape[1])])
        train_df_pca = pd.concat([train_df_pca_genes, train_df_pca_cells], axis=1)

        valid_df_pca_genes = pd.DataFrame(valid_df_pca_genes, columns=[
            f"pca_g-{i}" for i in range(train_df_pca_genes.shape[1])])
        valid_df_pca_cells = pd.DataFrame(valid_df_pca_cells, columns=[
            f"pca_c-{i}" for i in range(train_df_pca_cells.shape[1])])
        valid_df_pca = pd.concat([valid_df_pca_genes, valid_df_pca_cells], axis=1)

        test_pca_genes = pd.DataFrame(test_pca_genes, columns=[
                                      f"pca_g-{i}" for i in range(test_pca_genes.shape[1])])
        test_pca_cells = pd.DataFrame(test_pca_cells, columns=[
                                      f"pca_c-{i}" for i in range(test_pca_cells.shape[1])])
        test_pca = pd.concat([test_pca_genes, test_pca_cells], axis=1)

        # feature selection
        pca_features = train_df_pca.columns
        # print(pca_features)
        # print(len(pca_features))
        variancethreshold = VarianceThreshold(variancethreshold_for_FS)
        train_fe_select = variancethreshold.fit_transform(train_df_pca)
        # print(variancethreshold.get_support(indices=True))
        pca_features = pca_features[variancethreshold.get_support(indices=True)]
        train_fe_select = pd.DataFrame(train_fe_select, columns=pca_features)

        valid_fe_select = variancethreshold.transform(valid_df_pca)
        valid_fe_select = pd.DataFrame(valid_fe_select, columns=pca_features)

        test_fe_select = variancethreshold.transform(test_pca)
        test_fe_select = pd.DataFrame(test_fe_select, columns=pca_features)

        x_train = pd.concat([train_df, train_fe_select], axis=1)
        x_valid = pd.concat([valid_df, valid_fe_select], axis=1)
        x_test = pd.concat([x_test, test_fe_select], axis=1)

        # print("Number of train_df PCA components in gen_features is: {}, explained variance is: {}".format(
        #     train_df_pca_genes.shape[1], genes_variance))
        # print("Number of train_df PCA components in cell_features is: {}, explained variance is: {}".format(
        #     train_df_pca_cells.shape[1], cells_variance))
        # print("Number of valid_df PCA components in gen_features is: {}, explained variance is: {}".format(
        #     valid_df_pca_genes.shape[1], genes_variance))
        # print("Number of valid_df PCA components in cell_features is: {}, explained variance is: {}".format(
        #     valid_df_pca_cells.shape[1], cells_variance))
        # print("Number of x_test PCA components in gen_features is: {}, explained variance is: {}".format(
        #     test_pca_genes.shape[1], genes_variance))
        # print("Number of x_test PCA components in cell_features is: {}, explained variance is: {}".format(
        #     test_pca_cells.shape[1], cells_variance))
        print("Number of all pcas in x_train, x_valid, x_test are {}".format(len(pca_features)))
    else:
        pass

    return x_train, x_valid, x_test, pca_features
