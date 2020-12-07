from sklearn.cluster import KMeans
import pandas as pd
import pickle as pk


def fe_cluster(x_train, x_test, genes_features, cells_features, n_cluster_g=22, n_cluster_c=4, seed=42, runty='traineval', path="./"):

    data_all = pd.concat([x_train, x_test], ignore_index=True)
    print("kMeans")

    cluster_genes = KMeans(n_clusters=n_cluster_g)
    cluster_cells = KMeans(n_clusters=n_cluster_c)

    if runty == 'traineval':
        print(f"Fitting for genes_features with {n_cluster_g} clusters")
        cluster_genes.fit(data_all[genes_features].values)
        print(f"Fitting for cells_features with {n_cluster_c} clusters")
        cluster_cells.fit(data_all[cells_features].values)

        pk.dump(cluster_genes, open(path+"cluster_genes.pkl", "wb"))
        pk.dump(cluster_cells, open(path+"cluster_cells.pkl", "wb"))
        print("Successfully saved cluster_genes.pkl and cluster_cells.pkl")

    elif runty == 'eval':
        cluster_genes = pk.load(open(path+"cluster_genes.pkl", 'rb'))
        cluster_cells = pk.load(open(path+"cluster_cells.pkl", 'rb'))
        print("Successfully loaded cluster_genes.pkl and cluster_cells.pkl")

    # transform on train set
    train_cluster_genes = cluster_genes.predict(x_train[genes_features].values)
    train_cluster_cells = cluster_cells.predict(x_train[cells_features].values)
    train_cluster_genes = pd.get_dummies(train_cluster_genes, columns=[
                                         'clusters_g'], prefix='clusters_g')
    train_cluster_cells = pd.get_dummies(train_cluster_cells, columns=[
                                         'clusters_c'], prefix='clusters_c')

    # transform on test set
    test_cluster_genes = cluster_genes.predict(x_test[genes_features].values)
    test_cluster_cells = cluster_cells.predict(x_test[cells_features].values)
    test_cluster_genes = pd.get_dummies(test_cluster_genes, columns=[
                                        'clusters_g'], prefix='clusters_g')
    test_cluster_cells = pd.get_dummies(test_cluster_cells, columns=[
                                        'clusters_c'], prefix='clusters_c')

    x_train = pd.concat([x_train, train_cluster_genes, train_cluster_cells], axis=1)
    x_test = pd.concat([x_test, test_cluster_genes, test_cluster_cells], axis=1)

    return x_train, x_test
