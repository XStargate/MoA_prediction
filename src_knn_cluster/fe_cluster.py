import pandas as pd
from sklearn.cluster import KMeans

from config import Config

from pdb import set_trace

def fe_cluster_genes(train, test, test_p, genes, cells, n_clusters_g = 22, SEED = 42):
    
    features_g = genes
    #features_c = CELLS
    
    def create_cluster(train, test, test_p, features, kind = 'g', n_clusters = n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        test_p_ = test_p[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans_genes = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        # dump(kmeans_genes, open('kmeans_genes.pkl', 'wb'))
        train[f'clusters_{kind}'] = kmeans_genes.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans_genes.predict(test_.values)
        test_p[f'clusters_{kind}'] = kmeans_genes.predict(test_p_.values)
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        test_p = pd.get_dummies(test_p, columns = [f'clusters_{kind}'])
        return train, test, test_p
    
    train, test, test_p = create_cluster(train, test, test_p, features_g, kind = 'g', n_clusters = n_clusters_g)
    # train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test, test_p

def fe_cluster_cells(train, test, test_p, genes, cells, n_clusters_c = 4, SEED = 42):
    
    #features_g = GENES
    features_c = cells
    
    def create_cluster(train, test, test_p, features, kind = 'c', n_clusters = n_clusters_c):
        train_ = train[features].copy()
        test_ = test[features].copy()
        test_p_ = test_p[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans_cells = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        # dump(kmeans_cells, open('kmeans_cells.pkl', 'wb'))
        train[f'clusters_{kind}'] = kmeans_cells.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans_cells.predict(test_.values)
        test_p[f'clusters_{kind}'] = kmeans_cells.predict(test_p_.values)
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        test_p = pd.get_dummies(test_p, columns = [f'clusters_{kind}'])
        return train, test, test_p
    
   # train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
    train, test, test_p = create_cluster(train, test, test_p, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test, test_p

def fe_cluster_pca(train, test, test_p, n_clusters=5,SEED = 42):
    data=pd.concat([train,test],axis=0)
    kmeans_pca = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
    # dump(kmeans_pca, open('kmeans_pca.pkl', 'wb'))
    train[f'clusters_pca'] = kmeans_pca.predict(train.values)
    test[f'clusters_pca'] = kmeans_pca.predict(test.values)
    test_p[f'clusters_pca'] = kmeans_pca.predict(test_p.values)
    train = pd.get_dummies(train, columns = [f'clusters_pca'])
    test = pd.get_dummies(test, columns = [f'clusters_pca'])
    test_p = pd.get_dummies(test_p, columns = [f'clusters_pca'])
    return train, test, test_p

def fe_stats(train, test, test_p, genes, cells):
    
    features_g = genes
    features_c = cells
    
    for df in train, test, test_p:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        
        df['c52_c42'] = df['c-52'] * df['c-42']
        df['c13_c73'] = df['c-13'] * df['c-73']
        df['c26_c13'] = df['c-26'] * df['c-13']
        df['c33_c6'] = df['c-33'] * df['c-6']
        df['c11_c55'] = df['c-11'] * df['c-55']
        df['c38_c63'] = df['c-38'] * df['c-63']
        df['c38_c94'] = df['c-38'] * df['c-94']
        df['c13_c94'] = df['c-13'] * df['c-94']
        df['c4_c52'] = df['c-4'] * df['c-52']
        df['c4_c42'] = df['c-4'] * df['c-42']
        df['c13_c38'] = df['c-13'] * df['c-38']
        df['c55_c2'] = df['c-55'] * df['c-2']
        df['c55_c4'] = df['c-55'] * df['c-4']
        df['c4_c13'] = df['c-4'] * df['c-13']
        df['c82_c42'] = df['c-82'] * df['c-42']
        df['c66_c42'] = df['c-66'] * df['c-42']
        df['c6_c38'] = df['c-6'] * df['c-38']
        df['c2_c13'] = df['c-2'] * df['c-13']
        df['c62_c42'] = df['c-62'] * df['c-42']
        df['c90_c55'] = df['c-90'] * df['c-55']
        
        
        for feature in features_c:
             df[f'{feature}_squared'] = df[feature] ** 2     

        gsquarecols=['g-574','g-211','g-216','g-0','g-255',
                     'g-577','g-153','g-389','g-60','g-370',
                     'g-248','g-167','g-203','g-177','g-301',
                     'g-332','g-517','g-6','g-744','g-224',
                     'g-162','g-3','g-736','g-486','g-283',
                     'g-22','g-359','g-361','g-440','g-335',
                     'g-106','g-307','g-745','g-146','g-416',
                     'g-298','g-666','g-91','g-17','g-549',
                     'g-145','g-157','g-768','g-568','g-396']                
        for feature in gsquarecols:
            df[f'{feature}_squared'] = df[feature] ** 2        
        
    return train, test, test_p


def fe_cluster_all(train_features, test_features, test_features_private,
                   train_features2, test_features2, test_features_private2, 
                   train_pca, test_pca, test_pca_p):
    
    genes = [col for col in train_features2.columns if col.startswith('g-')]
    cells = [col for col in train_features2.columns if col.startswith('c-')]
    
    train_features2, test_features2, test_features_private2 =   \
        fe_cluster_genes(train_features2, test_features2, test_features_private2, genes, cells)
    train_features2, test_features2, test_features_private2 =   \
        fe_cluster_cells(train_features2, test_features2, test_features_private2, genes, cells)
    
    train_cluster_pca, test_cluster_pca, test_cluster_pca_p =   \
        fe_cluster_pca(train_pca, test_pca, test_pca_p)

    cfg = Config()
    train_cluster_pca = train_cluster_pca.iloc[:, (cfg.ncomp_g+cfg.ncomp_c):]
    test_cluster_pca = test_cluster_pca.iloc[:, (cfg.ncomp_g+cfg.ncomp_c):]
    test_cluster_pca_p = test_cluster_pca_p.iloc[:, (cfg.ncomp_g+cfg.ncomp_c):]
    
    features_cluster =  \
        [col for col in train_features2.columns if col.startswith('clusters_')]
    
    train_features_cluster = train_features2[features_cluster]
    test_features_cluster = test_features2[features_cluster]
    test_features_cluster_p = test_features_private2[features_cluster]
    
    train_features2, test_features2, test_features_private2 =  \
        fe_stats(train_features2, test_features2, test_features_private2, genes, cells)
        
    features_stats = ['g_sum', 'g_mean', 'g_std', 'g_kurt', 'g_skew', 'c_sum', 'c_mean', 
                      'c_std', 'c_kurt', 'c_skew', 'gc_sum', 'gc_mean', 'gc_std', 'gc_kurt', 
                      'gc_skew', 'c52_c42', 'c13_c73', 'c26_c13', 'c33_c6', 'c11_c55', 
                      'c38_c63', 'c38_c94', 'c13_c94', 'c4_c52', 'c4_c42', 'c13_c38', 
                      'c55_c2', 'c55_c4', 'c4_c13', 'c82_c42', 'c66_c42', 'c6_c38', 'c2_c13',
                      'c62_c42', 'c90_c55'] 
    features_stats += [col for col in train_features2.columns if col.endswith('_squared')]
        
    train_features_stats = train_features2[features_stats]
    test_features_stats = test_features2[features_stats]
    test_features_stats_p = test_features_private2[features_stats]
    
    train_features = pd.concat((train_features, train_features_cluster, train_cluster_pca,
                                train_features_stats), axis=1)
    test_features = pd.concat((test_features, test_features_cluster,test_cluster_pca,
                               test_features_stats), axis=1)
    test_features_p = pd.concat((test_features_private, test_features_cluster_p,test_cluster_pca_p,
                               test_features_stats_p), axis=1)
    
    return train_features, test_features, test_features_p