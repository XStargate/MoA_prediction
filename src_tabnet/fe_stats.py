def fe_stats(train, test, genes_features, cells_features):

    print("Add statistics")

    gsquarecols = ['g-574', 'g-211', 'g-216', 'g-0', 'g-255', 'g-577', 'g-153', 'g-389', 'g-60', 'g-370', 'g-248', 'g-167', 'g-203', 'g-177', 'g-301', 'g-332', 'g-517', 'g-6', 'g-744', 'g-224', 'g-162', 'g-3',
                   'g-736', 'g-486', 'g-283', 'g-22', 'g-359', 'g-361', 'g-440', 'g-335', 'g-106', 'g-307', 'g-745', 'g-146', 'g-416', 'g-298', 'g-666', 'g-91', 'g-17', 'g-549', 'g-145', 'g-157', 'g-768', 'g-568', 'g-396']

    for df in train, test:
        df['g_sum'] = df[genes_features].sum(axis=1)
        df['g_mean'] = df[genes_features].mean(axis=1)
        df['g_std'] = df[genes_features].std(axis=1)
        df['g_kurt'] = df[genes_features].kurtosis(axis=1)
        df['g_skew'] = df[genes_features].skew(axis=1)
        df['c_sum'] = df[cells_features].sum(axis=1)
        df['c_mean'] = df[cells_features].mean(axis=1)
        df['c_std'] = df[cells_features].std(axis=1)
        df['c_kurt'] = df[cells_features].kurtosis(axis=1)
        df['c_skew'] = df[cells_features].skew(axis=1)
        df['gc_sum'] = df[genes_features + cells_features].sum(axis=1)
        df['gc_mean'] = df[genes_features + cells_features].mean(axis=1)
        df['gc_std'] = df[genes_features + cells_features].std(axis=1)
        df['gc_kurt'] = df[genes_features + cells_features].kurtosis(axis=1)
        df['gc_skew'] = df[genes_features + cells_features].skew(axis=1)

        # df['c52_c42'] = df['c-52'] * df['c-42']
        # df['c13_c73'] = df['c-13'] * df['c-73']
        # df['c26_c13'] = df['c-23'] * df['c-13']
        # df['c33_c6'] = df['c-33'] * df['c-6']
        # df['c11_c55'] = df['c-11'] * df['c-55']
        # df['c38_c63'] = df['c-38'] * df['c-63']
        # df['c38_c94'] = df['c-38'] * df['c-94']
        # df['c13_c94'] = df['c-13'] * df['c-94']
        # df['c4_c52'] = df['c-4'] * df['c-52']
        # df['c4_c42'] = df['c-4'] * df['c-42']
        # df['c13_c38'] = df['c-13'] * df['c-38']
        # df['c55_c2'] = df['c-55'] * df['c-2']
        # df['c55_c4'] = df['c-55'] * df['c-4']
        # df['c4_c13'] = df['c-4'] * df['c-13']
        # df['c82_c42'] = df['c-82'] * df['c-42']
        # df['c66_c42'] = df['c-66'] * df['c-42']
        # df['c6_c38'] = df['c-6'] * df['c-38']
        # df['c2_c13'] = df['c-2'] * df['c-13']
        # df['c62_c42'] = df['c-62'] * df['c-42']
        # df['c90_c55'] = df['c-90'] * df['c-55']
        #
        # for feature in cells_features:
        #     df[f'{feature}_squared'] = df[feature] ** 2
        #
        # for feature in gsquarecols:
        #     df[f'{feature}_squared'] = df[feature] ** 2

    return train, test
