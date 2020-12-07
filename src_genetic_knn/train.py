import cuml
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from config import Config

class train_test():
    def __init__(self, train, targets, test):
        self.train = train
        self.targets = targets
        self.test = test
        self.cfg = Config()

    def genetic_search(self):
    
        oof = np.zeros((len(self.train), self.targets.shape[1]-2))
        dna = np.random.uniform(0,1,(self.cfg.population,875))**2.0
        cvs = np.zeros((self.cfg.population))
    
        for jj in range(self.cfg.generations):
        
            # all previous population "DNA" and cv scores
            df = pd.DataFrame(data=dna)
            df['cv'] = cvs
            df.sort_values('cv', inplace=True)
            print('Evolving...')
        
            # generate and evaluate children
            for k in range(self.cfg.population):
                print(k, ', ', end='')
            
                # generate child
                if jj!= 0:
                    parent1 = k//self.cfg.parents; parent2 = k%self.cfg.parents
                    TMP = np.random.randint(0,2,875)
                    dna[k,] = TMP * df.iloc[parent1,:-1] + (1-TMP) * df.iloc[parent2,:-1]
                    x = np.random.uniform(0,1,875)
                    IDX = np.where(x<self.cfg.mutate)[0]
                    dna[k,IDX] = np.random.uniform(0,1,len(IDX))**2.0
                else:
                    dna[k,] = df.iloc[k,:-1]
                
                # knn weights
                WGT = dna[k,]
                # weights for cp_type, cp_time, cp_dose
                WGT[0]= 100 ; WGT[1] = 12/2; WGT[2] = 5/2
            
                # knn kfold validate
                for fold in range(self.cfg.folds):
                    model = cuml.neighbors.KNeighborsClassifier(n_neighbors=1000)
                    model.fit(self.train.loc[self.train.fold!=fold, self.train.columns[1:-1] ].values * WGT,
                              self.targets.loc[self.targets.fold!=fold, self.targets.columns[1:-1] ] )
    
                    pp = model.predict_proba(self.train.loc[self.train.fold==fold, self.train.columns[1:-1] ].values * WGT )
                    pp = np.stack( [(1 - pp[x][:,0]) for x in range(len(pp))] ).T
                    oof[self.targets.fold==fold,] = pp
                
                cv_score = log_loss(self.targets.iloc[:,1:-1].values.flatten(), oof.flatten() )
                cvs[k] = cv_score
            
        return oof
    
    def predict_test(self):
        
        WGT = df.iloc[0,1:].values
        oof = np.zeros((len(self.train),self.targets.shape[1]-2))
        preds = np.zeros((len(self.test),self.targets.shape[1]-2))
        
        for fold in range(self.cfg.folds):
            print('FOLD %i'%(fold+1), ' ', end='')
            
            model = cuml.neighbors.KNeighborsClassifier(n_neighbors=1000)
            model.fit(self.train.loc[self.train.fold!=fold, self.train.columns[1:-1]].values * WGT,
                      self.targets.loc[self.targets.fold!=fold, self.targets.columns[1:-1]])
            
            pp = model.predict_proba(self.train.loc[self.train.fold==fold, self.train.columns[1:-1] ].values * WGT )
            pp = np.stack( [(1 - pp[x][:,0]) for x in range(len(pp))] ).T
            oof[self.targets.fold==fold,] = pp
            
            pp = model.predict_proba(self.test[self.test.columns[1:]].values * WGT )
            pp = np.stack( [(1 - pp[x][:,0]) for x in range(len(pp))] ).T
            preds += pp/self.cfg.folds
            
        print()
        cv_score = log_loss(self.targets.iloc[:,1:-1].values.flatten(), oof.flatten() )
        print('CV SCORE = %.5f'%cv_score)
        
        return preds
        