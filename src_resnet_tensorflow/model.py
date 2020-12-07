import os
import numpy as np
from tensorflow.keras import (callbacks, layers, Sequential, 
                              Model, models, losses, optimizers)
from tensorflow.random import set_seed

from config import Config
from loss import logloss
from utils import process_data

from pdb import set_trace


class train_test():
    def __init__(self, x_train, x_valid, data_test, x_train_2, x_valid_2,
                 data_test_2, y_train_sc, y_train_ns, y_valid_sc, y_valid_ns,
                 save_path, load_path, fold, runty='traineval'):
        self.x_train = x_train
        self.x_valid = x_valid
        self.data_test = data_test
        self.x_train_2 = x_train_2
        self.x_valid_2 = x_valid_2
        self.data_test_2 = data_test_2
        self.y_train_sc = y_train_sc
        self.y_train_ns = y_train_ns
        self.y_valid_sc = y_valid_sc
        self.y_valid_ns = y_valid_ns
        
        self.save_path = save_path
        self.load_path = load_path
        self.fold = fold
        self.runty = runty
        
        self.cfg = Config()

    def _freeze_unfreeze(self, model, layer=13, runty='freeze'):
        if runty == 'freeze':
            for layer in model.layers[:layer]:
                layer.trainable = False
        elif runty == 'unfreeze':
            for layer in model.layers:
                layer.trainable = True
            
        return model
    
    def run_training(self, seed):
    
        x1_train = self.x_train
        x2_train = self.x_train_2
        y_sc_train = self.y_train_sc
        y_ns_train = self.y_train_ns
        x1_valid = self.x_valid
        x2_valid = self.x_valid_2
        y_sc_valid = self.y_valid_sc
        y_ns_valid = self.y_valid_ns
        fold = self.fold
        
        oof = np.zeros((len(x1_train)+len(x2_valid), y_sc_train.shape[1]))
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_logloss', min_delta=1E-5, patience=self.cfg.patience_ns,
            mode='auto', restore_best_weights=True)
        check_point = callbacks.ModelCheckpoint(
            os.path.join(self.save_path, f'weights_seed{seed}_fold{fold}'),
            save_best_only=True, save_weights_only=True,
            verbose=0, mode='auto')
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_logloss', factor=0.5, patience=5, verbose=0,
            mode='auto', min_lr=1E-5)
    
        input1_ = layers.Input(shape=(self.x_train.shape[1]))
        input2_ = layers.Input(shape=(self.x_train_2.shape[1]))
    
        output1_layer = Sequential([
            layers.BatchNormalization(),
            layers.Dropout(0.2), 
            layers.Dense(512, activation="elu"),
            layers.BatchNormalization(),
            layers.Dense(256, activation="elu"),
        ], name='output1_layer')
    
        output1 = output1_layer(input1_)
    
        answer1_layer = Sequential([
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu')
        ], name='answer1_layer')
    
        answer1_input = layers.Concatenate()([output1, input2_])
        answer1 = answer1_layer(answer1_input)
    
        answer2_layer = Sequential([
            layers.BatchNormalization(),
            layers.Dense(512, activation='elu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu')
        ], name='answer2_layer')
    
        answer2_input = layers.Concatenate()([output1, input2_, answer1])
        answer2 = answer2_layer(answer2_input)
    
        answer3_layer = Sequential([
            layers.BatchNormalization(),
            layers.Dense(256, activation='elu')
        ], name='answer3_layer')
    
        answer3_input = layers.Concatenate()([answer1, answer2])
        answer3 = answer3_layer(answer3_input)
    
        answer3_layer_ = Sequential([
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu')
        ], name='answer3_layer_')
    
        answer3_input_ = layers.Concatenate()([answer1, answer2, answer3])
        answer3_ = answer3_layer_(answer3_input_)
    
        answer4_layer_ = Sequential([
            layers.BatchNormalization(),
            layers.Dense(256, kernel_initializer='lecun_normal', activation='selu', name='last_frozen'),
        ], name='answer4_layer_')
    
        answer4_input_ = layers.Concatenate()([output1, answer2, answer3, answer3_])
        answer4_ = answer4_layer_(answer4_input_)
    
        answer4_layer = Sequential([
            layers.BatchNormalization(),
            layers.Dense(206, kernel_initializer='lecun_normal', activation='selu')
        ], name='answer4_layer')
    
        answer4 = answer4_layer(answer4_)
    
        # non-scored training
        answer5_layer = Sequential([
            layers.BatchNormalization(),
            layers.Dense(y_ns_train.shape[1], activation='sigmoid')
        ], name='answer5_layer')
    
        answer5_ns = answer5_layer(answer4)
    
        model = Model(inputs=[input1_, input2_], outputs=answer5_ns)
        model.compile(optimizer='adam', loss=losses.BinaryCrossentropy(
            label_smoothing=self.cfg.label_smoothing), metrics=logloss)
    
        hist = model.fit([x1_train, x2_train], y_ns_train, epochs=self.cfg.epochs_ns, 
                         batch_size=self.cfg.batchsize,
                         validation_data=([x1_valid, x2_valid], y_ns_valid),
                         callbacks=[reduce_lr, check_point, early_stopping])
    
        # Load best model
        model.load_weights(
            os.path.join(self.save_path, f'weights_seed{seed}_fold{fold}'))
    
        # scored training
        answer5_layer_sc = Sequential([
            layers.BatchNormalization(),
            layers.Dense(y_sc_train.shape[1], activation='sigmoid')
        ])
    
        answer5_sc = answer5_layer_sc(answer4)
    
        model = Model(inputs=[input1_, input2_], outputs=answer5_sc)
        model.compile(optimizer='Adam', 
                      loss=losses.BinaryCrossentropy(
                          label_smoothing=self.cfg.label_smoothing), metrics=logloss)
    
        train_metric_old = model.evaluate(
            [x1_train, x2_train], y_sc_train, return_dict=True)['loss']
        valid_metric_old = model.evaluate(
            [x1_valid, x2_valid], y_sc_valid, return_dict=True)['loss']
        
        print(f'''After nonscored training: train_loss= {train_metric_old}, 
              valid_loss= {valid_metric_old}''')
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_logloss', min_delta=1E-5, patience=self.cfg.patience_sc,
            mode='auto', restore_best_weights=True)
        check_point = callbacks.ModelCheckpoint(
            os.path.join(self.save_path, f'weights_seed{seed}_fold{fold}'),
            save_best_only=True, save_weights_only=True,
            verbose=0, mode='auto')
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_logloss', factor=0.5, patience=5, verbose=0,
            mode='auto', min_lr=1E-5)
    
        hist = model.fit([x1_train, x2_train], y_sc_train, epochs=self.cfg.epochs_sc, 
                         batch_size=self.cfg.batchsize,
                     validation_data=([x1_valid, x2_valid], y_sc_valid),
                     callbacks=[reduce_lr, check_point, early_stopping])
        # load best model
        model.load_weights(
            os.path.join(self.save_path, f'weights_seed{seed}_fold{fold}'))
    
        train_metric_old = model.evaluate(
            [x1_train, x2_train], y_sc_train, return_dict=True)['loss']
        valid_metric_old = model.evaluate(
            [x1_valid, x2_valid], y_sc_valid, return_dict=True)['loss']
        
        print(f'''After scored training: train_loss= {train_metric_old}, 
              valid_loss= {valid_metric_old}''')
        # big loop
        loop = 1
        while True:
            model = self._freeze_unfreeze(model, runty='freeze')
            model.compile(optimizer=optimizers.Adadelta(learning_rate=self.cfg.learning_rate/3),
                          loss=losses.BinaryCrossentropy(
                              label_smoothing=self.cfg.label_smoothing), metrics=logloss)
            reps = 0
            while True:
                hist = model.fit([x1_valid, x2_valid], y_sc_valid, 
                                 epochs=1, batch_size=self.cfg.batchsize)
                train_metric = model.evaluate(
                    [x1_train, x2_train], y_sc_train, return_dict=True)['loss']
                valid_metric = model.evaluate(
                    [x1_valid, x2_valid], y_sc_valid, return_dict=True)['loss']
                
                if (train_metric_old >= train_metric and valid_metric_old >= valid_metric):
                    reps += 1
                    train_metric_old = train_metric
                    valid_metric_old = valid_metric
                    model.save_weights(os.path.join(self.save_path, 
                                                    f'weights_seed{seed}_fold{fold}'))
                else:
                    model.load_weights(os.path.join(self.save_path, 
                                                f'weights_seed{seed}_fold{fold}'))
                    print(f'''{loop} loop ---> After frozen-step best train loss = 
                          {train_metric_old}, valid loss = {valid_metric_old}, after 
                          {reps} epochs''')
                    break
            if reps == 0:
                break
            
            model = self._freeze_unfreeze(model, runty='unfreeze')
            model.compile(optimizer=optimizers.Adadelta(learning_rate=self.cfg.learning_rate/5),
                          loss=losses.BinaryCrossentropy(
                              label_smoothing=self.cfg.label_smoothing), metrics=logloss)
            reps = 0
            while True:
                hist = model.fit([x1_train, x2_train], y_sc_train, 
                                 epochs=1, batch_size=self.cfg.batchsize)
                train_metric = model.evaluate(
                    [x1_train, x2_train], y_sc_train, return_dict=True)['loss']
                valid_metric = model.evaluate(
                    [x1_valid, x2_valid], y_sc_valid, return_dict=True)['loss']
                if (train_metric_old >= train_metric and valid_metric_old >= valid_metric):
                    reps += 1
                    train_metric_old = train_metric
                    valid_metric_old = valid_metric
                    model.save_weights(os.path.join(self.save_path, 
                                            f'weights_seed{seed}_fold{fold}'))
                else:
                    model.load_weights(os.path.join(self.save_path, 
                                                f'weights_seed{seed}_fold{fold}'))
                    print(f'''{loop} loop ---> After non frozen-step best train loss = 
                          {train_metric_old}, valid loss = {valid_metric_old}, after 
                          {reps} epochs''')
                    break
            if reps == 0:
                break
            
            loop += 1
            scores_val_loss = valid_metric_old
            scores_loss = train_metric_old
        
        model.save(os.path.join(self.save_path, 
                                f'Final_seed{seed}_fold{fold}'))
        
        # oof[val_idx] = model.predict([x1_valid, x2_valid])
        
        return None
            
    def run_evaluate(self, seed):
        
        """ model = models.load_model(
            os.path.join(self.load_path, f'Final_seed{seed}_fold{self.fold}'),
            custom_objects={'logloss': logloss}) """
            
        model = models.load_model(
            os.path.join(self.load_path, f'Final_seed{seed}_fold{self.fold}'),
            custom_objects={'logloss': logloss}, compile=False)
        
        predictions = model.predict([self.data_test, self.data_test_2])
        y_val = model.predict([self.x_valid, self.x_valid_2])
        return y_val, predictions
        
    def run_k_fold(self, seed):
        # oof = np.zeros((self.train1.shape[0], self.targets_sc.shape[1]-2))
        # predictions = np.zeros((self.data_test.shape[0], self.y_train_sc.shape[1]))
        
        if (self.runty == 'traineval'):
            self.run_training(seed)
        y_val, predictions = self.run_evaluate(seed)
        
        return y_val, predictions

def build_model(n_features, n_features_2, n_labels, label_smoothing = 0.0005):        
    input_1 = layers.Input(shape = (n_features,), name = 'Input1')
    input_2 = layers.Input(shape = (n_features_2,), name = 'Input2')

    head_1 = Sequential([
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(512, activation="elu"), 
        layers.BatchNormalization(),
        layers.Dense(256, activation = "elu")
        ],name='Head1') 

    input_3 = head_1(input_1)
    input_3_concat = layers.Concatenate()([input_2, input_3])

    head_2 = Sequential([
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, "relu"),
        layers.BatchNormalization(),
        layers.Dense(512, "elu"),
        layers.BatchNormalization(),
        layers.Dense(256, "relu"),
        layers.BatchNormalization(),
        layers.Dense(256, "elu")
        ],name='Head2')

    input_4 = head_2(input_3_concat)
    input_4_avg = layers.Average()([input_3, input_4]) 

    head_3 = Sequential([
        layers.BatchNormalization(),
        layers.Dense(256, kernel_initializer='lecun_normal', activation='selu'),
        layers.BatchNormalization(),
        layers.Dense(n_labels, kernel_initializer='lecun_normal', activation='selu'),
        layers.BatchNormalization(),
        layers.Dense(n_labels, activation="sigmoid")
        ],name='Head3')

    output = head_3(input_4_avg)


    model = Model(inputs = [input_1, input_2], outputs = output)
    model.compile(optimizer='adam', loss=losses.BinaryCrossentropy(label_smoothing=label_smoothing), metrics=logloss)
    
    return model

if __name__ == '__main__':
    pass