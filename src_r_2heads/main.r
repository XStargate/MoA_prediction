library(tensorflow)
library(tidyverse)
library(recipes)
library(modelr)
library(keras)
library(rjson)

PATH <- "../input/lish-moa/"
SEEDS <- c(23, 228, 1488, 1998, 2208, 2077, 404)
KFOLDS <- 10


label_smoothing_alpha = 0.0005

P_MIN <- label_smoothing_alpha
P_MAX <- 1 - P_MIN

tr <- read_csv(str_c(PATH, "train_features.csv"))
head(tr)
rem_col = tr$cp_type != "ctl_vehicle"
tr = tr[rem_col,]

te <- read_csv(str_c(PATH, "test_features.csv"))

Y <- read_csv(str_c(PATH, "train_targets_scored.csv")) %>% select(-sig_id) %>% data.matrix()
Y = Y[rem_col,]

Y0 <- read_csv(str_c(PATH, "train_targets_nonscored.csv")) %>% select(-sig_id) %>% data.matrix()
Y0 = Y0[rem_col,]

sub <- read_csv(str_c(PATH, "sample_submission.csv")) %>% mutate_if(is.numeric, ~ 0)

tmp = fromJSON(file = "../input/t-test-pca-rfe-logistic-regression/main_predictors.json")

preds = tmp$start_predictors

second_Xtrain = tr[, preds] %>% as.matrix()

head(second_Xtrain)

second_Xtest = te[, preds] %>% as.matrix()

logloss <- function (y, y_h) {
  y_h <- k_clip(y_h, P_MIN, P_MAX)
  -k_mean(y * k_log(y_h) + (1 - y) * k_log(1 - y_h))
}

scores_val_loss <- c()
scores_loss <- c()

for (s in SEEDS) {
    
  set.seed(s)
  k = 0
  for (rs in crossv_kfold(tr, KFOLDS)$train) {

        k = k + 1
        file_name = paste0('model weights of seed ',s,' and fold ',k,'.h5')
      
      
        tri <- as.integer(rs)
      
          
        input1_preprocessor = tr[tri,] %>%
               recipe(~ .) %>%
               step_rm(sig_id,starts_with("cp_")) %>% 
      
              step_mutate(g_mean = apply(across(starts_with("g-")), 1, mean),
                c_mean = apply(across(starts_with("c-")), 1, mean)) %>% 
      
                step_mutate_at(contains("g-"), fn = list(copy_g = function(x) x)) %>%
                step_mutate_at(contains("c-"), fn = list(copy_c = function(x) x)) %>%
                step_normalize(all_numeric()) %>%
                step_pca(contains("copy_g"), num_comp = 2, prefix = "g_pca") %>%
                step_pca(contains("copy_c"), num_comp = 180, prefix = "c_pca") %>%                                             
                prep()

      
        input1 = juice(input1_preprocessor, composition = "matrix")
        validation1 = bake(input1_preprocessor, tr[-tri, ], composition = "matrix")
        test1 <- bake(input1_preprocessor, te, composition = "matrix")
      
      
        input2_preprocessor = second_Xtrain[tri,] %>%
               recipe(~ .) %>%
               step_normalize(all_numeric()) %>%
                  prep()  
      
        input2 = juice(input2_preprocessor, composition = "matrix")
        validation2 = bake(input2_preprocessor, second_Xtrain[-tri, ], composition = "matrix")
        test2 <- bake(input2_preprocessor, second_Xtest, composition = "matrix")
      

      
      
        early_stopping <- callback_early_stopping(patience = 10, min_delta = 1e-05)
        check_point <- callback_model_checkpoint(file_name, save_best_only = TRUE, verbose = 0, mode = "auto")
        reduce_lr <- callback_reduce_lr_on_plateau(factor = 0.5, patience = 4, verbose = 0, mode = "auto")

                                                         
                                                         
       
                                                         
        # start configuration                                                 
        input1_ = layer_input(shape = c(ncol(input1)))
        input2_ = layer_input(shape = c(ncol(input2)))
    
        output1 = input1_ %>% 
            layer_batch_normalization() %>% 
            layer_dropout(0.2) %>% 
            layer_dense(512, "elu") %>% 
            layer_batch_normalization() %>%
            layer_dense(256, "elu") 
    
        #output2 = input2 %>% layer_dense(256, "elu") 
    
        answer1 = layer_concatenate(list(output1, input2_)) %>% 
            layer_batch_normalization() %>%
            layer_dropout(0.3) %>% 
            layer_dense(512, "relu") 
        
        answer2 =  layer_concatenate(list(output1, input2_, answer1)) %>%     
            layer_batch_normalization() %>%                                                                                       
            layer_dense(512, "elu") %>% 
            layer_batch_normalization() %>%
            layer_dense(256, "relu")
        
        answer3 = layer_concatenate(list(answer1, answer2)) %>% 
            layer_batch_normalization() %>%
            layer_dense(256, "elu") 
                                                         
        answer3_ = layer_concatenate(list(answer1, answer2, answer3)) %>% 
            layer_batch_normalization() %>%
            layer_dense(256, "relu")
                                                         
        answer4 = layer_concatenate(list(output1, answer2, answer3, answer3_)) %>% 
            layer_batch_normalization() %>%
            layer_dense(256, kernel_initializer='lecun_normal', activation='selu', name = 'last_frozen') %>%
            layer_batch_normalization() %>%
            layer_dense(206, kernel_initializer='lecun_normal', activation='selu')
        
                                                 
                                                                                                        
        
        # non-scored training
        answer5 = answer4 %>%
            layer_batch_normalization() %>% 
            layer_dense(units = ncol(Y0), "sigmoid")


        m_nn = keras_model(inputs = list(input1_, input2_), answer5)
    
        m_nn %>% keras::compile(optimizer = "adam", 
                   loss = tf$losses$BinaryCrossentropy(label_smoothing = label_smoothing_alpha),
                   metrics = logloss) 
                                                         

        history <- m_nn %>% keras::fit(list(input1, input2), Y0[tri, ],
                                    epochs = 50,
                                    batch_size = 128,
                                    validation_data = list(list(validation1, validation2), Y0[-tri, ]),
                                    callbacks = list(early_stopping, check_point, reduce_lr),
                                    verbose = 0)

        load_model_weights_hdf5(m_nn, file_name)

                                                                                                             
        # scored training
                                                                                                               
        answer5 = answer4 %>%
            layer_batch_normalization() %>% 
            layer_dense(units = ncol(Y), "sigmoid")
                                                         
        m_nn = keras_model(inputs = list(input1_, input2_), answer5)
                                                         
        m_nn %>% keras::compile(optimizer = "adam", 
                   loss = tf$losses$BinaryCrossentropy(label_smoothing = label_smoothing_alpha),
                   metrics = logloss) 
        
        train_metric_old = evaluate(m_nn, list(input1, input2), Y[tri, ])[['loss']]
        valid_metric_old = evaluate(m_nn, list(validation1, validation2), Y[-tri, ])[['loss']]   
        
        early_stopping <- callback_early_stopping(patience = 10, min_delta = 1e-05)
        check_point <- callback_model_checkpoint(file_name, save_best_only = TRUE, verbose = 0, mode = "auto")
        reduce_lr <- callback_reduce_lr_on_plateau(factor = 0.5, patience = 4, verbose = 0, mode = "auto")                                                 
                                                         
        cat('After non-scored training: validation_loss =',valid_metric_old,'; train_loss =',train_metric_old,'\n')
                                                         
        history <- m_nn %>% keras::fit(list(input1, input2), Y[tri, ],
                                    epochs = 50,
                                    batch_size = 128,
                                    validation_data = list(list(validation1, validation2), Y[-tri, ]),
                                    callbacks = list(early_stopping, check_point, reduce_lr),
                                    verbose = 0)

        load_model_weights_hdf5(m_nn, file_name)                                                 


        val_loss = min(history$metrics$val_loss)
        arg_val_loss = which.min(history$metrics$val_loss)
        loss = history$metrics$loss[arg_val_loss]

        #scores_val_loss <- c(scores_val_loss, val_loss)
        #scores_loss <- c(scores_loss, loss)
        cat("Best val-loss:", val_loss, "at", arg_val_loss, "step",'\n')
                                                         
        save_model_weights_hdf5(m_nn, 'tmp.h5')                                        
        
        train_metric_old = evaluate(m_nn, list(input1, input2), Y[tri, ])[['loss']]
        valid_metric_old = evaluate(m_nn, list(validation1, validation2), Y[-tri, ])[['loss']]                                                 
        
        cat('Before loop: validation_loss =',valid_metric_old,'; train_loss =',train_metric_old,'\n')
                                                         
        # big loop
        loop = 1
        repeat{
            
            freeze_weights(m_nn, to = 'last_frozen')                                                 
            m_nn %>% keras::compile(optimizer = optimizer_adadelta(lr = 0.001 / 3), 
                   loss = tf$losses$BinaryCrossentropy(label_smoothing = label_smoothing_alpha),
                   metrics = logloss)
            
            reps = 0
            #frozen
            repeat{
                history <- m_nn %>% keras::fit(list(validation1, validation2), Y[-tri, ],
                                    epochs = 1,
                                    batch_size = 128,
                                    verbose = 0)
            
                train_metric = evaluate(m_nn, list(input1, input2), Y[tri, ])[['loss']]
                valid_metric = evaluate(m_nn, list(validation1, validation2), Y[-tri, ])[['loss']]
                #cat(train_metric_old, train_metric)
                if(train_metric_old >= train_metric & valid_metric_old >= valid_metric){
                    reps = reps + 1
                    train_metric_old = train_metric
                    valid_metric_old = valid_metric
                    save_model_weights_hdf5(m_nn, 'tmp.h5')
                #cat(' -- best valid =',valid_metric, 'on train',train_metric,'after',reps,'epochs \n')
                } else{
                    load_model_weights_hdf5(m_nn, 'tmp.h5')
                    cat(loop, 'loop ---> After frozen-step best valid =',valid_metric_old, 'on train',train_metric_old,'after',reps,'epochs \n')
                        
                    break
                }
            }
            
            if(reps == 0){ # no progress? STOP!
                break
            }
            
            
            
            
            unfreeze_weights(m_nn, to = 'last_frozen')                                                 
            m_nn %>% keras::compile(optimizer = optimizer_adadelta(lr = 0.001 / 5), 
                   loss = tf$losses$BinaryCrossentropy(label_smoothing = label_smoothing_alpha),
                   metrics = logloss)
            
            reps = 0
            #unfrozen
            repeat{
                history <- m_nn %>% keras::fit(list(input1, input2), Y[tri, ],
                                    epochs = 1,
                                    batch_size = 128,
                                    verbose = 0)
            
                train_metric = evaluate(m_nn, list(input1, input2), Y[tri, ])[['loss']]
                valid_metric = evaluate(m_nn, list(validation1, validation2), Y[-tri, ])[['loss']]
                #cat(train_metric_old, train_metric)
                if(valid_metric_old >= valid_metric){
                    reps = reps + 1
                    train_metric_old = train_metric
                    valid_metric_old = valid_metric
                    save_model_weights_hdf5(m_nn, 'tmp.h5')
                #cat(' -- best valid =',valid_metric, 'on train',train_metric,'after',reps,'epochs \n')
                } else{
                    load_model_weights_hdf5(m_nn, 'tmp.h5')
                    cat(loop, 'loop ---> After nonfrozen-step best valid =',valid_metric_old, 'on train',train_metric_old,'after',reps,'epochs \n')
                        
                    break
                }
            }
            
            if(reps == 0){
                break
            }
            
            loop = loop + 1
            scores_val_loss <- c(scores_val_loss, valid_metric_old)
            scores_loss <- c(scores_loss, train_metric_old)
            
        }
                                                         
                                                         
                                                                                                                                                                                                                                                                                                                                                                                              
        sub[, -1] <- sub[, -1] + predict(m_nn, list(test1, test2)) 

        rm(tri, m_nn, history)
        cat('\n')
        #file.remove("model.h5")
  }
}

sub[, -1] <- sub[, -1] / KFOLDS / length(SEEDS)
sub[te$cp_type == "ctl_vehicle", -1] <- 0
write_csv(sub, "submission.csv")