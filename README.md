# Mechanisms of Action (MoA) Prediction
**Kaggle competition**

Ranked 98th (Top 2.3%) among 4373 teams

src_genetic_knn: KNN algorithm based on genetic search.  
src_transfer_drugid_new: preprocess the dataset according to drug id.  
src_resnet_tensorflow: Resnet network based on PCA and Quantile transformation.  
src_rankgauss_new: FCNN(MLP) network based on PCA and Quantile transformation.  
src_knn_cluster: create features by clustering based on KNN cluster.__
src_tabnet: Tabnet network based on PCA and Quantile transformation.  

The final result is from a blend model:  
0.33 * src_resnet_tensorflow + 0.33 * src_transfer_drugid_new + 0.2 * src_knn_cluster + 0.095 * src_rankgauss_new + 0.045 * src_tabnet
