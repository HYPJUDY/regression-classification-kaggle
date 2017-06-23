# Regression and Classification Competition in Kaggle
Kaggle link: [Linear Regression-SYSU-2017](https://inclass.kaggle.com/c/linear-regression-sysu-2017/) and [Large-scale classification-SYSU-2017](https://inclass.kaggle.com/c/large-scale-classification-sysu-2017/)

## Regression
### Data
1. `save_train.csv`
    
    The format of each line is `id,value0,value1,...,value383,reference` where value0,value1,...,value383 are the features.
2. `save_test.csv`
   
   This file contains the features whose references you need to predict. The format of each line is `id,value0,value1,...,value383`.
3. `sample_submission.csv` (or `result.csv`)
   
   The format of each line is `id,reference`.

### Method
1. Linear Regression
    
    Gradient Descent and Normal Equation by MATLAB
2. Multi-layer Perceptron (MLP) by TensorFlow
3. KNN by scikit-learn

### Result
![](https://hypjudy.github.ioimages/reg-classification-kaggle/regression-score.gif)
![](https://hypjudy.github.ioimages/reg-classification-kaggle/regression-time.gif)

## Classification
### Data
1.`train_data.txt`
  
  The format of each line is `label index1:value1 index2:value2 index4:value4 ...` where value1,value2,... are the features(the ignored value equals 0 e.g. the value3 here equals 0) and there are only 2 classes(label) in all,indexed from 0 to 1.
2.`test_data.txt`
  
  This file contains the features without labels that you need to predict. The format of each line is `id index1:value1 index2:value2 index3:value3 ...`
3.`sample_submission.txt` (or `result.txt`)
  
  The file you submit should have the same format as this file,the format of each line is `id,label`

### Method
1. Logistic Regression
    
    batchGradAscent, stocGradAscent, minibatchGradAscent, batchGradAscentMultiProcess, Python multiprocess paralleling of batchGradAscent
2. Gradient Descent Boosting Tree by XGBoost

### Result
![](https://hypjudy.github.ioimages/reg-classification-kaggle/logistic-regression-gradient-ascent-acc.gif)
![](https://hypjudy.github.ioimages/reg-classification-kaggle/multiprocess-parallel-time.gif)

## Details
Blog in chinese: [[数据挖掘] 回归和分类Kaggle实战](https://hypjudy.github.io/2017/06/23/regression-classification-kaggle/)

详细探讨解决[Kaggle](https://www.kaggle.com/)上某回归/分类比赛的全过程。包括用线性回归之梯度下降法/正规方程法（MATLAB）、神经网络之多层感知器（TensorFlow）、最近邻（scikit-learn）进行**回归**预测和用逻辑回归之梯度上升法（Python）、梯度提升决策树（XGBoost）进行**分类**预测。读者将理解线性回归和逻辑回归的**原理/实现**、其他框架的**使用/调参**，以及如何利用Python的多进程对逻辑回归的运算进行**并行化**提高效率。