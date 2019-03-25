###	目录结构
```
./
|
|--- feature_analyzation
|                       |--- feature_analyzation_yangjie.pdf
|
|
|--- result
|          |--- titanic_pred_LogisticRegression_yangjie.csv
|          |--- titanic_pred_svm_yangjie.csv
|          |--- result_yangjie.png
|
|
|--- src
|       |--- model_yangjie.py
|
|
|--- readme.md
```
### 简述

####    第一次尝试
*	特征工程使用六个特征，分别是Pclass,Parch,SibSp,Fare, Sex, cabin, Embarked
*	模型使用了LogisticRegression和SVM,准确率分别是74.162%和76.55%

####    第二次尝试
#####   做了如下调整：
*	删除Cabin特征
*	Parch和SibSp合并作为一个新特征FamilySize
*	Age的划分有所改变，分为四个年龄段
*	增加了新特征Name
*	多尝试了几种模型，都是直接用sickit-learn调用，GradientBoosting在kaggle上准确率到了80.382% 
