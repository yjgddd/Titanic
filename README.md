# 目录结构
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
# 简述

## 题目链接
[Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview/tutorials)
## 运行环境
python3、Jupyter
## 数据预处理
通过数据质量分析，检查原始数据中是否存在脏数据，脏数据包括：缺失值，异常值，不一致的值，重复数据及含有特殊符号（如#、￥、*）的值。本题目中主要是缺失值的处理。 

导入可能用到的包，将训练集和测试集合并，统一进行数据预处理 

```python
import numpy as np
import pandas as pd
import seaborn as sns 
from collections import Counter
import matplotlib.pyplot as plt
train=pd.read_csv('../Titanic/all/train.csv')#读入训练数据集
test=pd.read_csv('../Titanic/all/test.csv')#读入测试数据集
print('训练数据集规模',train.shape)#训练数据集规模 (891, 12)
print('测试数据集规模',test.shape)#测试数据集规模 (418, 11)
df=train.append(test,sort=True) #合并数据集，方便进行数据预处理
print('合并后的数据集规模',df.shape)#合并后的数据集规模 (1309, 12)
```
训练数据集的规模是891\*12,891条数据，12个属性，测试数据集的规模418\*11属性，测试集少一个属性 Survived，这是需要通过模型预测的,合并后的数据集规模是1309\*12。 

查看前 10 条数据了解一下数据集的格式： 

```python
df.head(10)
```
![1.png](https://i.loli.net/2019/04/07/5ca97a62dc27d.png)
查看基本的统计信息： 


```python
pd.options.display.float_format='{:,.3f}'.format # 数据显示模式
df.describe() #查看数据的基本统计信息
```
![2.png](https://i.loli.net/2019/04/07/5ca97a99c3964.png)

查看每一列的数据总数和数据类型，对于每个属性，除了Survived应当是891个，其他属性的合理数据总数应该是 1309。 

```python
df.info()#查看每一列的数据以及数据类型，进行缺失值分析
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1309 entries, 0 to 417
    Data columns (total 12 columns):
    Age            1046 non-null float64
    Cabin          295 non-null object
    Embarked       1307 non-null object
    Fare           1308 non-null float64
    Name           1309 non-null object
    Parch          1309 non-null int64
    PassengerId    1309 non-null int64
    Pclass         1309 non-null int64
    Sex            1309 non-null object
    SibSp          1309 non-null int64
    Survived       891 non-null float64
    Ticket         1309 non-null object
    dtypes: float64(3), int64(4), object(5)
    memory usage: 132.9+ KB
缺失值分析：

属性 | 含义|数据总数 | 缺失值| 缺失率
:---:|:---:|:---:|:---:|:---:
Age|年龄|1046|263|20.09%
Cabin|船舱号|295|1014|77.46%
Embarked|登船港口|1307|2|0.15%
Fare|船票价格|1308|1|0.08%
Name|姓名|1309|0|0
Parch|父母/子女数目|1309|0|0
PassengerID|乘客编号|1309|0|0
Pclass|乘客等级（三个等级，1=一等，2=二等，2=3等)|1309|0|0
Sex|性别(男为male,女为female)|1309|0|0
SibSp|兄弟姐妹、配偶数列|1309|0|0
Ticket|船票号码|1309|0|0
Surviied|是否存活(存活是1，死亡是0)|891|0|0


需要进行缺失值处理的有：Age、Cabin、Embarked、Fare。 

- 年龄 Age 是数值型，缺失值暂时采用平均值填充。 

```python
df['Age']=df['Age'].fillna(df['Age'].mean())#年龄缺失值用均值填充
```

- 船舱号 Cablin 缺失较多，可以搁置此特征不使用。

- Embarked 缺失值较少，且是分类数据，可以采用众数填充，众数是'S'。

```python
df['Embarked']=df['Embarked'].fillna('S')#Embarked缺失值用众数'S'填充
```

- Fare 是船票价格，数值型数据，只缺失一个数据，可以用平均值填充。 

```python
df['Fare']=df['Fare'].fillna(df['Fare'].mean())#船票价格缺失值用均值填充
```
 
缺失值处理之后的数据信息如下： 
```python
#查看缺失值处理之后的数据
df.info()
```

		<class 'pandas.core.frame.DataFrame'>
		    Int64Index: 1309 entries, 0 to 417
		    Data columns (total 12 columns):
		    Age            1309 non-null float64
		    Cabin          1309 non-null object
		    Embarked       1309 non-null object
		    Fare           1309 non-null float64
		    Name           1309 non-null object
		    Parch          1309 non-null int64
		    PassengerId    1309 non-null int64
		    Pclass         1309 non-null int64
		    Sex            1309 non-null object
		    SibSp          1309 non-null int64
		    Survived       891 non-null float64
		    Ticket         1309 non-null object
		    dtypes: float64(3), int64(4), object(5)
		    memory usage: 132.9+ KB
## 数据分析
对于分类数据和是否存活的关系作出柱状图。
柱状图可以写一个函数调用。

```python
def feature_plot(df, features, hue):
    f, ax = plt.subplots(len(features),figsize = [5,16])
    for i,x in enumerate(features):
       sns.countplot(x=x,hue=hue,data=train,ax=ax[i])
    plt.tight_layout(pad=0)
```
```python
feature_plot(df=df,features=['Embarked','Parch','SibSp','Pclass','Sex'],hue = "Survived")
```
对每个特征进行必要的分析，观察其和存活是否有关系：

1、	Age 

```python
age = sns.FacetGrid(train, hue="Survived",aspect=2)
age.map(sns.kdeplot,'Age',shade= True)
age.set(xlim=(0, train['Age'].max()))
age.add_legend()
```
![10.png](https://i.loli.net/2019/04/16/5cb586b94b20a.png)

看到明显小孩和年轻人存活下来的可能性更大，特征明显影响存活率，要保留该项特征。 
 
2、	Cabin 

Cablin 数据有很多缺失值，删除该特征

  
3、	Embarked 

Embarked 共有三类，分别是S、C、Q,存活的情况有一定差别，C 港口的生存概率大一些，此特征保留。

![5.png](https://i.loli.net/2019/04/16/5cb5c1a0e01a1.png)

  
  
4、	Fare 

Fare 是船票的价格，当价格超过一定值的时候，船票价值高的存活率要更高，此特征保留。

![11.png](https://i.loli.net/2019/04/16/5cb587dba28da.png)
 
5、	Name 

Name 中可能包含一些重要的信息，比如人的称呼、头衔。对此特征进行处理，提取出新的有用特征。

```python
df['Title'] = df['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
df['Title'] = df['Title'].map(Title_Dict)
sns.countplot(data=df,x='Title',hue='Survived')
```
![12.png](https://i.loli.net/2019/04/16/5cb58b8ebdb20.png)

称谓一定程度影响了存活率，比如称谓为Mr的存活率很低。保留此特征。

6、	Parch 

Parch 表示父母/子女数目，可以看到有父母/子女陪同的，比单独出行的生存率高，此特征也需要保留。

![6.png](https://i.loli.net/2019/04/15/5cb45b21c0358.png)

 
7、	PassengerID 
乘客的 ID，ID 是无关信息，删除。 

8、	Pclass 
可以看到社会等级高的阶层存活概率显然更大，此特征对生存概率有很大影响，需要保留。 

![8.png](https://i.loli.net/2019/04/15/5cb45b2221146.png)
  

9、	Sex 
女性的存活概率更大，性别特征对生存率有影响，此项特征保留。 

![9.png](https://i.loli.net/2019/04/15/5cb45b222129f.png)
  
  
10、SibSp

![13.png](https://i.loli.net/2019/04/16/5cb5ab9f54f4d.png)

SibSp表示兄弟姐妹的数量，有至少一个兄弟姐妹陪同的存活概率更高。
  
 
11、	Ticket 
船票号码，无关特征，删除。 

12、增加新特征
增加新特征FamilySize=Parch+SibSp+1,观察和存活率的关系。

![14.png](https://i.loli.net/2019/04/16/5cb5adba6c0a8.png)

FamilySize>1的更可能存活，将其作为新特征。

## 特征工程
由于 Age 是连续性变量，不好观察特征，将年龄划分为4 类，0-14 岁为 child，15-24 岁为 youth，25-64 为 adult，大于 64 岁为 old，分类，进行 one-hot 编码。

Parch和SibSp结合为新特征FamilySize，划分为三个类别1,1-4和大于4
 
Pclass 为 1、2、3 类，Embarked有三类，分别为S、C、Q，均使用 one-hot 编码。
 
Sex，1 代表 male，0 代表 female,进行one-hot编码。


Fare根据划分Fare<=2.5和Fare>2.5.进行one-hot编码。

Name根据已经划分的类别进行one-hot编码

最终保留特征： 

Pclass,Fare,Age,Name,Sex,Embarked,FamilySize并且都转换为分类数据进行one-hot编码。

## 模型运用
采用GBDT，Kaggle上准确率为0.80382
