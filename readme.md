# classification

包括 1）单分类器模型：逻辑斯蒂回归（LR）、决策树（DT）、支持向量机（SVM）、K-近邻（KNN）；2）集成分类器：随机森林（Random Forest）、套袋法（Bagging）、Adaboost、Gradient Boosting Decison Tree（GBDT）；3）神经网络模型（自己实现）

## 介绍

根据患者的各项生命体征，构建分类模型预测患者的呼吸系统评分。

## 环境

python 3.7  
pytorch 1.1  
tqdm  
sklearn

## 数据预处理

在 main.py 中 取消 pre_process 的注释，跑一遍，可以得到处理过的数据文件，之后无需再跑。

pre_process 中有 _delete 参数，该参数决定了对于空值的数据的处理方式，若 _delete 为 True， 则删除该行数据； 反之，则取与该数据评分相同的所有数据该项的平均值。

### ！！！**这里可以按照文档里面数据处理部分提到的，再多加两个处理方式，都挺好加的。！！！**



## 使用说明

```
# 训练并测试：
# --k-fold 参数加上就会跑五折交叉验证，不加默认就不跑
# 1）单个分类器
# 逻辑斯蒂回归（LR）
python main.py --model LR --k-fold

# 决策树（DT）
python main.py --model DT --k-fold

# 支持向量机（SVM）
python main.py --model LR --k-fold

# K-近邻（KNN）K指定KNN参数
python main.py --model KNN --k 5 --k-fold

# 2）集成分类器：
# 随机森林（Random Forest）
python main.py --model RF --k-fold

# 套袋法（Bagging）
python main.py --model BAG --k-fold

# Adaboost
python main.py --model BST --k-fold

# ！！！！！TODO ！！！！！
# Gradient Boosting Decison Tree（GBDT）
python main.py --model GBDT --k-fold

# 3）神经网络NN
python main.py --model NN --gpu-id 0 --batch-size 10000 --hidden-size 32 --max-patience 600 --learning-rate 0.008 --k-fold
```


