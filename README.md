## spam_email

email_v2 used SVM as classifier 

email_3 used Naive Bayes Classifier as classifier 


## 代码思路

通过 Python 实现识别垃圾邮件和非垃圾邮件需要以下几个步骤。

1. 收集带标签的电子邮件消息数据集 (标记垃圾邮件为“SPAM”, 标记另一组非垃圾邮
件为“HAM”)。
2. 通过清理和转换文本来预处理数据 (删除停用词等)
3. 从预处理数据中提取特征 Extract features
4. 使用监督学习算法 (Naive 贝叶斯, SVM) 在提取的特征上训练机器学习模型。
5. 在单独的数据集上测试训练模型并评估其性能 (使用准确性指标)。
6. 最后，使用训练好的模型将新收到的邮件分类为垃圾邮件和非垃圾邮件。

## Supervised Learning Algorism: 监督学习算法 (Naive 贝叶斯, SVM)

### Naive 贝叶斯
朴素贝叶斯是最简单但最强大的分类器算法之一，它基于贝叶斯定理公式，假设预测 变量之间相互独立。给定假设 a 和证据 B，根据贝叶斯定理计算器，得到证据 P(A) 前的 假设概率与得到证据 P(A |B) 后的假设概率之间的关系

### 支持向量机(SVM)
支持向量机 (SVM) 是一种有监督的机器学习算法，既可以用于分类，也可以用于回归。 然而，它主要用于分类问题。在 SVM 算法中，我们将每个数据项绘制为 n 维空间中的一 个点 (其中 n 是你拥有的特征的数量)，每个特征的值是特定坐标的值。然后，通过寻找能 够很好地区分两个类别的超平面从而进行分类。
