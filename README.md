# dogs-VS-cats-pytorch
Pytorch实现Kaggle竞赛“猫狗分类”，准确率超过99%。

Kaggle网站：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition

数据集下载：
- 链接：https://pan.baidu.com/s/1pzyUh8T7m67sD6C3FWTnXg 
- 提取码：9r3f 


### Step1：特征提取

使用```feature_extract.py```提取特征。

采用迁移学习的思想，使用Pytorch预训练的模型“GoogLeNet”、“ResNet”和“ResNeXt”提取图像特征。

选择预训练模型的全局平均池化层的输出为新的特征，注意到对于每张图像，GoogLeNet提取到1024维特征；ResNet和ResNeXt提取到2048维特征；最后组合成5120维特征。

特征提取这一步比较花时间，这里提供了特征提取后的结果下载：
- 链接：https://pan.baidu.com/s/1pzyUh8T7m67sD6C3FWTnXg 
- 提取码：9r3f 


### Step2：模型训练

使用```train_test.py```训练模型。

使用提取的特征作为输入进行二分类，直接用一个全连接层，输入5120维，输出2维(Softmax分类)。

使用Dropout，设置p=0.5。

训练速度相对于使用raw image就很快了，CPU上几秒完成。


### Step3：结果预测

使用```train_test.py```预测结果。

模型训练好以后，就可以对测试集进行预测，然后提交到 kaggle 上查看最终成绩。

预测时使用了一个小技巧，将每个预测值限制到[0.005, 0.995]的区间内，这是由于kaggle官方的评估标准是LogLoss，对于预测正确的样本，0.995和1相差无几；但是对于预测错误的样本，0和0.005的差距是非常大的。

值得一提的是，使用ImageFolder读取文件是按照以下顺序（而不是顺序编号）：

```
['test/1.jpg',
 'test/10.jpg',
 'test/100.jpg',
 'test/1000.jpg',
 'test/10000.jpg',
 'test/10001.jpg',
 'test/10002.jpg',
 'test/10003.jpg',
 ......
 ```
 
 故需建立测试数据与预测结果之间的联系(见代码)。
