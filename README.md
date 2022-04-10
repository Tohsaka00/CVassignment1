# CVassignment1
MNIST数据集：

数据源：https://s3.amazonaws.com/img-datasets/mnist.npz

train set size: 55000

validation set size: 5000

test set size: 10000


两层神经网络架构：y = softmax(w2 * (sigmoid(w1 * x + b1)) + b2)

如何训练：运行train.py，train.py包括了训练网络的函数，参数查找的过程，最终得到的模型保存为“model.pth”

如何测试训练好的模型:运行test.py，导入模型并输出精度。
