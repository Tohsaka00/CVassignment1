# CVassignment1
MNIST数据集：mnist.npz

数据源：https://s3.amazonaws.com/img-datasets/mnist.npz

train set size: 55000

validation set size: 5000

test set size: 10000


两层神经网络架构：y = softmax(w2 * (sigmoid(w1 * x + b1)) + b2)

如何训练：打开train.py，修改load_mnist函数中的路径为存储mnist数据集的路径。运行train.py，train.py包括了训练网络的函数，参数查找的过程，通过运行train.py我们可以找到最好的超参数。

随后运行test.py，根据之前得到的超参数再训练一个最终的神经网络模型，导出为“model.pth”

它可以被导入，并在预测集上进行预测，输出一个精度。
