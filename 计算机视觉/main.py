# This is a sample Python script.
import numpy as np
import pickle
from PIL import Image
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.





# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 激活函数的梯度计算
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


# 输出函数，转换为概率输出
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)  # 防止溢出
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 转换为one_hot编码
def one_hot(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T


def load_mnist():
    # https://s3.amazonaws.com/img-datasets/mnist.npz
    # 从上面地址下载文件到本地
    file = 'E:/ChromeCoreDownloads/2022/mnist.npz'
    f = np.load(file)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    # x_train格式默认是[60000,28,28],即为60000个样本，28*28的矩阵
    # 进行平面化处理，
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    # 正规化0.0~1.0的值,灰度值取值范围是0-255,所以除以255即可归一
    # 正规化的目的：1、防止计算溢出；2、梯度下降需要，防止由于输出值太大导致梯度过大的情况(梯度一大，又要调整学习率，而我们使用固定学习率)
    x_train = x_train.astype(np.float32)
    x_train /= 255.0
    x_test = x_test.astype(np.float32)
    x_test /= 255.0
    # 将标签用one_hot编码
    # 即为[0,1,0,0,0,0,0,0,0,0]形式，其中等于1的正确值的标签
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    f.close()
    return (x_train, y_train), (x_test, y_test)



class twolayerNet:
    def __init__(self, input_size, hidden_size, output_size, init_std=0.01):
        self.params = {}
        self.params['w1'] = init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params['w2'] = init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    # 预测计算函数
    def predict(self, x):
        # y=softmax(w2*(sigmoid(w1*x+b1))+b2)
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)
        return y

    # 使用均方误差
    def loss(self, x, t):
        y = self.predict(x)
        return mean_squared_error(y, t)

    # 计算精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 梯度计算
    def gradient(self, x, t):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        batch_size = x.shape[0]
        # 先保存前向计算值
        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)
        # 后向计算梯度：输出层(y)->隐含层2(w2,b2)->激活函数(sigmoid_grad)->隐含层1(w1,b1)
        # 输出层梯度
        dy = (y - t) / batch_size
        # 隐含层2的梯度
        grads['w2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        dHidden2 = np.dot(dy, w2.T)
        # 激活函数的梯度
        dSigmoid = sigmoid_grad(a1) * dHidden2
        # 隐含层1的梯度
        grads['w1'] = np.dot(x.T, dSigmoid)
        grads['b1'] = np.sum(dSigmoid, axis=0)
        return grads




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 加载数据，分为学习数据和测试数据，x为输入，y为标签
    (x_train, y_train), (x_test, y_test) = load_mnist()
    # mini-batch的实现
    # print(x_train.shape) #输出:[60000,784]
    # print(y_train.shape)#输出:[10000,10]
    train_size = x_train.shape[0]
    # 每次迭代的样本数
    batch_size = 100
    # 迭代计算次数
    iters_num = 10000
    # 学习率
    learning_rate = 0.2
    iter_per_epoch = max(train_size / batch_size, 1)
    network = twolayerNet(input_size=784, hidden_size=100, output_size=10)
    # 迭代计算
    for i in range(iters_num):
        # 随机选取batch_size个样本
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]
        # 计算梯度
        grads = network.gradient(x_batch, y_batch)
        # 梯度下降
        for key in ('w1', 'b1', 'w2', 'b2'):
            network.params[key] -= learning_rate * grads[key]
        # 计算损失
        loss = network.loss(x_batch, y_batch)
        # 结果输出
        if i % iter_per_epoch == 0:
            # 计算精度并输出
            train_acc = network.accuracy(x_train, y_train)
            test_acc = network.accuracy(x_test, y_test)
            print("train accuracy:" + str(train_acc) + ";" + "test accuracy " + str(test_acc))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
