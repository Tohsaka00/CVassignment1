# This is a sample Python script.
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import torch
import os
from scipy.misc import imsave
from numpy import ndarray
from sklearn.decomposition import PCA
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def normalization(x):
    Max = np.max(x)
    Min = np.min(x)
    x = (x - Min)/(Max - Min)
    return x


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


# 转换为one_hot编码
def one_hot(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def regulation(w, r):
    return 0.5 * r * np.sum(np.square(w))


def load_mnist():
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
    x_valid = x_train[range(55000,60000)]
    y_valid = y_train[range(55000,60000)]
    x_train = x_train[range(55000)]
    y_train = y_train[range(55000)]
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def vec2img(vec):
    result = np.zeros(shape=(int(np.sqrt(vec.shape[0])), int(np.sqrt(vec.shape[0])), 3))
    for i in range(3):
        mat: ndarray = np.array(vec[:, i].reshape(int(np.sqrt(vec.shape[0])), int(np.sqrt(vec.shape[0]))))
        result[:,:,i] = mat
    return result

class twolayerNet:
    def __init__(self, input_size, hidden_size, output_size, init_std=0.01, lambd=0.01, learning_rate=0.1, decay_rate=1/2000):
        self.params = {}
        self.params['w1'] = init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params['w2'] = init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)
        self.lambd = lambd
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

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
        batch_size = x.shape[0]
        y = self.predict(x)
        # 正则化
        penalty = 0
        for key in {'w1', 'b1', 'w2', 'b2'}:
            penalty += regulation(self.params[key], self.lambd)
        return (mean_squared_error(y, t) + penalty) / batch_size

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
        grads['w2'] = np.dot(z1.T, dy) + self.lambd * w2 / batch_size
        grads['b2'] = np.sum(dy, axis=0) + self.lambd * b2 / batch_size
        dHidden2 = np.dot(dy, w2.T)
        # 激活函数的梯度
        dSigmoid = sigmoid_grad(a1) * dHidden2
        # 隐含层1的梯度
        grads['w1'] = np.dot(x.T, dSigmoid) + self.lambd * w1 / batch_size
        grads['b1'] = np.sum(dSigmoid, axis=0) + self.lambd * b1 / batch_size
        return grads

    def learning_decay(self, r, i):
        return r * np.exp(-(self.decay_rate * i))


# Press the green button in the gutter to run the script.
def training(x_train, y_train, x_test, y_test, lr=1.7, bs=100, lambd=0.01, hs=100, dr=1/1000):
    # 加载数据，分为学习数据和测试数据，x为输入，y为标签

    # mini-batch的实现
    # print(x_train.shape) #输出:[60000,784]
    # print(y_train.shape)#输出:[10000,10]
    train_size = x_train.shape[0]
    # 每次迭代的样本数
    batch_size = bs
    # 迭代计算次数
    iters_num = 5000
    # 学习率
    learning_rate = lr
    # 正则化系数lambda
    r = lambd
    # 隐藏层大小
    hidden_size = hs
    # 学习率衰减系数
    decay_rate = dr
    iter_per_epoch = max(train_size / batch_size, 1)
    network1 = twolayerNet(input_size=784, hidden_size=hidden_size, output_size=10, lambd=r, learning_rate=learning_rate, decay_rate=decay_rate)
    # 绘制accuracy曲线
    train_iterations = []
    train_accuracy = []
    train_loss = []
    test_loss = []
    test_accuracy = []

    # 迭代计算SGD
    for i in range(iters_num):
        # 随机选取batch_size个样本
        learning_rate = network1.learning_decay(lr, i=i/iter_per_epoch)
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]
        # 计算梯度
        grads = network1.gradient(x_batch, y_batch)
        # 梯度下降
        for key in ('w1', 'b1', 'w2', 'b2'):
            network1.params[key] -= learning_rate * grads[key]
        # 计算损失
        loss = network1.loss(x_batch, y_batch)
        # 结果输出
        if i % iter_per_epoch == 0:
            # 计算精度并输出
            test_acc = network1.accuracy(x_test, y_test)
            tr_loss = network1.loss(x_train, y_train)
            te_loss = network1.loss(x_test, y_test)
            train_iterations.append(i)
            train_loss.append(tr_loss)
            test_loss.append(te_loss)
            test_accuracy.append(test_acc)
    print("valid accuracy " + str(test_acc) + ";" + "learning rate: " + str(lr))

    return network1, test_acc, train_loss, test_loss, test_accuracy, train_iterations


if __name__ == '__main__':
    (x_train1, y_train1), (x_valid1, y_valid1), (x_test1, y_test1) = load_mnist()
    # 这里是最终的网络
    network, final_acc, train_loss1, test_loss1, test_accuracy1, train_iterations = \
        training(x_train1, y_train1, x_valid1, y_valid1)
    train_loss = train_loss1
    test_loss = test_loss1
    test_accuracy = test_accuracy1
    # 绘制曲线图
    host = host_subplot(111)
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx()
    # 设置类标
    host.set_xlabel("iterations")
    host.set_ylabel("loss")
    par1.set_ylabel("accuracy")

    # 绘制曲线
    p1, = host.plot(train_iterations, train_loss, "b-", label="training loss")
    p2, = host.plot(train_iterations, train_loss, ".")  # 曲线点
    p3, = par1.plot(train_iterations, test_accuracy, label="validation accuracy")
    p4, = par1.plot(train_iterations, test_accuracy, "1")
    p5, = host.plot(train_iterations, test_loss, "g", label="validation loss")
    p6, = host.plot(train_iterations, test_loss, ".")  # 曲线点
    # 设置图标
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)

    # 设置颜色
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p3.get_color())
    # 设置范围
    host.set_xlim([-10, 10000])
    plt.draw()
    plt.show()
    model = torch.load('model.pth')
    test_acc = model.accuracy(x_test1, y_test1)
    print(test_acc)