import numpy as np
import myNet as n
import myDataset as d
import myActFunct as a
import myErrorFunct as ef

def run():
    train_X, test_X, train_lab, test_lab = d.loadDataset('C:/Users/Pietro20/Desktop/mnistDataset/')
    # ld = d.loadDataset('C:/Users/anton/Downloads/Telegram_Desktop/')
    list_err_train, list_err_val, combinations = n.crossValidationKFold(train_X, train_lab, ef.crossEntropy, 784, 10, list_hidden_size=[[256, 128], 256], list_eta_pos=[1, 0.1], list_eta_neg=[0.001, 0.0001])
    n.myPlot(list_err_train, list_err_val, combinations)

if __name__ == '__main__':
    run()

# net = n.newNetwork(784, 12, 10)
# n.getInfo(net)

# ld = d.loadDataset('C:/Users/Pietro20/Desktop/mnistDataset/')
# ld = d.loadDataset('C:/Users/anton/Downloads/Telegram_Desktop/')
# print(ld[0].shape)
# print(ld[1].shape)
# print(ld[2].shape)
# print(ld[3].shape)
# print(np.max(ld[0]))

def splitDataset(n=10000):
    x_data = np.loadtxt('C:/Users/Pietro20/Desktop/mnistDataset/mnist_train.csv', delimiter=',')
    y_data = np.loadtxt('C:/Users/Pietro20/Desktop/mnistDataset/mnist_test.csv', delimiter=',')
    # x_data = np.loadtxt('C:/Users/Pietro20/Desktop/mnistDataset/mnist_train_10k.csv', delimiter=',')
    # y_data = np.loadtxt('C:/Users/Pietro20/Desktop/mnistDataset/mnist_test_10k.csv', delimiter=',')

    x_data = x_data[:n+1]
    y_data = y_data[:n+1]

    np.savetxt('C:/Users/Pietro20/Desktop/mnistDataset/mnist_train_10k.csv', x_data, delimiter=',')
    np.savetxt('C:/Users/Pietro20/Desktop/mnistDataset/mnist_test_10k.csv', y_data, delimiter=',')

# print(x_data)
# print(y_data)
# np.savetxt('C:/Users/Pietro20/Desktop/mnistDataset/mnist_sample_train.csv', x_data, delimiter=',')
# np.savetxt('C:/Users/Pietro20/Desktop/mnistDataset/mnist_sample_test.csv', y_data, delimiter=',')

# res = n.forwardPropagation(net, ld[0])
# print(res)

# res, der_res = n.trainForwardPropagation(net, ld[0])
# print('res: ', res, '\nder_res: ', der_res)

# wd, bd = n.backPropagation(net, ld[0], ld[2], ef.crossEntropy)
# print('Weight der: ', wd, '\nBiases der: ', bd)

# et, ev = n.trainBackPropagation(net, ld[0], ld[2], ld[1], ld[3], ef.crossEntropy, 20)
# print('Error train: ', et, '\nError val: ', ev)

# et, 5ev = n.trainResilientPropagation(net, ld[0], ld[2], ld[1], ld[3], ef.crossEntropy, n_epoch=20)
# print('Error train: ', et, '\nError val: ', ev)

# res = n.crossValidationKFold(ld[0], ld[2], ef.crossEntropy, 784, 10, list_hidden_size=[[20,30], 20], list_eta_pos=[1, 0.1], list_eta_neg=[0.01, 0.001], k=3)

# print('Error train: ', res[0], '\nError val: ', res[1])
# n.myPlot(res[0], res[1], res[2])
