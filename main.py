import numpy as np
import myNet as n
import myDataset as d
#import myActFunct as af
import myErrorFunct as ef
import mynetlib as proflib
import matplotlib.pyplot as plt

def run(datapath):
    train_X, test_X, train_lab, test_lab = d.loadDataset(datapath)
    # ld = d.loadDataset('C:/Users/anton/Downloads/Telegram_Desktop/')
    list_err_train, list_err_val, list_acc_train, list_acc_val, combinations = n.crossValidationKFold(train_X, 
                                                                                                    train_lab, 
                                                                                                    ef.crossEntropy, 
                                                                                                    784, 
                                                                                                    10, 
                                                                                                    list_hidden_size=[64], 
                                                                                                    list_eta_pos=[1, 0.1], 
                                                                                                    list_eta_neg=[0.001, 0.0001], 
                                                                                                    k=10) #aggiuntilist_acc_train, list_acc_val,
    # list_err_train, list_err_val, combinations = n.crossValidationKFold(train_X, train_lab, ef.crossEntropy, 784, 10, list_hidden_size=[60], 
    #                                                                     list_eta_pos=[1, 0.1], list_eta_neg=[0.001, 0.0001])
    n.myPlot(list_err_train, list_err_val, combinations)
    n.myPlot2(list_acc_train, list_acc_val, combinations)

def splitDataset(datapath, n=10000):
    x_data = np.loadtxt(datapath + 'mnist_train.csv', delimiter=',', skiprows=1)
    y_data = np.loadtxt(datapath + 'mnist_test.csv', delimiter=',', skiprows=1)
    # x_data = np.loadtxt('C:/Users/Pietro20/Desktop/mnistDataset/mnist_train_10k.csv', delimiter=',')
    # y_data = np.loadtxt('C:/Users/Pietro20/Desktop/mnistDataset/mnist_test_10k.csv', delimiter=',')

    x_data = x_data[:n]
    y_data = y_data[:n]

    np.savetxt(datapath + 'mnist_train_10k.csv', x_data, delimiter=',')
    np.savetxt(datapath + 'mnist_test_10k.csv', y_data, delimiter=',')


def plot():
    acc_train = [0.6, 0.8]
    acc_valid = [0.5, 0.7]
    combinations = [[80, 1, 0.01],[80, 1, 0.001]]
    n.myPlot2(acc_train, acc_valid, combinations)


if __name__ == '__main__':
    train_X, test_X, train_lab, test_lab = d.loadDataset(d.data_path_2)
    #indT, indV = proflib.splitTrainValDataSet(train_lab)
    # XT=train_X[:,indT]
    # YT=train_lab[:,indT]
    # XV=train_X[:,indV]
    # YV=train_lab[:,indV]
    net = n.newNetwork(784, 50, 10)
    n.getInfo(net)
    err_train, err_val, acc_train, acc_val = n.trainResilientPropagation(net, train_X, train_lab, test_X, test_lab, ef.crossEntropy, n_epoch=20, eta=0.001, eta_pos=0.01, eta_neg=0.0001)
    plt.figure 
    plt.plot(err_train,'r')
    plt.plot(err_val,'b')
    plt.show()
    #plot()
    #splitDataset(d.data_path_2)
    #run(d.data_path_2)


# net = n.newNetwork(784, 12, 10)
# n.getInfo(net)

# ld = d.loadDataset('C:/Users/Pietro20/Desktop/mnistDataset/')
# ld = d.loadDataset('C:/Users/anton/Downloads/Telegram_Desktop/')
# print(ld[0].shape)
# print(ld[1].shape)
# print(ld[2].shape)
# print(ld[3].shape)
# print(np.max(ld[0]))



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
