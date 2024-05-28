import numpy as np
import myNet as n
import myDataset as d
import myActFunct as af
import myErrorFunct as ef
import matplotlib.pyplot as plt

def run(datapath):
    train_X, test_X, train_lab, test_lab = d.loadDataset(datapath)
    list_err_train, list_err_val, combinations = n.crossValidationKFold(X=train_X, 
                                                                        Y=train_lab, 
                                                                        err_funct=ef.crossEntropy, 
                                                                        net_input_size=784, 
                                                                        net_output_size=10, 
                                                                        list_hidden_size=[64], 
                                                                        eta=0.001,
                                                                        list_eta_pos=[0.01], 
                                                                        list_eta_neg=[0.00001], 
                                                                        k=10,
                                                                        n_epoch=50)
    n.myPlot(list_err_train, list_err_val, combinations)

if __name__ == '__main__':
    train_X, test_X, train_lab, test_lab = d.loadDataset(d.data_path_2)
    net = n.newNetwork(784, 50, 10)
    n.getInfo(net)
    err_train, err_val = n.trainBackPropagation(net=net,
                                                X_t=train_X,
                                                Y_t=train_lab,
                                                X_v=test_X,
                                                Y_v=test_lab,
                                                err_funct=ef.crossEntropy,
                                                n_epoch=150,
                                                eta=0.0001)
    # err_train, err_val = n.trainResilientPropagation(net, train_X, train_lab, test_X, test_lab, ef.crossEntropy, n_epoch=20, eta=0.001, eta_pos=0.01, eta_neg=0.0001)
    # plt.figure 
    # plt.plot(err_train,'r')
    # plt.plot(err_val,'b')
    # plt.show()
    #plot()
    #splitDataset(d.data_path_2)
    #run(d.data_path_2)
