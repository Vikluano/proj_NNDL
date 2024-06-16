import numpy as np
import myNet as n
import myDataset as d
import myErrorFunct as ef
from itertools import product
import matplotlib.pyplot as plt

def runCrossValid(datapath):
    train_X, test_X, train_lab, test_lab = d.loadDataset(datapath)
    list_err_train, list_err_val, list_acc_train, list_acc_test, combinations = n.crossValidationKFold(X=train_X, 
                                                                                Y=train_lab,
                                                                                test_X=test_X,
                                                                                test_Y=test_lab,
                                                                                err_funct=ef.crossEntropy, 
                                                                                net_input_size=784, 
                                                                                net_output_size=10, 
                                                                                list_hidden_size=[64, 128, 256],
                                                                                eta=0.001,
                                                                                list_eta_pos=[1.1, 1.2, 1.3], 
                                                                                list_eta_neg=[0.6, 0.7, 0.8],
                                                                                k=10,
                                                                                n_epoch=100,
                                                                                write_on_file=True)
    n.myPlot(list_err_train, list_err_val, list_acc_train, list_acc_test, combinations)

def runResilientTrain(datapath):
    train_X, test_X, train_lab, test_lab = d.loadDataset(datapath)
    avg_acc_train = 0
    avg_acc_test = 0
    avg_loss_train = 0
    k=10
    for i in range(k):
        net = n.newNetwork(input_size=784, hidden_size=128, output_size=10)
        train_loss = n.trainResilientPropagation(net=net,
                                    X_t=train_X,
                                    Y_t=train_lab,
                                    err_funct=ef.crossEntropy,
                                    eta_pos=1.3,
                                    eta_neg=0.8,
                                    eta=0.001,
                                    n_epoch=100)
        #n.getInfo(net)
        train_acc = n.testAccuracy(net, train_X, train_lab)
        test_acc = n.testAccuracy(net, test_X, test_lab)
        print(i+1, ' Accuracy train: ', train_acc)
        print(i+1, ' Accuracy test: ', test_acc)
        print(i+1, ' Train loss: ', train_loss[-1])
        avg_acc_train += train_acc
        avg_acc_test += test_acc
        avg_loss_train += train_loss[-1]
    print('Average accuracy train: ', avg_acc_train/k)
    print('Average accuracy test: ', avg_acc_test/k)
    print('Average train loss: ', avg_loss_train/k)
        # y_pred=n.forwardPropagation(net, test_X[:,400:401])
        # y_pred=ef.softMax(y_pred)
        # print('Previsione y:', y_pred)
        # print(test_lab[:,400:401])
    
if __name__ == '__main__':
    #runCrossValid(d.data_path_2)
    runResilientTrain(d.data_path_2)
