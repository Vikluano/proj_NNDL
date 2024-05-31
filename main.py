import myNet as n
import myDataset as d
import myErrorFunct as ef

def run(datapath):
    train_X, test_X, train_lab, test_lab = d.loadDataset(datapath)
    list_err_train, list_err_val, list_acc_train, list_acc_test, combinations = n.crossValidationKFold(X=train_X, 
                                                                        Y=train_lab,
                                                                        test_X=test_X,
                                                                        test_Y=test_lab,
                                                                        err_funct=ef.crossEntropy, 
                                                                        net_input_size=784, 
                                                                        net_output_size=10, 
                                                                        list_hidden_size=[50], 
                                                                        eta=0.001,
                                                                        list_eta_pos=[0.01, 0.001], 
                                                                        list_eta_neg=[0.00001, 0.0001], 
                                                                        k=10,
                                                                        n_epoch=100)
    n.myPlot(list_err_train, list_err_val, list_acc_train, list_acc_test, combinations)

if __name__ == '__main__':
    run(d.data_path_1)
