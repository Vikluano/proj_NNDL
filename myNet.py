import numpy as np
import myActFunct as act
from itertools import product
import matplotlib.pyplot as plt
import myErrorFunct as ef

def newNetwork(input_size, hidden_size, output_size, list_act_funct=[]):    
    sigma = 0.1
    biases = []
    weights = []
    act_funct = []
    prev_layer = input_size
    
    if np.isscalar(hidden_size):
        hidden_size = [hidden_size]

    for layer in hidden_size: 
        biases.append(sigma * np.random.normal(size = [layer, 1]))
        weights.append(sigma * np.random.normal(size = [layer, prev_layer]))
        prev_layer = layer
    act_funct = setActFunct(len(weights), list_act_funct)
    biases.append(sigma * np.random.normal(size = [output_size, 1]))
    weights.append(sigma * np.random.normal(size = [output_size, prev_layer]))
    act_funct.append(act.identity)
    net = {'Weights':weights,'Biases':biases,'ActFun':act_funct,'Depth':len(weights)}
    
    return net

def setActFunct(depth, list_act_funct, act_def=act.tanh):
    if not list_act_funct:
        return [act_def for _ in range(0, depth)]
    elif len(list_act_funct) < depth:
        act_list_1 = [list_act_funct[i] for i in range(0, len(list_act_funct))]
        act_list_2 = [act_def for _ in range(len(list_act_funct), depth)]
        return act_list_1 + act_list_2
    else:
        raise Exception("Exception: Too many item in the activation function list\n")
    
def getInfo(net):
    hidden_layers = net['Depth'] - 1
    print("Depth network:", net["Depth"])
    print("Number of input neurons:", net["Weights"][0].shape[1])
    print("Number of hidden layers:", hidden_layers)
    print("Number of hidden neurons:", [net["Weights"][layer].shape[0] for layer in range(0, hidden_layers)])
    print("Number of output neurons:", net["Weights"][(net["Depth"] - 1)].shape[0])
    print("Weights shape:", [net["Weights"][i].shape for i in range(0, (1 + hidden_layers + 1) - 1)])
    print("Activation functions:", [(net["ActFun"][i]).__name__ for i in range(0, net["Depth"])])

def getBiasesList(net):
    return net['Biases']

def getWeightsList(net):
    return net['Weights']

def getActFunList(net):
    return net['ActFun']

def forwardPropagation(net, X):
    B = getBiasesList(net)
    W = getWeightsList(net)
    AF = getActFunList(net)
    d = net['Depth']
    res = X

    for layer in range(d):
        ith_layer = np.matmul(W[layer], res) + B[layer]
        res = AF[layer](ith_layer)

    return res

def trainForwardPropagation(net, X):
    B = getBiasesList(net)
    W = getWeightsList(net)
    AF = getActFunList(net)
    d = net['Depth']
    ith_layer = []
    res = []
    der_act = []
    res.append(X)

    for layer in range(d):
        ith_layer.append(np.matmul(W[layer], res[layer]) + B[layer])
        a, da = AF[layer](ith_layer[layer], 1)
        der_act.append(da)
        res.append(a)

    return res, der_act

def backPropagation(net, X, Y_true, err_funct):
    W = getWeightsList(net)
    d = net['Depth']
    X_list, X_der_list = trainForwardPropagation(net, X)
    delta_list = []
    delta_list.insert(0, err_funct(X_list[-1], Y_true, 1) * X_der_list[-1])
    
    for layer in range(d-1, 0, -1):
        delta = X_der_list[layer-1] * np.matmul(W[layer].transpose(), delta_list[0])
        delta_list.insert(0, delta)

    weight_der = []
    bias_der = []

    for layer in range(0, d):
        der_w = np.matmul(delta_list[layer], X_list[layer].transpose())
        weight_der.append(der_w)
        bias_der.append(np.sum(delta_list[layer], 1, keepdims=True))
    
    return weight_der, bias_der

def trainBackPropagation(net, X_t, Y_t, X_v, Y_v, err_funct, n_epoch=1, eta=0.1):
    err_train = []
    err_val = []
    Y_t_fp = forwardPropagation(net, X_t)
    training_error = err_funct(Y_t_fp, Y_t)
    err_train.append(training_error)
    Y_v_fp = forwardPropagation(net, X_v) 
    validation_error = err_funct(Y_v_fp, Y_v)
    err_val.append(validation_error)
    
    d = net['Depth']
    epoch = 0

    print("Epoch:", epoch, "Training error:", training_error,
            "Accuracy Training:", networkAccuracy(Y_t_fp, Y_t),
            "Validation error:", validation_error,
            "Accuracy Validation:", networkAccuracy(Y_v_fp, Y_v))

    while epoch < n_epoch:
        der_weights, der_biases= backPropagation(net, X_t, Y_t, err_funct)
        
        for layer in range(d):
            net['Weights'][layer] = net['Weights'][layer] - eta * der_weights[layer]
            net['Biases'][layer] = net['Biases'][layer] - eta * der_biases[layer]

        Y_t_fp = forwardPropagation(net, X_t)
        training_error = err_funct(Y_t_fp, Y_t)
        err_train.append(training_error)
        Y_v_fp = forwardPropagation(net, X_v)
        validation_error = err_funct(Y_v_fp, Y_v)
        err_val.append(validation_error)

        epoch += 1

        print("Epoch:", epoch, "Training error:", training_error,
            "Accuracy Training:", networkAccuracy(Y_t_fp, Y_t),
            "Validation error:", validation_error,
            "Accuracy Validation:", networkAccuracy(Y_v_fp, Y_v), end='')
        print('\r', end='') 
    print()
    
    return err_train, err_val

def trainResilientPropagation(net, X_t, Y_t, X_v=None, Y_v=None, err_funct=ef.crossEntropy, eta_pos=1, eta_neg=0.01, eta=0.1, n_epoch=1, alpha=1.2, beta=0.5):
    err_train = []
    err_val = []
    acc_train = []
    acc_val = []
    der_w_list = []
    der_b_list = []
    
    d  = net['Depth']
    epoch = 0
    eta_ij = eta * d 

    Y_t_fp = forwardPropagation(net, X_t)
    training_error = err_funct(Y_t_fp, Y_t)
    err_train.append(training_error)
    training_accuracy = networkAccuracy(Y_t_fp, Y_t)
    acc_train.append(training_accuracy)        


    if X_v is not None:
        Y_v_fp = forwardPropagation(net, X_v)
        validation_error = err_funct(Y_v_fp, Y_v)
        err_val.append(validation_error)
        validation_accuracy = networkAccuracy(Y_t_fp, Y_t)
        acc_val.append(validation_accuracy)

        print("Epoch:", epoch, "Training error:", training_error,
                "Accuracy Training:", networkAccuracy(Y_t_fp, Y_t),
                "Validation error:", validation_error,
                "Accuracy Validation:", networkAccuracy(Y_v_fp, Y_v))
    else:
        print("Epoch:", epoch, "Training error:", training_error,
            "Accuracy Training:", networkAccuracy(Y_t_fp, Y_t))

    while epoch < n_epoch:
        der_weights, der_biases = backPropagation(net, X_t, Y_t, err_funct)
        der_w_list.append(der_weights)
        der_b_list.append(der_biases)

        for layer in range(d):
            if epoch > 0:
                neurons = net['Weights'][layer].shape
                for n in range(neurons[0]):
                    for i in range(neurons[1]):
                        prod_der_w = der_w_list[epoch-1][layer][n][i] * der_w_list[epoch][layer][n][i]
                        if prod_der_w > 0:
                            eta_ij = min(eta_ij * alpha, eta_pos)
                        elif prod_der_w < 0:
                            eta_ij = max(eta_ij * beta, eta_neg)
                        
                        net['Weights'][layer][n][i] -= eta_ij * np.sign(der_weights[layer][n][i])
                    
                    prod_der_b = der_b_list[epoch-1][layer][n] * der_b_list[epoch][layer][n]
                    if prod_der_b > 0:
                        eta_ij = min(eta_ij * alpha, eta_pos)
                    elif prod_der_b < 0:
                        eta_ij = max(eta_ij * beta, eta_neg)
                    
                    net['Biases'][layer][n] -= eta_ij * np.sign(der_biases[layer][n])

        Y_t_fp = forwardPropagation(net, X_t)
        training_error = err_funct(Y_t_fp, Y_t)
        err_train.append(training_error)

        epoch += 1

        if X_v is not None:
            Y_v_fp = forwardPropagation(net, X_v)
            validation_error = err_funct(Y_v_fp, Y_v)
            err_val.append(validation_error)

            print("Epoch:", epoch, "Training error:", training_error,
                "Accuracy Training:", networkAccuracy(Y_t_fp, Y_t),
                "Validation error:", validation_error,
                "Accuracy Validation:", networkAccuracy(Y_v_fp, Y_v), end='')
        else:
            print("Epoch:", epoch, "Training error:", training_error,
                "Accuracy Training:", networkAccuracy(Y_t_fp, Y_t), end='')
        print('\r', end='') 
    print()

    return err_train, err_val

def networkAccuracy(Y, Y_true):
    tot = Y_true.shape[1]
    true_positive = 0
    for i in range(0, tot):
        true_label = np.argmax(Y_true[:, i])
        y_label = np.argmax(Y[:, i])
        if true_label == y_label:
            true_positive += 1
    return true_positive / tot

def testAccuracy(net, test_X, test_Y):
    net_Y = forwardPropagation(net, test_X)
    return networkAccuracy(net_Y, test_Y)
    
def crossValidationKFold(X, Y, test_X, test_Y, err_funct, net_input_size, net_output_size, list_hidden_size=[], list_eta_pos=[], list_eta_neg=[], eta=0.1, k=10, n_epoch=50):
    combinations = list(product(list_hidden_size, list_eta_pos, list_eta_neg))
    samples_dim = Y.shape[1]
    list_err_train = []
    list_err_val = []
    list_acc_train = []
    list_acc_test = []
    if (samples_dim % k) == 0:
        d = Y.shape[0]
        partitions = [np.ndarray(0,int) for i in range(k)]
        for lab in range(d):
            bool_vector=(Y.argmax(0)==lab)
            index_vector=np.argwhere(bool_vector==True).flatten()
            lab_partition=np.array_split(index_vector,k)
            for sample in range(len(partitions)):
                partitions[sample]=np.append(partitions[sample],lab_partition[sample])
        for i in range(k):
            partitions[i]=np.random.permutation(partitions[i]) 
        count = 1
        n_combination = len(combinations)
        for combination in combinations:
            s_acc_train = 0
            s_acc_test = 0
            net = newNetwork(input_size=net_input_size, hidden_size=combination[0], output_size=net_output_size, list_act_funct=[])
            getInfo(net)
            print('\nIperparametri selezionati:\n-hidden size ', combination[0], '\n-eta+ ', combination[1], '\n-eta- ', combination[2], '\n')
            print(count, '/', n_combination, ' Combinazioni iperparametri\n')
            for v in range(k):
                s_err_train = 0
                s_err_val = 0
                XV = X[:, partitions[v]]
                YV = Y[:, partitions[v]]
                train_partition = partitions.copy()
                del train_partition[v]
                t_index = np.concatenate(train_partition)
                XT = X[:, t_index]
                YT = Y[:, t_index]
                net = newNetwork(input_size=net_input_size, hidden_size=combination[0], output_size=net_output_size, list_act_funct=[])
                err_train, err_val = trainResilientPropagation(net, 
                                                                XT, YT, 
                                                                XV, YV, 
                                                                err_funct, 
                                                                eta_pos=combination[1], 
                                                                eta_neg=combination[2], 
                                                                eta=eta, 
                                                                n_epoch=n_epoch)
                print('\n', v+1, '/', k, ' Partizioni analizzate\n')

                s_err_train += err_train[-1]
                s_err_val += err_val[-1]
                acc_train = testAccuracy(net, X, Y)
                acc_test = testAccuracy(net, test_X, test_Y)
                s_acc_train += acc_train
                s_acc_test += acc_test
                print('Accuracy train set partizione:', acc_train)
                print('Accuracy test set partizione:', acc_test)
                print()

            list_err_train.append(s_err_train/k)
            list_err_val.append(s_err_val/k)
            avg_acc_train = s_acc_train/k
            avg_acc_test = s_acc_test/k
            print('Combinazione numero', count, 'accuracy train set:', avg_acc_train)
            print('Combinazione numero', count, 'accuracy test set:', avg_acc_test)
            print()
            list_acc_train.append(avg_acc_train)
            list_acc_test.append(avg_acc_test)
            count += 1

        print('List loss train:', list_err_train)
        print('List loss val:', list_err_val)
        print('List avg acc train: ', list_acc_train)#
        print('List avg acc test: ', list_acc_test)#
            
        return list_err_train, list_err_val, list_acc_train, list_acc_test, combinations

    else:
        raise Exception("Exception: each fold must be the same size")
    
def myPlot(list_err_train, list_err_val, list_acc_train, list_acc_test, combinations):
    x_plot_lab = []
    for c in combinations:
        x_plot_lab.append(str(c))

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x_plot_lab, list_err_train, color='r', label='Train loss')
    axs[0].plot(x_plot_lab, list_err_val, color='g', label='Validation loss')
    axs[0].set_title("Loss")

    axs[1].set_title("Accuracy")
    axs[1].plot(x_plot_lab, list_acc_train, color='r', label='Train accuracy')
    axs[1].plot(x_plot_lab, list_acc_test, color='g', label='Test accuracy')
    fig.tight_layout()
    plt.show()
