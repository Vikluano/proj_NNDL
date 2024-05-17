import numpy as np
import myActFunct as act

def newNetwork(input_size, hidden_size, output_size, list_act_funct=[]):
    sigma = 0.1
    biases = []
    weights = []
    act_funct = []
    prev_layer = input_size
    
    if np.isscalar(hidden_size):
        hidden_size = [hidden_size]

    for layer in hidden_size: #aggiungere passo iniziale al range for
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
    elif len(list_act_funct) == 1:
        return [list_act_funct[0] for _ in range(0, depth)]
    elif len(list_act_funct) < depth:
        act_list_1 = [list_act_funct[i] for i in range(0, len(list_act_funct))]
        act_list_2 = [act_def for _ in range(len(list_act_funct), depth)]
        return act_list_1 + act_list_2
    else:
        raise Exception("Exception: Too many item in the activation function list\n")
    
def getInfo(net):
    hidden_layers = net['Depth'] - 1
    print("Depth network: ", net["Depth"])
    print("Number of input neurons: ", net["Weights"][0].shape[1])
    print("Number of hidden layers: ", hidden_layers)
    print("Number of hidden neurons: ", [net["Weights"][layer].shape[0] for layer in range(0, hidden_layers)])
    print("Number of output neurons: ", net["W"][(net["Depth"] - 1)].shape[0])
    print("Weights shape: ", [net["Weights"][i].shape for i in range(0, (1 + hidden_layers + 1) - 1)])
    print("Activation shape: ", [(net["ActFun"][i]).__name__ for i in range(0, net["Depth"])])

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
    delta_list.append(err_funct(X_list[-1], Y_true, 1) * X_der_list[-1])
    
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

def trainBackPropagation(net, X_t, Y_t, X_v, Y_v, err_funct, n_epoch=0, eta=0.1):
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

        # Eventualmente aggiungere if
        print("Epoch: ", epoch, "Training error: ", training_error,
            "Accuracy Training: ", networkAccuracy(Y_t_fp, Y_t),
            "Validation error: ", validation_error,
            "Accuracy Validation: ", networkAccuracy(Y_v_fp, Y_v))

    return err_train, err_val

def trainResilientPropagation(net, X_t, Y_t, X_v, Y_v, err_funct, eta_pos=1, eta_neg=0.01, eta=0.1, n_epoch=0, alpha=1.2, beta=0.5):
    err_train = []
    err_val = []
    Y_t_fp = forwardPropagation(net, X_t)
    training_error = err_funct(Y_t_fp, Y_t)
    err_train.append(training_error)
    Y_v_fp = forwardPropagation(net, X_v)
    validation_error = err_funct(Y_v_fp, Y_v)
    err_val.append(validation_error)
    
    d  = net['Depth']
    epoch = 0
    eta_ij = eta * d

    while epoch < n_epoch:
        der_weights, der_biases= backPropagation(net, X_t, Y_t, err_funct)
        
        for layer in range(d):
            neurons = net['Weights'][layer].size
            for n in range(neurons):
                if net['Weights'][layer][n] > 0:
                    eta_ij = min(eta_ij*alpha, eta_pos)
                elif net['Weights'][layer][n] < 0:
                    eta_ij = max(eta_ij*beta, eta_neg)
                
                net['Weights'][layer][n] = net['Weights'][layer][n] - (eta_ij * np.sign(der_weights[layer][n]))
                net['Biases'][layer][n] = net['Biases'][layer][n] - (eta_ij * np.sign(der_biases[layer][n]))

        Y_t_fp = forwardPropagation(net, X_t)
        training_error = err_funct(Y_t_fp, Y_t)
        err_train.append(training_error)
        Y_v_fp = forwardPropagation(net, X_v)
        validation_error = err_funct(Y_v_fp, Y_v)
        err_val.append(validation_error)

        epoch += 1

        # Eventualmente aggiungere if
        print("Epoch: ", epoch, "Training error: ", training_error,
            "Accuracy Training: ", networkAccuracy(Y_t_fp, Y_t),
            "Validation error: ", validation_error,
            "Accuracy Validation: ", networkAccuracy(Y_v_fp, Y_v))

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

def crossValidationKFold(X, Y, err_funct, list_hidden_size=[], list_eta_pos=[], list_eta_neg=[], eta=0.1, k=10):
    combinations = list(product(list_hidden_size, list_eta_pos, list_eta_neg))
    samples_dim = Y.shape[1]
    if (samples_dim % k) == 0:
        pass
    else:
        raise Exception("Exception: each fold must be the same size\n")
