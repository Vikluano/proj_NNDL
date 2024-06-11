import numpy as np

data_path_1 = 'C:/Users/Pietro20/Desktop/proj_NNDL/mnistDataset/'
data_path_2 = 'C:/Users/anton/git_workspace/proj_NNDL/mnistDataset/'

def loadDataset(datapath='C:/Users/Pietro20/Desktop/'):
    train_set = np.loadtxt(datapath + "mnist_train.csv", delimiter=',', skiprows=1)
    test_set = np.loadtxt(datapath + "mnist_test.csv", delimiter=',', skiprows=1)

    train_X_norm = (train_set[:, 1:]) / 255 
    test_X_norm = (test_set[:, 1:]) / 255

    train_Y = train_set[:, 0]
    test_Y = test_set[:, 0]

    train_lab = oneHotEnc(train_Y)
    test_lab = oneHotEnc(test_Y)

    return train_X_norm.transpose(), test_X_norm.transpose(), train_lab.transpose(), test_lab.transpose()

def oneHotEnc(labels_set):
    row = len(labels_set)
    col = 10 
    
    sparse_matrix = np.zeros((row, col), dtype=int)
    labels = labels_set.astype(int)

    for i in range(row):
        sparse_matrix[i, labels[i]] = 1

    return sparse_matrix
