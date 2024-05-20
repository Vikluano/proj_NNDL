import numpy as np

def loadDataset(datapath='C:/Users/Pietro20/Desktop/proj_NNDL/'):
    # train_set = np.loadtxt(datapath + "mnist_train.csv", delimiter=',', skiprows=1)
    # test_set = np.loadtxt(datapath + "mnist_test.csv", delimiter=',', skiprows=1)
    train_set = np.loadtxt(datapath + "mnist_sample_train.csv", delimiter=',')
    test_set = np.loadtxt(datapath + "mnist_sample_test.csv", delimiter=',')

    train_X_norm = (train_set[:, 1:]) / 255 # Tutte le colonne tranne la prima
    test_X_norm = (test_set[:, 1:]) / 255

    train_Y = train_set[:, 0] # Solo la prima colonna
    test_Y = test_set[:, 0]

    train_lab = oneHotEnc(train_Y)
    test_lab = oneHotEnc(test_Y)

    return train_X_norm.transpose(), test_X_norm.transpose(), train_lab.transpose(), test_lab.transpose()

def oneHotEnc(labels_set):
    row = len(labels_set) # Numero di array
    col = 10 # Numero di classi (0, ..., 9)
    
    sparse_matrix = np.zeros((row, col), dtype=int)
    labels = labels_set.astype(int)

    for i in range(row):
        sparse_matrix[i, labels[i]] = 1

    return sparse_matrix
