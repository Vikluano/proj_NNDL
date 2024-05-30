import numpy as np

data_path_1 = 'C:/Users/Pietro20/Desktop/mnistDataset/'
data_path_2 = 'C:/Users/anton/Downloads/Telegram_Desktop/'

def loadDataset(datapath='C:/Users/Pietro20/Desktop/'):
    train_set = np.loadtxt(datapath + "mnist_train_10k.csv", delimiter=',')
    test_set = np.loadtxt(datapath + "mnist_test_10k.csv", delimiter=',')
    # train_set = np.loadtxt(datapath + "mnist_sample_train.csv", delimiter=',')
    # test_set = np.loadtxt(datapath + "mnist_sample_test.csv", delimiter=',')

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
