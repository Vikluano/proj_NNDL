import numpy as np
import myNet as n
import myDataset as d
import myActFunct as a
import myErrorFunct as ef

net = n.newNetwork(784, 12, 10)
# n.getInfo(net)

ld = d.loadDataset('C:/Users/Pietro20/Desktop/mnistDataset/')
# print(ld[0].shape)
# print(ld[1].shape)
# print(ld[2].shape)
# print(ld[3].shape)
# print(np.max(ld[0]))

# x_data = np.loadtxt('C:/Users/Pietro20/Desktop/mnistDataset/mnist_train.csv', delimiter=',', skiprows=1)
# y_data = np.loadtxt('C:/Users/Pietro20/Desktop/mnistDataset/mnist_test.csv', delimiter=',', skiprows=1)

# x_data = x_data[:6]
# y_data = y_data[:6]

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

et, ev = n.trainResilientPropagation(net, ld[0], ld[2], ld[1], ld[3], ef.crossEntropy)
print('Error train: ', et, '\nError val: ', ev)
