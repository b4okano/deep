import argparse

import torch

from model import CNNet, FCNet, CommonNet
from dataIO import DataIO
from learning import Learning

parameterFileBody = 'params'

def Main(args):
    # CUDA(GPU) or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: ' + device)

    # select model
    model = args.model

    # neural network
    # net : CommonNet = CNNet() # CNN
    if model == 'cnn':
        net : CommonNet = CNNet() # CNN
    else:
        net : CommonNet = FCNet()  # FCNet
    net.to(device)

#    print('model: ' + model)
#    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)

    epoch = args.epoch

    dataIO = DataIO()

    learning = Learning(net, device, epoch, dataIO)

    if args.finetuning:
        learning.load(parameterFileBody+ '-' + model + '.pth')


    if args.finetuning:
        trainLoaders = dataIO.loadData(args.trainDir)
        testLoaders  = dataIO.loadData(args.testDir)
        optimizer = torch.optim.Adam(params=net.parameters(), lr=0.00001)
        learning.setOptimizer(optimizer)
    else:
        trainLoaders = dataIO.loadDataMNISTTrain(args.mnistDir)
        testLoaders  = dataIO.loadDataMNISTVerify(args.mnistDir)

#    print(net)

    dataIO.cleanDirectory()

    # run learning and testing
    for ep in range(epoch):
        learning.train(trainLoaders, ep)
        learning.test(testLoaders, ep)

    learning.save(parameterFileBody + '-' + model + '.pth')

    # output the graphs
    historyTrain, historyTest = learning.getHistories()
    dataIO.outputGraph(epoch, historyTrain, historyTest)
    return


if __name__ == '__main__':
    """
    Learning Program

    trainDir : directory path for train images
    testDir  : directory path for test images
    epoch    : epoch
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--trainDir", type=str, default='number_generated/train/')
    parser.add_argument("-ts", "--testDir", type=str, default='number_generated/verify/')
    parser.add_argument("-md", "--mnistDir", type=str, default='MNIST')

    parser.add_argument("-ep", "--epoch", type=int, default=10)
    parser.add_argument("-mdl", "--model", type=str, default="cnn") # cnn or fcn
    parser.add_argument("-ft", "--finetuning", type=bool, default=False)
    args = parser.parse_args()

    Main(args)
