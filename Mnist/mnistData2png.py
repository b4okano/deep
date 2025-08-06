import os
from torchvision import datasets
import argparse

def main(miter):
    print('start')
    rootdir = './number_generated'
    traindir = os.path.join(rootdir, 'train')
    testdir = os.path.join(rootdir, 'verify')

    train_dataset = datasets.MNIST(root=rootdir, train=True, download=True)
    test_dataset = datasets.MNIST(root=rootdir, train=False, download=True)

    count = 0
    for img, label in train_dataset:
        os.makedirs(traindir, exist_ok=True)
        img_name = 'mnist' + str(count).zfill(5) + '_ans' + str(label) + '.png'
        savepath = os.path.join(traindir, img_name)
        img.save(savepath)
        count += 1
        if count > miter:
            break
        print(savepath)

    count = 0
    for img, label in test_dataset:
        os.makedirs(testdir, exist_ok=True)
        img_name = 'mnist' + str(count).zfill(5) + '_ans' + str(label) + '.png'
        savepath = os.path.join(testdir, img_name)
        img.save(savepath)
        count += 1
        if count > miter:
            break
        print(savepath)

    print('finished')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-max', '--max', type=int, default=60000)

    args = parser.parse_args()
    maxiter = args.max
    main(maxiter)
