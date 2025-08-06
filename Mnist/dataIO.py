import os
import glob

import cv2
import torch
import torchvision
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

class DataIO:
    def loadData(self, dir_path):
        img_paths = os.path.join(dir_path, "*.png")
        img_path_list = glob.glob(img_paths)

        data = []
        labels = []

        for img_path in img_path_list:
            img = TF.to_tensor(cv2.imread(img_path,cv2.IMREAD_GRAYSCALE))

            # to add the img to data, temporally change into ndaarray type
            data.append(img.detach().numpy())

            # answer label into labels
            begin = img_path.find('ans') + len('ans')
            tail = img_path.find('.png')
            ans = int(img_path[begin:tail])
            labels.append(ans)

        # Again convert the data into tensor in oder to manage with PyTorch
        data = torch.tensor(np.array(data))
        labels = torch.tensor(labels)

        # pair (data, labels) is into dataset
        dataset = torch.utils.data.TensorDataset(data, labels)

        # shuflle into mini batches
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)


        #data_size = len(img_path_list)

        return loader #, data_size


    def loadDataMNISTTrain(self, path):
        train_dataset = torchvision.datasets.MNIST(root=path,
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download = True)
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        return loader

    def loadDataMNISTVerify(self, path):
        verify_dataset = torchvision.datasets.MNIST(root=path,
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download = True)
        loader = torch.utils.data.DataLoader(verify_dataset, batch_size=128, shuffle=True)
        return loader


    def cleanDirectory(self):
        dir_path = './NG_figure_CNN'
        if os.path.isdir(dir_path):
            import shutil
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)



    def last_epoch_NG_output(self, data, test_pred, target, counter):

        dir_path = "./NG_figure_CNN"

        for i, img in enumerate(data):
            pred_num = test_pred[i].item()
            ans = target[i].item()

            if pred_num != ans:
                data_num = str(counter+i).zfill(5)
                img_name = f"{data_num}-pre-{pred_num}-ans-{ans}.jpg"
                fname = os.path.join(dir_path, img_name)

                torchvision.utils.save_image(img, fname)

        return

    def outputGraph(self, epoch, history_train, history_test):
        dir_path = './CNNLearningResult'
        os.makedirs(dir_path, exist_ok=True)

        # loss graph
        plt.figure()
        plt.plot(range(1, epoch+1), history_train['loss'], label='train loss', marker='.')
        plt.plot(range(1, epoch+1), history_test['loss'], label='test loss', marker='.')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(dir_path,'loss_cnn.png'))

        # accuracy graph
        plt.figure()
        plt.plot(range(1, epoch+1), history_train['accuracy'], label='train accuracy', marker='.')
        plt.plot(range(1, epoch+1), history_test['accuracy'], label='test accuracy', marker='.')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(dir_path,'acc_cnn.png'))

        return
