import torch
import torch.nn.functional as f

class CommonNet(torch.nn.Module):
    def __init__(self, input_dim=(1, 28, 28)):
        super(CommonNet, self).__init__()


    def forward(self, x):
        return x



class CNNet(CommonNet):

    def __init__(self, input_dim=(1, 28, 28)):
        """
        CNN
        """
        super(CommonNet, self).__init__()

        filter_num = 32
#        filter_size = 5 # filter kernel size


#        self.conv1 = torch.nn.Conv2d(in_channels=input_dim[0], out_channels=filter_num, kernel_size=filter_size)


        self.conv1 = torch.nn.Conv2d(1,16,3)
        self.conv2 = torch.nn.Conv2d(16,32,3)

        self.relu = torch.nn.ReLU()

        pooling_size = 2
        self.pool = torch.nn.MaxPool2d(pooling_size, stride=pooling_size)

        # Affine layer
#        fc1_size = (input_dim[1] - filter_size + 1) // pooling_size # self.pool終了時の縦横サイズ = 12
        fc1_size = 5
        self.fc1 = torch.nn.Linear(filter_num * fc1_size * fc1_size, 100)
        self.fc2 = torch.nn.Linear(100, 10)


    def forward(self, x):
        # conv2d - ReLU - Pooling
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # align image data to 1 dim data
        x = x.view(x.size()[0], -1)

        # Affine - ReLU - Affine
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FCNet(CommonNet):

    def __init__(self, input_dim=(1, 28, 28)):
        """
        FCN
        """
        super(CommonNet, self).__init__()

        in_unit, hidden_unit1, hidden_unit2, out_unit = 784, 2048, 1024, 10

        self.fc1 = torch.nn.Linear(in_features=in_unit, out_features=hidden_unit1)
        self.relu1 =  torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=hidden_unit1, out_features=hidden_unit2)
        self.relu2 =  torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(in_features=hidden_unit2, out_features=out_unit)


    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
