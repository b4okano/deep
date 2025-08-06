import torch
import torch.nn.functional as func

from model import CommonNet

class Learning:

    def __init__(self, net, device, epoch, dataIO):
        self.optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.epoch = epoch
        self.net = net
        self.device = device
        self.learning_history = {
         'loss': [],
         'accuracy': [],
        }
        self.testing_history = {
         'loss': [],
         'accuracy': [],
        }
        self.dataIO = dataIO


    def train(self, loaders, e):

        loss = None
        loss_sum = 0
        train_correct_counter = 0
        numTrain = 0

        self.net.train(True)

        for i, (data, labels) in enumerate(loaders):
            # GPU or CPU
            data = data.to(self.device)
            labels = labels.to(self.device)
            numTrain += len(labels)

            # learning
            self.optimizer.zero_grad()   # reset weights
            output = self.net(data)      # predicate

            # loss = func.nll_loss(output, labels)
            loss = self.criterion(output, labels)
            loss_sum += loss.item() * data.size()[0]

            loss.backward()         # back propergate
            self.optimizer.step()   # update

            train_pred = output.argmax(dim=1, keepdim=True)
            train_correct_counter += train_pred.eq(labels.view_as(train_pred)).sum().item() # 推論と答えを比較し、正解数を加算

            # progress (per 8 batch)
            if i % 8 == 0:
                print('Training log: epoch {} ({} / {}). Loss: {}'.format(e+1, (i+1)*loaders.batch_size, numTrain, loss.item()))

        #ave_loss = loss_sum / data_size
        #ave_accuracy = train_correct_counter / data_size
        ave_loss = loss_sum / numTrain
        ave_accuracy = train_correct_counter / numTrain
        self.learning_history['loss'].append(ave_loss)
        self.learning_history['accuracy'].append(ave_accuracy)
        print(f"Train Loss: {ave_loss} , Accuracy: {ave_accuracy}")

        return

    def test(self, loaders, e):
        self.net.eval()

        loss = None
        loss_sum = 0
        test_correct_counter = 0
        data_num = 0
        numTest = 0

        history = {
         'loss': [],
         'accuracy': [],
        }

        with torch.no_grad():
           for data, labels in loaders:
                # GPU or CPU
                data = data.to(self.device)
                labels = labels.to(self.device)
                numTest += len(labels)

                output = self.net(data)
                loss = self.criterion(output, labels)
                loss_sum += loss.item() * data.size()[0]
                #loss_sum += func.nll_loss(output, labels, reduction='sum').item()

                test_pred = output.argmax(dim=1, keepdim=True)
                test_correct_counter += test_pred.eq(labels.view_as(test_pred)).sum().item() # 推論と答えを比較し、正解数を加算

                # from only the last epoch we retrive NG images
                if e == self.epoch - 1:
                    self.dataIO.last_epoch_NG_output(data, test_pred, labels, data_num)
                    data_num += loaders.batch_size

        ave_loss = loss_sum / numTest
        ave_accuracy = test_correct_counter / numTest
        self.testing_history['loss'].append(ave_loss)
        self.testing_history['accuracy'].append(ave_accuracy)
        print(f'Test Loss: {ave_loss} , Accuracy: {ave_accuracy}\n')

        return

    def getHistories(self):
        return self.testing_history, self.learning_history


    def save(self, name):
        torch.save(self.net.state_dict(), name)
#        torch.save(self.optimizer.state_dict(), 'opt-' + name)
        return

    def load(self, name):
        params = torch.load(name, map_location=torch.device(self.device))
        self.net.load_state_dict(params)
#        params = torch.load('opt-' + name, map_location=torch.device(self.device))
#        self.optimizer.load_state_dict(params)
        self.net.to(self.device) # reconfigure the net depending on the device (GPU or CPU)
        return

    def setOptimizer(self, opt):
        self.otimizer = opt
        return