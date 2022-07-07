 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
import torchvision.models as models

class NeuralCNN(torch.nn.Module):
    def __init__(self,num_classes =2, num_extra_features =0, dropout_p = 0, freeze_layers = False):
        super(NeuralCNN, self).__init__()
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        self.cnn = models.resnet18(pretrained=True)
        if freeze_layers:
            for param in self.cnn.parameters():
                param.requires_grad = False

        self.cnn.fc = nn.Sequential(nn.Linear(512, 32))
        for param in self.cnn.fc.parameters():
            param.requires_grad = True
        self.bn0 = nn.BatchNorm1d(32)
        self.relu0 = nn.LeakyReLU()

        self.fc = nn.Linear(32 + num_extra_features,32)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(32, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.LeakyReLU()
        

        self.dropout = nn.Dropout(dropout_p)
        if num_classes < 3:
            self.fc_out = nn.Linear(16, 1)
            self.final_ac = nn.Sigmoid()
        else:
            self.fc_out = nn.Linear(16, num_classes)
            self.final_ac = nn.Softmax(dim =-1)

    def forward(self, x, additional_feature = None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        
        batch = self.cnn(x)
        #batch = self.bn0(batch)
        #print(batch.shape)
        if additional_feature != None:
            batch = torch.cat((batch ,additional_feature),1)
        batch = self.dropout(self.bn(self.relu(self.fc(batch))))
        batch = self.dropout(self.bn1(self.relu1(self.fc1(batch))))
        batch = self.final_ac(self.fc_out(batch))
        return batch