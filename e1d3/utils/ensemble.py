import torch
from torch import nn

class MyEnsemble(nn.Module):

    def __init__(self,modelA,modelB):
        super(MyEnsemble,self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(4,2)

    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1,x2),dim=1)
        x = self.classifier(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        return x