import torch
from torch import nn
import torch.nn.functional as F
from utils.unet import UnetArchitecture3d
from utils.enc1_dec1 import E1D1

class MyEnsemble(nn.Module):

    def __init__(self,modelA,modelB):
        super(MyEnsemble,self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(192,96)

    def forward(self, x):
        outA = self.modelA(x)
        outB = self.modelB(x)
        out = torch.cat((outA,outB),dim=4)
        x = self.classifier(F.relu(out))
        return x

if __name__ == '__main__':
    net1 = UnetArchitecture3d().cuda().train()
    net1.print_model_parameters()

    net2 = UnetArchitecture3d().cuda().train()
    net2.print_model_parameters()

    x = torch.randn(3, 4, 96, 96, 96).cuda()
    y1 = net1(x)
    y2 = net2(x)
    print(y1.shape,y2.shape)
    loss1 = y1.sum()
    loss1.backward()
    loss2 = y2.sum()
    loss2.backward()