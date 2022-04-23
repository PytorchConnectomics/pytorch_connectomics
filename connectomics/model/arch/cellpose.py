from turtle import forward
import torch.nn as nn

class Cellpose(nn.Module):

    def __init__(self,**_):
        super(Cellpose, self).__init__()
        self.model = nn.Conv2d( 1, 3, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        # print(x.shape)
        return x