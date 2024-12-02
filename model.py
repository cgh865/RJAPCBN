from torch import nn
import torch



def AT(k,t,v):

    return 1/(1+torch.exp(-k*(v-t)))




class DS_CNN(nn.Module):
    def __init__(self,M,N):
        super(DS_CNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(M * N, 2 * M, kernel_size=[3, 3], padding="same"), nn.BatchNorm2d(2 * M), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Conv2d(2 * M, 2 * M, kernel_size=[3, 3], padding="same"), nn.BatchNorm2d(2 * M), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Conv2d(2 * M, 2 * M, kernel_size=[3, 3], padding="same"), nn.BatchNorm2d(2 * M), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Conv2d(2 * M, 2 * M, kernel_size=[3, 3], padding="same"), nn.BatchNorm2d(2 * M), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Conv2d(2 * M, 2 * M, kernel_size=[3, 3], padding="same"), nn.BatchNorm2d(2 * M), nn.Tanh())

        self.layer6 = nn.Sequential(nn.Conv2d(M * N, 2 * M, kernel_size=[1, 1], padding="same"), nn.BatchNorm2d(2 * M), nn.Tanh())
        self.layer_a=nn.ReLU()


        self.layer7 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=[1, 1], padding="same"), nn.BatchNorm2d(1),nn.ReLU(True))


    def forward(self, x):
        x1 = self.layer1(x)
        # print(np.shape(x1))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x)

        x7=self.layer_a(x5+x6)

        # x_avg = torch.mean(x, dim=(1))
        # s_x_avg=np.shape(x_avg)
        # x_t=self.layer7(torch.reshape(x_avg,(s_x_avg[0],1,s_x_avg[1],s_x_avg[2])))
        # x7_avg=torch.mean(x7, dim=(1))
        # s_x7_avg = np.shape(x7_avg)
        # x7_avg=torch.reshape(x7_avg,(s_x7_avg[0],1,s_x7_avg[1],s_x7_avg[2]))
        #
        # x8= AT(20,x7_avg, x_t)
        # x9=x8*x7






        return x7

