# coding: utf8

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import datareshape

torch.set_default_tensor_type(torch.DoubleTensor)
import scipy.io as sio
import CNN_selfattention
import time

import model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_tensor_type(torch.DoubleTensor)





batch_size = 64
learning_rate = 0.1
p_d=20
p_p=20
t_p=2000
par=0
Q=16
I=16
M=4
N=2



data_input=datareshape.list_data_1
data_input=np.abs(data_input)
data_loss_input=datareshape.data_1



train_data_input_square_torch=torch.tensor(data_input)
train_data_input_square_torch_T=train_data_input_square_torch.permute(0,3,2,1)
train_data = DataLoader(train_data_input_square_torch_T, batch_size=batch_size, shuffle=True)


train_data_loss_input_torch=torch.tensor(data_loss_input)


model = model.DS_CNN(M,N)
if torch.cuda.is_available():
    model = model.cuda()




optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, foreach=False )

print('starting')
loss_1=[]
T=[]
epoch = 0
z_0=[]
for data in train_data:
    if epoch<(len(train_data)-1):




        data_loss = train_data_loss_input_torch[
                    epoch * batch_size * datareshape.N: epoch * batch_size * datareshape.N + batch_size * datareshape.N,
                    :]





        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        data=data.to(device)
        data=data.cuda()
        data_loss=data_loss.to(device)
        data_loss=data_loss.cuda()


        start_time=time.time()
        out= model(data)

        s_out=torch.ones((batch_size,M,Q,I))

        loss,n_zero = CNN_selfattention.my_ob(out, data_loss,Q,I,M,N,batch_size,0,0,0,0, device)

        print('loss',loss)
        print('n_zero', n_zero)
        print_loss = loss.data.item()


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        end_time=time.time()
        print("Total number of paramerters in networks is {}  ".format(model.parameters()))



        print('loss', loss.grad)
        t=(end_time-start_time)
        loss_1.append(print_loss)
        T.append(t)
        z_0.append(n_zero)

        epoch+=1
        # if epoch%50 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
        print("time: {:.15f}秒".format(t))




plt.figure()
plt.plot(loss_1)
plt.ylabel("Training loss")
plt.xlabel("Epoch")
plt.show()

plt.figure()
plt.plot(z_0)
plt.ylabel("Training loss")
plt.xlabel("Epoch")
plt.show()
