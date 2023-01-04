import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy

class DataPrepare(Dataset):
    def __init__(self,inputSize=3,outputSize=3,trainPercent = 0.8,maxLen=10000):
        self.inputSize = inputSize
        self.outputSize = outputSize
        
        self.trainPercent = trainPercent
        self.X1  = torch.zeros(1,inputSize)
        self.X2  = torch.zeros(1,inputSize)
        self.action = torch.zeros(1)
        self.reward = torch.zeros(1)        
        
        self.maxLength = maxLen
        
        self.actionsDict = {}
    
    def __len__(self):
        return self.X1.shape[0]
        
    def __getitem__(self, idx):
        return self.X1[idx,:],self.X2[idx,:],self.action[idx],self.reward[idx]
    
    def add( self , add : list):
        for data in add:
            [state1,action,profit,steps,state2] = data
            
            ff=torch.tensor([*state1]).reshape((1,self.inputSize)).float()
            self.X1 = torch.cat((self.X1,ff), 0)
            ff=torch.tensor([*state2]).reshape((1,self.inputSize)).float()
            self.X2 = torch.cat((self.X2,ff), 0)
            
            self.reward = torch.cat((self.reward,torch.tensor([profit*1.0])))
            
            try :
                self.action = torch.cat((self.action,torch.tensor([self.actionsDict[str(action)]])))
            except :
                self.actionsDict[str(action)] = len(self.actionsDict)*1.0
                self.action = torch.cat((self.action,torch.tensor([self.actionsDict[str(action)]])))
                pass

            
    def selection(self,n):
        if self.X1.shape[0]>n:
            idx=np.random.choice(range(self.X1.shape[0]),size=n,replace=False)
            self.X1 = self.X1[idx]
            self.X2 = self.X2[idx]
            self.action = self.action[idx]
            self.reward = self.reward[idx]

    def addRandom(self,n):
        for i in range(n):
            state1 = (np.random.choice(range(1000),3)*1.1-5)
            state2 = list(1+0.8*state1)
            state1 = list(state1)
        
            period = np.random.choice([90,50,88])
            profit = (np.sum(state1))/100
            
            steps = np.random.choice(100)
            
            self.add([[state1,period,profit,steps,state2]])


        
class NeuralNetwork(nn.Module):
    def __init__(self,hidden_size = 10, input_size=5, output_size = 4):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
        self.lin1 = nn.Linear(input_size , 30)
        self.lin2 = nn.Linear(30, hidden_size)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.lin4 = nn.Linear(hidden_size, output_size)
        # self.lin5 = nn.Linear(hidden_size, hidden_size)
        # self.lin6 = nn.Linear(hidden_size, hidden_size)
        # self.lin7 = nn.Linear(hidden_size, hidden_size)
        # self.lin8 = nn.Linear(hidden_size, output_size)
                
    def forward(self, inputs):
        o=self.lin1(inputs)
        o=F.relu(o)
        o=self.lin2(o)
        o=F.relu(o)
        o=self.lin3(o)
        o=F.relu(o)
        o=self.lin4(o)
        # o=F.relu(o)
        # o=self.lin5(o)
        # o=F.relu(o)
        # o=self.lin6(o)
        # o=F.relu(o)
        # o=self.lin7(o)
        # o=F.relu(o)
        # o=self.lin8(o)
        # o=self.lin4(o)
        # o=self.lin5(o)
        return o

class RLAgent:
    def __init__(self,discount_factor=0.9,hidden_size = 10, input_size=5, output_size = 4,learningRate=1e-5,
                 device='cpu',stepSize=1000,gamma=0.98,momentum=0.9):
        self.device = device
        
        self.model = NeuralNetwork(hidden_size , input_size, output_size).to(self.device)
        # self.copyModel = copy.deepcopy(self.model).to(self.device)
        self.makeNewCopy()
        
        self.dp = DataPrepare(input_size,output_size)
#        self.dl = DataLoader(dp, batch_size=batchSize,shuffle=True, num_workers=4)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learningRate,momentum=momentum)
        # self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=learningRate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=stepSize, gamma=gamma)
        
        self.trainLosses=[]
        
        self.learningRate=learningRate
        self.discount_factor = discount_factor

        
    def trainData(self,data=[],epochs=100,batchSize=10):
        self.dp.add(data)
    
        for _ in range(epochs):
            idxs = np.random.choice(len(self.dp),len(self.dp),False)
            sum_loss = 0
            
            for j in range(int(len(self.dp)/batchSize)):
                idx = idxs[j*batchSize:(j+1)*batchSize]
                X1 , X2 , action , reward = self.dp[idx]
                
                q_value = self.model.forward(X2.to(self.device))
                q_value=q_value.detach()
                a=action.long()
                b=torch.arange(batchSize).long()
                
                # reward function 
                # q_value[(b,a)]=(self.discount_factor*q_value[(b,a)]+reward.to(self.device)).float()
                # q_value=q_value.detach()
                
                self.model.train()
                self.optimizer.zero_grad()
                
                q_pred = self.model(X1.to(self.device))
                notZero = torch.prod(X2==0,axis=1)==0
                notZero = torch.arange(notZero.shape[0])[notZero]
                notZero = notZero.long()

                # reward function 
                q=q_pred.clone().detach()
                q[(b,a)]=reward.to(self.device).float()
                try:
                    q[(notZero,a[notZero])]+=(self.discount_factor*torch.max(q[notZero],axis=1).values).float()
                except:
                    print('dastaaaaaan in nn')
                    pass
                
                loss = self.loss_fn(q,q_pred)
                loss.backward()
                self.optimizer.step()
        
                sum_loss += loss.item()
                            
                self.scheduler.step()
            self.trainLosses+=[sum_loss*batchSize/self.dp.X1.shape[0]]

        # self.copyModel = copy.deepcopy(self.model).to(self.device)
        self.makeNewCopy()
        # print(self.trainLosses[-1])
        
    def makeNewCopy(self):
        self.copyModel = copy.deepcopy(self.model).to(self.device)
    
    def testData(self,data):
        self.copyModel.eval()
        data=data.to(self.device)
        return self.copyModel(data)

# rl = RLAgent(learningRate=1e-6,hidden_size = 20,device='cpu',input_size=3,output_size=3,momentum=0.9)
# for _ in range(1):
#     rl.dp.addRandom(1000)
#     rl.trainData([],10,10)
# #     rl.dp.selection(800)

# self=rl

# import matplotlib.pyplot as plt
# plt.plot(rl.trainLosses)
# plt.yscale('log')
# plt.show()

# print(rl.trainLosses)
# rl.copyModel.forward(torch.tensor([6.1,8,5,10,11]))

# rl.dp.X1.shape[0]
# %matplotlib inline        
            
            
#model = NeuralNetwork()
#model.dp.addRandom(100)
#model.trainData([],100,15)
#
#import matplotlib.pyplot as plt
#plt.plot(model.trainLosses)
#plt.yscale('log')

#epochs = 100
#discount_factor = 0.99
#batchSize = 15
#learningRate=1e-5
#
#dp = DataPrepare()
#dp.addRandom(100)
#model = NeuralNetwork()
#dl = DataLoader(dp, batch_size=batchSize,shuffle=True, num_workers=4)
#
#loss_fn = torch.nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
#
#sum_loss=0
#trainLosses=[]
#testLosses=[]
#
#self = model
#
#for _ in range(epochs):
#    idxs = np.random.choice(len(dp),len(dp),False)
#    sum_loss = 0
#    
#    for j in range(int(len(dp)/batchSize)):
#        idx = idxs[j*batchSize:(j+1)*batchSize]
#        X1 , X2 , action , reward = dp[idx]
#        
#        q_value = model.forward(X2)
#        a=action.long()
#        b=torch.arange(batchSize).long()
#        q_value[(b,a)]=discount_factor*q_value[(b,a)]+reward
#        q_value=q_value.detach()
#        
#        model.train()
#        optimizer.zero_grad()
#        
#        q_pred = model(X1)
#        loss = loss_fn(q_value,q_pred)
#        loss.backward()
#        optimizer.step()
#
#        sum_loss += loss.item()
#        
##        print(list(model.parameters()))
#    
#    scheduler.step()
#    trainLosses+=[sum_loss]
