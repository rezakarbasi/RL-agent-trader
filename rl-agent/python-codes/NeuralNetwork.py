import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy

class DataPrepare(Dataset):
    def __init__(self,inputSize=3,actions=3,trainPercent = 0.8,maxLen=10000):
        self.inputSize = inputSize
        self.outputSize = len(actions)
        
        self.trainPercent = trainPercent
        self.X1  = torch.zeros(1,inputSize)
        self.X2  = torch.zeros(1,inputSize)
        self.action = torch.zeros(1)
        self.reward = torch.zeros(1)        
        
        self.maxLength = maxLen
        
        self.actionsDict = {str(float(a)): i for i, a in enumerate(actions)}
    
    def __len__(self):
        return self.X1.shape[0]
        
    def __getitem__(self, idx):
        return self.X1[idx,:],self.X2[idx,:],self.action[idx],self.reward[idx]
    
    def add( self , add : list):
        for data in add:
            [state1,state2,profit,action] = data
            profit = profit[0]
            action = action[0]
            
            ff=torch.tensor([*state1]).reshape((1,self.inputSize)).float()
            self.X1 = torch.cat((self.X1,ff), 0)
            ff=torch.tensor([*state2]).reshape((1,self.inputSize)).float()
            self.X2 = torch.cat((self.X2,ff), 0)
            
            self.reward = torch.cat((self.reward,torch.tensor([profit*1.0])))
            
            self.action = torch.cat((self.action,torch.tensor([self.actionsDict[str(action)]])))

            
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
    
        self.lin1 = nn.Linear(input_size , hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.lin4 = nn.Linear(hidden_size, output_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(output_size)
                
    def forward(self, inputs):
        o=self.lin1(inputs)
        o=self.bn1(o)
        o=F.relu(o)

        o=self.lin2(o)
        o=self.bn2(o)
        o=F.relu(o)

        o=self.lin3(o)
        o=self.bn3(o)
        o=F.relu(o)

        o=self.lin4(o)
        o=self.bn4(o)
        
        return o

class RLAgent:
    def __init__(self,discount_factor=0.9,hidden_size = 10, input_size=5, actions = [1, 2, 3],learningRate=1e-5,
                 device='cpu',stepSize=1000,gamma=0.98,momentum=0.9):
        output_size = len(actions)
        self.device = device
        
        self.model = NeuralNetwork(hidden_size , input_size, output_size).to(self.device)
        # self.copyModel = copy.deepcopy(self.model).to(self.device)
        self.makeNewCopy()
        
        self.dp = DataPrepare(input_size,actions=actions)
#        self.dl = DataLoader(dp, batch_size=batchSize,shuffle=True, num_workers=4)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learningRate,momentum=momentum)
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
                    print('ERROR in nn')
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
