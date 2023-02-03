import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import copy

class DataPrepare(Dataset):
    def __init__(self,input_size=3,actions=3,train_percent = 0.8,max_length=10000):
        self.input_size = input_size
        self.outputSize = len(actions)
        
        self.train_percent = train_percent
        self.X1  = torch.zeros(1,input_size)
        self.X2  = torch.zeros(1,input_size)
        self.action = torch.zeros(1)
        self.reward = torch.zeros(1)        
        
        self.maxLength = max_length
        
        self.actions_dictionary = {str(float(a)): i for i, a in enumerate(actions)}
    
    def __len__(self):
        return self.X1.shape[0]
        
    def __getitem__(self, idx):
        return self.X1[idx,:],self.X2[idx,:],self.action[idx],self.reward[idx]
    
    def add( self , add : list):
        for data in add:
            [state1,state2,profit,action] = data
            profit = profit[0]
            action = action[0]
            
            ff=torch.tensor([*state1]).reshape((1,self.input_size)).float()
            self.X1 = torch.cat((self.X1,ff), 0)
            ff=torch.tensor([*state2]).reshape((1,self.input_size)).float()
            self.X2 = torch.cat((self.X2,ff), 0)
            
            self.reward = torch.cat((self.reward,torch.tensor([profit*1.0])))
            
            self.action = torch.cat((self.action,torch.tensor([self.actions_dictionary[str(action)]])))

            
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

class ReinforcementLearningAgent:
    def __init__(self,discount_factor=0.9,hidden_size = 10, input_size=5, actions = [1, 2, 3],learning_rate=1e-3,
                 device='cpu',step_size=1000,gamma=0.98,momentum=0.9):
        output_size = len(actions)
        self.device = device
        
        self.model = NeuralNetwork(hidden_size , input_size, output_size).to(self.device)
        self.makeNewCopy()
        
        self.dp = DataPrepare(input_size,actions=actions)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate, momentum=momentum)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
        self.train_losses=[]
        
        self.learning_rate=learning_rate
        self.discount_factor = discount_factor

        
    def trainData(self,data=[],epochs=100,batch_size=10):
        self.dp.add(data)
    
        for _ in range(epochs):
            idxs = np.random.choice(len(self.dp),len(self.dp),False)
            sum_loss = 0
            
            for j in range(int(len(self.dp)/batch_size)):
                idx = idxs[j*batch_size:(j+1)*batch_size]
                X1 , X2 , action , reward = self.dp[idx]
                
                q_value = self.model.forward(X2.to(self.device))
                q_value=q_value.detach()
                a=action.long()
                b=torch.arange(batch_size).long()
                
                # reward function 
                # q_value[(b,a)]=(self.discount_factor*q_value[(b,a)]+reward.to(self.device)).float()
                # q_value=q_value.detach()
                
                self.model.train()
                self.optimizer.zero_grad()
                
                q_pred = self.model(X1.to(self.device))
                not_zero = torch.prod(X2==0,axis=1)==0
                not_zero = torch.arange(not_zero.shape[0])[not_zero]
                not_zero = not_zero.long()

                # reward function 
                q=q_pred.clone().detach()
                q[(b,a)]=reward.to(self.device).float()
                try:
                    q[(not_zero,a[not_zero])]+=(self.discount_factor*torch.max(q[not_zero],axis=1).values).float()
                except:
                    print('ERROR in nn')
                    pass
                
                loss = self.loss_fn(q,q_pred)
                loss.backward()
                self.optimizer.step()
        
                sum_loss += loss.item()
                            
                # self.scheduler.step()
            self.train_losses+=[sum_loss*batch_size/self.dp.X1.shape[0]]

        self.makeNewCopy()
        
    def makeNewCopy(self):
        self.copy_model = copy.deepcopy(self.model).to(self.device)
    
    def testData(self,data):
        self.copy_model.eval()
        data=data.to(self.device)
        return self.copy_model(data)
