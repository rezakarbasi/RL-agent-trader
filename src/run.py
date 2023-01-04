import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from threading import Thread
import threading
import torch

startingTime = int(time.time())

from .MT5Connection import MT5Connector
from .NeuralNetwork import RLAgent

#%%
def fibo(n):
    a=3
    b=5
    for i in range(n):
        c=a+b
        a=b
        b=c
    return b

class RunSocket(Thread):
    def __init__(self,rl):
        super().__init__()
        self.rl=rl
        sleep_interval=0
        self._kill = threading.Event()
        self._interval = sleep_interval

    def run(self):
        global connector
    
        counter=0
        action=0;

        while True:
            global epsilon

            prob=[0]
            try:
                # print('socket good place')
                for i in connector.Listen():
                    counter = 1
                    
                    global data
                    
                    ######################## TODO : check this each time
                    steps = i[0]
                    profit = i[1]
                    action = i[2]
                    state1 = i[3:6]
                    state2 = i[6:]
                    
                    profit = float(profit)
                    if profit<1 and profit>-1:
                        profit=0
                    elif profit>20:
                        profit=20
                    elif profit<-20:
                        profit=-20
                    
                    # if profit<0:
                    #     profit*=2
                    # elif profit>50.0:
                    #     profit=50.0
                    # if profit<-80:
                    #     profit=-80
                        
                    # profit-=float(steps)/20
                    
                    if i[0]!=0 and i[2]!=0:
                        data.append([state1,action,profit,steps,state2])

                    # best = torch.argmax(self.rl.testData(torch.tensor([*state2])))
                    qVal = (self.rl.testData(torch.tensor([*state2]))).cpu().detach().numpy()
                    prob = np.exp(qVal/epsilon)
                    prob /= np.sum(prob)
                    
                    o=[counter,np.random.choice(actions,p=prob)]
                    # if np.random.rand()>epsilon:
                    #     o=[counter,actions[best]]            
            
                    connector.Send(o)
                    
                    break
                
            except:
                print(prob)
                print(sys.exc_info()[0])
                counter+=1
                if counter>100 :
                    global th1
                    global th2
                    
                    th1.kill()
                    th2.kill()
                    
                # print('making socket again !')
                connector.End()
                connector = MT5Connector()

            is_killed = self._kill.wait(self._interval)
            if is_killed:
                connector.End()
                break

    def kill(self):
        self._kill.set()


class RunNetWork(Thread):
    def __init__(self, rl,sleep_interval=2):
        super().__init__()
        self.rl=rl
        self._kill = threading.Event()
        self._interval = sleep_interval

    def run(self):
        while True:
            global data
            # global rl

            if (len(data)<10):
                is_killed = self._kill.wait(self._interval)
                if is_killed:
                    break
                # print("sleep")
                continue

            global epsilon
            global epsDecay
            global epsThresh
            global epsCounter
            global epsStep
            epsCounter += 1
            if epsCounter>epsStep:
                # if epsilon<epsThresh:
                #     epsDecay=0.98
                #     rl.optimizer = torch.optim.SGD(rl.model.parameters(), lr=1e-8,momentum=0.9)
                # else :
                
                if epsilon<epsThresh:
                    epsDecay=0.9
                    
                rl.optimizer = torch.optim.SGD(rl.model.parameters(), lr=1e-7,momentum=0.9)
                rl.scheduler = torch.optim.lr_scheduler.StepLR(rl.optimizer, step_size=100000, gamma=0.93)
                epsilon*=epsDecay
                epsCounter=0

                print('new epsilon ' ,epsilon)
                torch.save(self.rl.model, 'nn model')
            
                plt.plot(self.rl.trainLosses)
                plt.yscale('log')
                plt.savefig(str(startingTime)+'.png')
                plt.show()
                plt.close()

            # print('start learning')
            print(str(len(data))+' -  epsCounter' + str(epsCounter))
            
            d=[i for i in data]
            data=[]
            
            self.rl.trainData(d,epochs,batchSize)
            self.rl.dp.selection(30000)
            
            if self.rl.trainLosses[-1]==np.nan:
                print('nan')
                    
            is_killed = self._kill.wait(self._interval)
            if is_killed:
                break

    def kill(self):
        # global rl
        torch.save(self.rl.model, 'nn model')
        self._kill.set()

# 2 inputs + distance to stoploss input
# 3 outputs for each Q value
actions = np.array([16,36,43])
rl = RLAgent(discount_factor=0.99,hidden_size = 20, input_size=3, output_size = 3,
             learningRate=1e-5,device='cuda',stepSize=1000,gamma=0.93)

aa={}
# for j,i in enumerate(actions):
#     aa[str(float(fibo(i)))]=j
for j,i in enumerate(actions):
    aa[str(i*1.0)]=j

rl.dp.actionsDict = aa
    
epochs = 3
batchSize = 100

epsilon = 5
epsDecay = 0.8
epsThresh = 0.5
epsCounter = 0
epsStep = 10
# epsilon = 1
# epsDecay = 0.95
# epsThresh = 0.1
# epsCounter = 0
# epsStep = 30

data =[]
d=[]
connector = MT5Connector()

# fileName = (input("enter file name : \n"))

th1 = RunSocket(rl)
th2 = RunNetWork(rl,2)

th1.start()
th2.start()

a=''
while a.lower()!="end":
    a=input('press end and Enter to exit the program !\n')

th1.kill()
th2.kill()
