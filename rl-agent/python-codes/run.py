#%% imports
import threading
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from NeuralNetwork import RLAgent
from socket_handling import ThreadedTCPRequestHandler, ThreadedTCPServer, get_len_input_list, get_input_list, set_model, set_epsilon, set_actions

#%% hyperparameters
host = "0.0.0.0"
port = 4455

actions = [1,2,3]
epochs = 3
batchSize = 100

epsilon = 5
epsDecay = 0.8
epsThresh = 0.5
epsCounter = 0
epsStep = 10

max_memory_capacity = 30000

#%% main
server = ThreadedTCPServer((host, port), ThreadedTCPRequestHandler)
server.__enter__()
ip, port = server.server_address

server_thread = threading.Thread(target=server.serve_forever)
server_thread.daemon = True
server_thread.start()

set_actions(actions)

actions = np.array([1,2,3])
rl = RLAgent(discount_factor=0.99,hidden_size = 50, input_size=15, actions = actions,
             learningRate=1e-5,device='cpu',stepSize=1000,gamma=0.93)

rewards = []

while(True):
    time.sleep(0.1)
    if len(rl.trainLosses)>0:
        print(f"\r{get_len_input_list()} {epsCounter} {epsilon} {rl.trainLosses[-1]}", end="")
    else:
        print(f"\r{get_len_input_list()} {epsCounter} {epsilon} {np.nan}", end="")
    if(get_len_input_list() > 100):
        records = get_input_list()

        rewards.append(
            np.mean([record[2][0] for record in records])
        )

        rl.trainData(records,epochs,batchSize)
        rl.dp.selection(max_memory_capacity)
        
        if rl.trainLosses[-1]==np.nan:
            print('nan')

        epsCounter += 1
        if epsCounter>epsStep:
            
            if epsilon<epsThresh:
                epsDecay=0.9
                
            # rl.optimizer = torch.optim.RMSprop(rl.model.parameters(), lr=1e-3)
            # rl.scheduler = torch.optim.lr_scheduler.StepLR(rl.optimizer, step_size=100000, gamma=0.93)
            epsilon*=epsDecay
            epsCounter=0

            print('new epsilon ' ,epsilon)
            torch.save(rl.model, 'tmp/model.chkpt')
        
            plt.plot(rl.trainLosses)
            plt.yscale('log')
            plt.title("Losses")
            plt.xlabel("steps")
            plt.savefig("tmp/losses.png")
            plt.close()

            plt.plot(rewards)
            plt.title("Rewards")
            plt.xlabel("steps")
            plt.savefig("tmp/rewards.png")
            plt.close()
        
        set_epsilon(epsilon)
        set_model(rl)
