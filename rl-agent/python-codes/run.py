#%% imports
import threading
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from reinforcement_learning import ReinforcementLearningAgent
from socket_handling import ThreadedTCPRequestHandler, ThreadedTCPServer, get_len_input_list, get_input_list, set_model, set_epsilon, set_actions

#%% hyperparameters
host = "127.0.0.1"
port = 19968

actions = [1,2,3]
epochs = 3
batch_size = 100

epsilon = 1
epsilon_decay = 0.95
epsilon_threshold = 0.1
epsilon_counter = 0
epsilon_step = 10

max_memory_capacity = 30_000

#%% main
server = ThreadedTCPServer((host, port), ThreadedTCPRequestHandler)
server.__enter__()
ip, port = server.server_address

server_thread = threading.Thread(target=server.serve_forever)
server_thread.daemon = True
server_thread.start()

set_actions(actions)

actions = np.array([1,2,3])
rl = ReinforcementLearningAgent(discount_factor=0.9,hidden_size = 50, input_size=15, actions = actions,
             learning_rate=1e-4,device='cpu',step_size=1000,gamma=0.93)

rewards = []

while(True):
    time.sleep(0.1)
    if len(rl.train_losses)>0:
        print(f"\r{get_len_input_list()} {epsilon_counter} {epsilon} {rl.train_losses[-1]}", end="")
    else:
        print(f"\r{get_len_input_list()} {epsilon_counter} {epsilon} {np.nan}", end="")
    if(get_len_input_list() > 100):
        records = get_input_list()

        rewards.append(
            np.mean([record[2][0] for record in records])
        )

        rl.train_data(records,epochs,batch_size)
        rl.dp.selection(max_memory_capacity)
        
        if rl.train_losses[-1]==np.nan:
            print('nan')

        epsilon_counter += 1
        if epsilon_counter>epsilon_step:
            
            if epsilon<epsilon_threshold:
                epsilon_decay=epsilon_threshold
                
            epsilon*=epsilon_decay
            epsilon_counter=0

            print('new epsilon ' ,epsilon)
            torch.save(rl.model, 'tmp/model.chkpt')
        
            plt.plot(rl.train_losses)
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
