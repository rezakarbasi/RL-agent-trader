import numpy as np
import torch
import socketserver
import copy


def decode_state(state):
    state = state.split(",")
    state = [float(i) for i in state]
    return state

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    input_list = []
    model = None
    epsilon = 100
    actions = [1,2,3]

    def handle(self):
        data = self.request.recv(1024)
        # last-state, new-state, reward, action        
        data = str(data)[2:-1]
        data = "0" + data if data[0]=="-" else data
        data = data.split("--")
        encoded = [decode_state(i) for i in data]
        if encoded[0][0] == 0:
            encoded[0] = [0 for i in encoded[1]]
        ThreadedTCPRequestHandler.input_list.append(encoded)

        response = np.random.choice(ThreadedTCPRequestHandler.actions)
        if not isinstance(ThreadedTCPRequestHandler.model, type(None)):
            qVal = (ThreadedTCPRequestHandler.model.testData(torch.tensor([*encoded[1]]))).cpu().detach().numpy()
            prob = np.exp(qVal/ThreadedTCPRequestHandler.epsilon)
            prob /= np.sum(prob)
            response = np.random.choice(ThreadedTCPRequestHandler.actions,p=prob)
            
        response = bytes(str(response),'ascii')
        self.request.sendall(response)

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

def get_len_input_list():
    return len(ThreadedTCPRequestHandler.input_list)

def get_input_list():
    out = copy.deepcopy(ThreadedTCPRequestHandler.input_list)
    ThreadedTCPRequestHandler.input_list = []
    return out

def set_model(model):
    ThreadedTCPRequestHandler.model = model

def set_epsilon(epsilon):
    ThreadedTCPRequestHandler.epsilon = epsilon

def set_actions(actions):
    ThreadedTCPRequestHandler.actions = actions
