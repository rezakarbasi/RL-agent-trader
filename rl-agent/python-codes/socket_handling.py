import threading
import socketserver
import random
import time
import copy


def decode_state(state):
    state = state.split(",")
    state = [float(i) for i in state]
    return state

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    input_list = []
    model = None

    def handle(self):
        data = self.request.recv(1024)
        data = str(data)[2:-1].split("--")
        encoded = [decode_state(i) for i in data]
        ThreadedTCPRequestHandler.input_list.append(encoded)
        response = random.sample([1,2,3],1)[0] \
            if isinstance(ThreadedTCPRequestHandler.model, type(None)) \
            else ThreadedTCPRequestHandler.model.choose(data)
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

if __name__ == "__main__":
    host = "0.0.0.0"
    port = 4455

    # Port 0 means to select an arbitrary unused port
    HOST, PORT = host, port

    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    with server:
        ip, port = server.server_address

        # Start a thread with the server -- that thread will then start one
        # more thread for each request
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
    
        while(True):
            time.sleep(1)
            print(f"\r{get_len_input_list()}    ",end="")
            if(get_len_input_list() > 1000):
                get_input_list()

        server.shutdown()