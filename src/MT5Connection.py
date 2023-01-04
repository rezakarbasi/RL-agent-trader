import socket

class MT5Connector:
    def __init__(self,ip='127.0.0.1',port=65432,bufferSize=1024):
        self.ip=ip
        self.port=port
        self.bufferSize=bufferSize
        
        self.server = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.server.bind((self.ip,self.port))
        self.server.settimeout(2)
        
        self.flag=True
        
    def Listen(self):

        while self.flag:
            message , addr = self.server.recvfrom(self.bufferSize)
            self.message = str(message)[2:-1]
            self.lastAddr = addr
            
#            print(message)
#            print(addr)
            
            yield self.InterpretMessage()
        
    def Send(self,data:list):
        send = str(data).replace(' ','')
        bytesToSend = str.encode(send)
        self.server.sendto(bytesToSend,self.lastAddr)
        
    def InterpretMessage(self):
        self.decode=[]
        a=self.message.replace('[','').replace(']','').split(',')

        if a[0].find('close')!=-1:
            self.flag=False
            return None
        
        self.decode=[float(i) for i in a]
        return self.decode
    
    def End(self):
        self.flag=False
        self.server.close()
