#property strict
#include <socket-library-mt4-mt5.mqh>

input string   Hostname = "localhost";
input ushort   ServerPort = 23456;
input string   FileToSend = "send.dat";

string SocketSendGet(string toPrint);

int counter = 0;
void OnTick()
{
   counter ++;
   string printed = IntegerToString(counter);
   string received = SocketSendGet(printed);
   Print(received);
}

string SocketSendGet(string toPrint){
   ClientSocket * socket = new ClientSocket(Hostname, ServerPort);
   if (socket.IsSocketConnected()) {
      socket.Send(toPrint);      
      string read;
      long timeout = 0;
      while (StringLen(read)==0){
         read = socket.Receive();
         timeout ++;
         if (timeout>30000){
            return SocketSendGet(toPrint);
         }
      }
      return read;
   }
   return "";
}