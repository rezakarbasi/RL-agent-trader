#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#property strict
#include <socket-library-mt4-mt5.mqh>
#include<Trade\Trade.mqh>

//input parameters
input string   Hostname    = "127.0.0.1";
input ushort   ServerPort  = 2323;

input double   lot         = 0.1;

// my local variables
double ask;
double bid;

const ENUM_POSITION_TYPE BUY = POSITION_TYPE_BUY;
const ENUM_POSITION_TYPE SELL = POSITION_TYPE_SELL;

ENUM_TIMEFRAMES my_period;
int last_bars = 0;

const int len_indicators = 6;
int indicator_periods[6]={5,10,20,40,80,120};
int indicator_ids[6];

double highests[6];
double lowests[6];
double middles[6];

int socket_counter = 0;
int socket_choice = 1;
string socket_last_state="";

CPositionInfo  position_info;

int trade_ticket = 0;
double trade_sl = 0;
double trade_entrance = 0;
ENUM_POSITION_TYPE trade_type = BUY;
double trade_reward = 0.0;
CTrade  trade_object;

bool first_step = true;

// defined functions
string SocketSendGet(string toPrint);
int ChangeStopLoss(int position_ticket,double new_sl);
int OpenContractNow(ENUM_POSITION_TYPE pos_type,double sl);

// defined OnInit function
int OnInit()
  {
   my_period = PERIOD_M15;
   last_bars = 0;
   bid = SymbolInfoDouble(_Symbol,SYMBOL_BID);
   ask = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   
   for (int i=0;i<len_indicators;i++)indicator_ids[i]=iCustom(_Symbol,_Period,"HighestLowest",indicator_periods[i]);

   return(INIT_SUCCEEDED);
}

// defined OnTick function
void OnTick()
{
   bid = SymbolInfoDouble(_Symbol,SYMBOL_BID);
   ask = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double price = (bid+ask)/2;
   
   int new_bars = Bars(_Symbol,my_period ) ;
   if(last_bars==new_bars)return ;
   last_bars = new_bars;
   
   
   MqlRates m[];
   CopyRates(_Symbol,my_period ,0,2,m);
   double close = m[0].close;
   double now = m[1].open;
   double high = m[0].high;
   double low = m[0].low;

   if (trade_reward != 0.0 || first_step);
   else if (trade_ticket != 0){
      ChangeStopLoss(trade_ticket, middles[socket_choice]);
   }
   else if (bid>highests[socket_choice]){
      Print("making buy trade");
      trade_ticket = OpenContractNow(POSITION_TYPE_BUY, middles[socket_choice]);
   }
   else if (ask<lowests[socket_choice]){
      Print("making sell trade");
      trade_ticket = OpenContractNow(POSITION_TYPE_SELL, middles[socket_choice]);
   }

   for (int i=0;i<len_indicators;i++){
      double a[];
      CopyBuffer(indicator_ids[i],0,0,1,a);
      highests[i]=a[0];
      CopyBuffer(indicator_ids[i],1,0,1,a);
      lowests[i]=a[0];
      middles[i]=(highests[i]+lowests[i])/2;
   }
      
   socket_counter++;
   if (socket_counter>10 || trade_reward!=0.0){
      socket_counter=0;
      string socket_new_state;
      
      if (trade_ticket!=0)socket_new_state = StringFormat("%f,%f,%f", trade_sl, trade_entrance, close);
      else socket_new_state = StringFormat("%f,%f,%f", close, close, close);
      
      for (int i=0;i<len_indicators;i++) StringAdd(socket_new_state, StringFormat(",%f,%f",lowests[i],highests[i]));      
      string sending_string = StringFormat("%s--%s--%f", socket_last_state, socket_new_state, trade_reward);
      string o = SocketSendGet(sending_string);
      int out = StringToInteger(o);
      if (out>0 && out<5)socket_choice = out;

      socket_last_state = "";
      StringAdd(socket_last_state, socket_new_state);
      trade_reward = 0.0;
      first_step = false;
   }   
}

void OnTrade()
{
   bool a = position_info.SelectByTicket(trade_ticket);
   //if(a==true && state2Cell.openContract==false)
   //{
   //   state2Cell.openContract=true;
   //   return;
   //}
   //else 
   if (a==true && trade_ticket==0){
      Print("ERROR IN ONTRADE ------------------------------------");
   }
   else if(a==false && trade_ticket!=0)
   {      
      bid = SymbolInfoDouble(_Symbol,SYMBOL_BID);
      ask = SymbolInfoDouble(_Symbol,SYMBOL_ASK);

      trade_ticket = 0;
      if(trade_type==BUY)
         trade_reward = (bid+ask)/2-trade_entrance;
      else if(trade_type==SELL)
         trade_reward = trade_entrance-(bid+ask)/2;
      return;
   }
}

// functions which I defined :
// socket transmission function
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
            return "0";
         }
      }
      return read;
   }
   return "0";
}

int OpenContractNow(ENUM_POSITION_TYPE pos_type,double sl)
{
   double price;
   
   if(pos_type==BUY)
   {
      price = ask;
      if(bid<sl) sl=bid-0.1;
      
      price=NormalizeDouble(price,_Digits);
      sl=NormalizeDouble(sl,_Digits);
      
      trade_sl = sl;
      
      if(!trade_object.Buy(lot,_Symbol,price,sl,0,"buy started !"))
      {
         Alert("---------------------------now trade problem-----------------------------");
         PrintFormat("make now error %d",GetLastError()); 
         return 0;
      }
   }
   else
   {
      price = bid;
      
      if(ask>sl)sl=ask+0.1;
      
      trade_sl = sl;

      price=NormalizeDouble(price,_Digits);
      sl=NormalizeDouble(sl,_Digits);
      
      if(!trade_object.Sell(lot,_Symbol,price,sl,0,"sell started !"))
      {
         Alert("---------------------------now trade problem-----------------------------");
         PrintFormat("make now error %d",GetLastError()); 
         return 0;
      }
   }
   trade_entrance = price;
   int positions = PositionsTotal();
   position_info.SelectByIndex(positions-1);
   return position_info.Ticket();
}

int ChangeStopLoss(int position_ticket,double new_sl)
{
   position_info.SelectByTicket(position_ticket);
   ENUM_POSITION_TYPE last_type = (ENUM_POSITION_TYPE)(position_info.PositionType());
   double last_sl = NormalizeDouble(position_info.StopLoss(),_Digits);
      
   if(last_type==BUY)
   {
      if(new_sl>bid)new_sl=bid-0.1;
      new_sl = NormalizeDouble(new_sl,_Digits);      
      if(new_sl<=last_sl)return 0;
   }
   else
   {
      if(new_sl<ask)new_sl=ask+0.1;
      new_sl = NormalizeDouble(new_sl,_Digits);
      if(new_sl>=last_sl)return 0;
   }
   
   if (position_info.Ticket() == 0){
      position_ticket = -1 ;
      return 0;
   }
   position_info.SelectByTicket(position_ticket);   
   if(!trade_object.PositionModify(position_ticket,new_sl,0))
   {
      Alert("---------------------------edit problem-----------------------------");
      PrintFormat("edit error %d",GetLastError()); 
      return 0;
   }
   trade_sl = new_sl;
   return 1;
}
