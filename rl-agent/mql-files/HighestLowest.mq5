//+------------------------------------------------------------------+
//|                                                HighestLowest.mq5 |
//|                                                     Reza Karbasi |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Reza Karbasi"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property indicator_chart_window
//#property indicator_separate_window
#property indicator_buffers 3
#property indicator_plots   3
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+

#define ARRAY_SIZE         100

//--- plot lowest
#property indicator_label1  "highest"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1
//--- plot highest
#property indicator_label2  "lowest"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrBlue
#property indicator_style2  STYLE_SOLID
#property indicator_width2  1
//--- plot velocity
//#property indicator_label3  "velocity"
//#property indicator_type3   DRAW_LINE
//#property indicator_color3  clrDarkKhaki
//#property indicator_style3  STYLE_SOLID
//#property indicator_width3  1

input    int      period   =  360;

double highest[];
double lowest[];
double velocity[];

int indexesH[ARRAY_SIZE];
double valuesH[ARRAY_SIZE];

int indexesL[ARRAY_SIZE];
double valuesL[ARRAY_SIZE];

int fillH;
int fillL;

int   flag;

int AddToHighArray(double &values[],int &indexes[],int filled,int newIndex,const double &source[]);
int AddToLowArray(double &values[],int &indexes[],int filled,int newIndex,const double &source[]);

int OnInit()
  {
//--- indicator buffers mapping
      SetIndexBuffer(0,highest,INDICATOR_DATA);
      SetIndexBuffer(1,lowest,INDICATOR_DATA);

      SetIndexBuffer(2,velocity,INDICATOR_DATA);
      
      fillH = 0;
      fillL = 0;
      
      flag = 1;
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
//---

   if(prev_calculated==rates_total)
      return(rates_total);

   if(rates_total<period)
      return(rates_total);
      
   int startIdx = prev_calculated;
   if(flag)
   {
      
      indexesH[fillH]=prev_calculated;
      valuesH[fillH]=high[prev_calculated];
      fillH++;
      
      flag=0;
      for(int i=prev_calculated+1; i<period; i++)
      {
         fillH=AddToHighArray(valuesH,indexesH,fillH,i,high);
         fillL=AddToLowArray(valuesL,indexesL,fillL,i,low);
         
         highest[i] = valuesH[0];
         lowest[i] = valuesL[0];
         
      velocity[i]=(highest[i]-lowest[i])*2000/(MathAbs(indexesH[0]-indexesL[0])+1)/(highest[i]+lowest[i]+0.1);
      }
      startIdx = period;
   }
      
   for(int i=startIdx;i<rates_total;i++)
   {
      fillH=AddToHighArray(valuesH,indexesH,fillH,i,high);
      highest[i] = valuesH[0];

      fillL=AddToLowArray(valuesL,indexesL,fillL,i,low);
      lowest[i] = valuesL[0];
      
      velocity[i]=(highest[i]-lowest[i])*2000/(MathAbs(indexesH[0]-indexesL[0])+1)/(highest[i]+lowest[i]+0.1);
   }

//--- return value of prev_calculated for next call

   return(rates_total);
  }
//+------------------------------------------------------------------+

int AddToHighArray(double &values[],int &indexes[],int filled,int newIndex,const double &source[])
{
   newIndex--;
   double newValue = source[newIndex];

   if(indexes[0]<=(newIndex-period))
   {
      for(int i = 1 ; i<filled ;i++)
      {
         indexes[i-1]=indexes[i];
         values[i-1]=values[i];
      }
      filled-=1;

      int maxIdx=indexes[filled-1]+1;
      double maxV=source[maxIdx];
      for(int i=indexes[filled-1]+2;i<newIndex+1;i++)
      {
         if(maxV<source[i])
         {
            maxV=source[i];
            maxIdx=i;
         }
      }
      newValue = maxV;
      newIndex = maxIdx;
   }
   
   int idx = filled-1; 
   while(idx>=0)
   {
      if(values[idx]>=newValue)break;
      idx--;
   }
   idx++;
   if(idx<period && idx<ARRAY_SIZE)
   {
      values[idx] = newValue;
      indexes[idx] = newIndex;
      filled = idx+1;
   }
   
   return filled;
}

int AddToLowArray(double &values[],int &indexes[],int filled,int newIndex,const double &source[])
{
   newIndex--;
   double newValue = source[newIndex];

   if(indexes[0]<=(newIndex-period))
   {
      for(int i = 1 ; i<filled ;i++)
      {
         indexes[i-1]=indexes[i];
         values[i-1]=values[i];
      }
      filled-=1;

      int minIdx=indexes[filled-1]+1;
      double minV=source[minIdx];
      for(int i=indexes[filled-1]+2;i<newIndex+1;i++)
      {
         if(minV>source[i])
         {
            minV=source[i];
            minIdx=i;
         }
      }
      newValue = minV;
      newIndex = minIdx;
   }
   
   int idx = filled-1; 
   while(idx>=0)
   {
      if(values[idx]<=newValue)break;
      idx--;
   }
   idx++;
   if(idx<period && idx<ARRAY_SIZE)
   {
      values[idx] = newValue;
      indexes[idx] = newIndex;
      filled = idx+1;
   }
   
   return filled;
}
