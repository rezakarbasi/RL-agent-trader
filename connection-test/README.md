# MQL - Python Connection Test

This is a simple test to check the connection between MQL5 and Python. The instructions and code have been taken from the [MQL5 forum](https://www.mql5.com/en/blogs/post/706665) with the note that it is claimed to work with both MQL4 and MQL5, but the author has only tested it on MQL5.

To set up the connection, follow these steps:

1. In the MQL main window, go to `Tools -> Options -> Expert Advisors`:
   - Check `Allow DLL imports`
   - Add `localhost` and `127.0.0.1` to `Allow WebRequest for listed URL`
2. Copy `./mql-files/socket-library-mt4-mt5.mqh` to your MQL4/5 include folder and compile it
3. Copy `./mql-files/client.mq5` to your MQL5 experts folder and compile it
4. Run the Python file `./python-files/server.py`
5. Run the expert advisor `client.mq5`

The connection port has been set to 23456 in both code files. If everything is working correctly, you should see the following output in the Python console:

Connected by ('127.0.0.1', PORT)
b'NUMBER'


And the following output in the Metatrader console:

```
  what's up in mql ?
  what's up in mql ?
  .
  .
  .