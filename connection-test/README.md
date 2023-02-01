# MQL - Python connection test

This is a simple test to check if the MQL4 and Python connection is working. I've got that from the [MQL5 forum](https://www.mql5.com/en/blogs/post/706665). Link claimed to work on MQL4 and MQL5, but **I only tested it on MQL5**.
To check the connection do the following: (connection port is 23456 and has been set in both code files)
- from MQL main window : `Tools -> Options -> Expert Advisors`:
  - check `Allow DLL imports`
  - add `localhost` and `127.0.0.1` to `Allow WebRequest for listed URL`
- copy `./mql-files/socket-library-mt4-mt5.mqh` to your MQL4/5 include folder and compile it
- copy `./mql-files/client.mq5` to your MQL5 experts folder and compile it
- run python file `./python-files/server.py`
- run the expert advisor `client.mq5`

Now metatrader sends a message to the python server and the python server sends a message back to metatrader. If everything works fine you should see the following output in the python console: (`PORT` is another port than 23456 and `NUMBER` increases every new packet)

    Connected by ('127.0.0.1', PORT)
    b'NUMBER'

Output in the metatrader console:

    what's up in mql ?
    what's up in mql ?
    .
    .
    .