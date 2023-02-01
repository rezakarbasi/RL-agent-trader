import socket

HOST = "127.0.0.1"
PORT = 23456

while True:
    with socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            data = conn.recv(1024)
            print(data)
            conn.sendall(b"what's up in mql ?")

