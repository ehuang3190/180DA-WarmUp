import socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('131.179.36.11', 4444)) #change based on server ip
client.sendall('I am  CLIENT\n'.encode())
from_server = client.recv(4096)
client.close()
print(from_server)

