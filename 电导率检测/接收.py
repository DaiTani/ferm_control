import socket

UDP_IP = "0.0.0.0"  # 监听所有网络接口
UDP_PORT = 1222

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"监听 UDP 端口 {UDP_PORT}...")

try:
    while True:
        data, addr = sock.recvfrom(1024)
        print(f"收到来自 {addr} 的消息: {data.decode()}")
except KeyboardInterrupt:
    print("\n服务器已停止")
finally:
    sock.close()
