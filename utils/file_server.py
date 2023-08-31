import socket


def start_server(host, port, openfile):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Server listening on {host}:{port}")

    client_socket, client_address = server_socket.accept()
    print(f"Connected to {client_address}")
    print(openfile)
    with open(f"{openfile}", "rb") as file:
        data = file.read(1024)
        while data:
            client_socket.send(data)
            data = file.read(1024)
    client_socket.close()
    server_socket.close()
    print("File sent successfully")


if __name__ == "__main__":
    server_host = "192.168.1.24"  # Replace with your server's IP address
    server_port = 5656        # Replace with desired port
    start_server(server_host, server_port)
