import socket

server_host = '192.168.1.24'  # Listen on all available interfaces
server_port = 5656


def send_image(client_socket, image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    image_size = len(image_data)
    client_socket.send(image_size.to_bytes(4, byteorder='big'))
    client_socket.send(image_data)


def fs_main(server_host, server_port, img_arr):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_host, server_port))
    server_socket.listen(1)

    print(f"Server is listening on {server_host}:{server_port}")

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address}")

        for image_index in range(len(img_arr)):
            # Replace with actual image path
            image_path = f'image{image_index}.png'
            send_image(client_socket, image_path)

        client_socket.close()
        print("Images sent successfully")


if __name__ == "__main__":
    main()
