
# 编译发送端
nvcc ipc_sender.cu -o ipc_sender -lcudart

# 编译接收端
nvcc ipc_receiver.cu -o ipc_receiver -lcudart
