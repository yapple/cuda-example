
# 编译进程A（中心KVCache）
nvcc kvcache_server_a.cpp -o kvcache_server_a -lcudart

# 编译进程B（显卡1）
nvcc kvcache_client_b.cpp -o kvcache_client_b -lcudart

# 编译进程C（显卡2）
nvcc kvcache_client_c.cpp -o kvcache_client_c -lcudart
