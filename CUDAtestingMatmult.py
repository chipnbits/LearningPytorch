import torch
import time
import matplotlib.pyplot as plt

check_cuda = torch.cuda.is_available()
if check_cuda:
    print("Cuda is available")
else:
    print("Cuda is not available")

# Another type check is to use the following:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
    
# The matrix location can be changed to CPU or GPU by changing the device
# These can be placed on GPU using .cuda() or CPU using .cpu() appended to the end of the tensor
x_gpu = torch.rand(100,100).cuda()
x_cpu = torch.rand(100,100).cpu()
# Another way to do this more flexibly is to use the device variable we created above
x = torch.rand(100,100).to(device) # This will place the tensor on the GPU if available, otherwise it will place it on the CPU
# Additionally you can use the following to place a tensor on a specific GPU
x = torch.rand(100,100).to('cuda:0') # This will place the tensor on the first GPU if available, otherwise it will place it on the CPU


# Function to perform matrix multiplication on GPU and measure time
def gpu_multiply(x,y):
    # Send data to GPU
    x_gpu = x.to('cuda')
    y_gpu = y.to('cuda')
    start = time.time()
    torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()  # Wait for the multiply to finish
    return time.time() - start

# Function to perform matrix multiplication on CPU and measure time
def cpu_multiply(x,y):
    # Send data to CPU instead of GPU
    start = time.time()
    torch.matmul(x, y)
    return time.time() - start

# Sizes of matrices to test
sizes = [100, 500, 1000, 2000, 4000, 8000]
gpu_times = []
cpu_times = []

# Measure and record the times for both GPU and CPU
for n in sizes:

    x = torch.rand(n, n)
    y = torch.rand(n, n)
    gpu_time = gpu_multiply(x,y)
    gpu_times.append(gpu_time)

    cpu_time = cpu_multiply(x,y)
    cpu_times.append(cpu_time)

    print(f"Size: {n}x{n}, GPU Time: {gpu_time:.4f}s, CPU Time: {cpu_time:.4f}s")

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(sizes, gpu_times, marker='o', label='GPU')
plt.plot(sizes, cpu_times, marker='x', label='CPU')
plt.title('GPU vs. CPU Matrix Multiplication Performance')
plt.xlabel('Matrix size')
plt.ylabel('Computation time (seconds)')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Use logarithmic scale for better visibility on large ranges
plt.show()

# The initial GPU time is much slower because it is initializing CUDA and the GPU
