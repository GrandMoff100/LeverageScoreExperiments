import torch

# Pick the cuda device with the least memory usage
def pick_most_free_cuda_device():
    if not torch.cuda.is_available():
        return None

    num_devices = torch.cuda.device_count()
    max_free_memory = float('-inf')
    best_device = None

    for device_id in range(num_devices):
        props = torch.cuda.get_device_properties(device_id)
        total_memory = props.total_memory
        reserved_memory = torch.cuda.memory_reserved(device_id)
        free_memory = total_memory - reserved_memory
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_device = device_id

    return best_device


if __name__ == "__main__":
    for device in range(torch.cuda.device_count()):
        print(f"Device {device}: {torch.cuda.get_device_name(device)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 2:.2f} MiB")
        print(f"  Allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MiB")
        print(f"  Reserved memory: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MiB")
        print("")
    best_device = pick_most_free_cuda_device()
    print(f"Device with most free memory: cuda:{best_device}")