import os


def get_gpu_memory_usage() -> int:
    _NVIDIA_SMI_COMMAND = (
        "nvidia-smi "
        "--query-compute-apps=pid,used_memory "
        "--format=csv,noheader,nounits"
    )
    # If multiple GPU-processes are running, we get multiple results, splitted
    # with new-line char.
    results = os.popen(_NVIDIA_SMI_COMMAND).read().strip().split("\n")

    current_pid = os.getpid()
    for result in results:
        pid, used_memory = result.split(",")
        used_memory = int(used_memory)
        if int(pid) == current_pid: # This is our process!
            return used_memory

    print("Warning: Current process ID (pid) is not found. Check that you are "
          "really using GPU (transfered data and model to GPU). Additionally, "
          "if you are using this inside container, make sure that the "
          "container is using the host process namespace by providing "
          "'docker run ... --pid host ...'")

    return -1  # Indicate process not found.


def show_memory_gpu_usage() -> None:
    gpu_memory_usage = get_gpu_memory_usage()
    if gpu_memory_usage != -1:
        print(f"GPU memory usage: {gpu_memory_usage} MiB")
    else:
        print("Error: Couldn't find current process in nvidia-smi output.")
