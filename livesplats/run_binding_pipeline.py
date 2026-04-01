import os
import subprocess
import glob
import sys
import argparse
import pynvml
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# --- Constants ---
TRAIN_SCRIPT = "scripts/train_dynamic_bind.py"
CPP_SOURCE = "scripts/pose_to_ply.cpp"
ASSOC_BINARY = "./pose_to_ply" 

def parse_args():
    parser = argparse.ArgumentParser(description="Run Training and Association Pipeline")
    
    parser.add_argument(
        "--base_root", 
        type=str, 
        required=True, 
        help="Path to the dataset root (e.g., data/players_A_pose/male)"
    )
    
    parser.add_argument(
        "--gpus", 
        nargs='+', 
        default=["all"],
        help="List of GPU IDs (e.g., '0 1 2') or 'all' to use all available GPUs."
    )
    
    return parser.parse_args()

def compile_cpp():
    """Compiles the C++ association tool."""
    print(f"--- Compiling {CPP_SOURCE} ---")
    
    if not os.path.exists(CPP_SOURCE):
        print(f"Error: Source file '{CPP_SOURCE}' not found.")
        sys.exit(1)

    cmd = ["g++", CPP_SOURCE, "-o", ASSOC_BINARY, "-O3"]
    
    try:
        subprocess.run(cmd, check=True)
        print("Compilation successful.\n")
    except subprocess.CalledProcessError:
        print("Error: Compilation failed. Please check your C++ code.")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'g++' not found. Ensure GCC is installed and in your PATH.")
        sys.exit(1)
        
def get_gpu_list(gpu_args):
    """
    Parses the --gpus argument.
    If 'all' is found, detects available GPUs via pynvml (no subprocess required).
    """
    if "all" in [str(g).lower() for g in gpu_args]:
        try:
            print("Detecting available GPUs...")
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()

            if count == 0:
                print("Error: pynvml reported 0 GPUs.")
                sys.exit(1)

            detected_gpus = list(range(count))
            print(f"-> Auto-detected {count} GPUs: {detected_gpus}")
            return detected_gpus

        except pynvml.NVMLError as e:
            print(f"Error: Could not detect GPUs via pynvml: {e}")
            print("Please specify IDs manually (e.g., --gpus 0 1)")
            sys.exit(1)
    else:
        # User provided specific IDs (e.g., "0" "1")
        try:
            return [int(g) for g in gpu_args]
        except ValueError:
            print(f"Error: Invalid GPU ID found in {gpu_args}")
            sys.exit(1)

def run_pipeline(task_args):
    """
    Worker function
    """
    # Unpack arguments
    clothing_path, skin_path, gpu_queue, config = task_args
    
    clothing_name = os.path.basename(clothing_path)
    skin_name = os.path.basename(skin_path)
    
    # Construct paths using the config dictionary
    model_out_dir = os.path.join(config["output_root"], clothing_name, skin_name)
    
    # Wait for a free GPU
    gpu_id = gpu_queue.get() 
    
    # Calculate CPU threads per job
    total_cores = os.cpu_count() or 4
    threads_per_job = max(1, total_cores // len(config["gpus"]))
    
    try:
        print(f"[Start Training] {clothing_name}/{skin_name} on GPU {gpu_id}")
        
        cmd_train = [
            "python", TRAIN_SCRIPT,
            "--source_path", config["base_root"],
            "--rgb_source_path", skin_path,
            "--model_path", model_out_dir,
            "--first_frame_obj", config["obj_path"],
            "--dataset_type", "basket_mv"
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Limit CPU threads to prevent system lag
        env["OMP_NUM_THREADS"] = str(threads_per_job)
        env["MKL_NUM_THREADS"] = str(threads_per_job)
        
        subprocess.run(cmd_train, env=env, check=True)
        print(f"[End Training]   {skin_name} finished.")

    except subprocess.CalledProcessError as e:
        print(f"!! TRAINING FAILED for {skin_name}: {e}")
        gpu_queue.put(gpu_id)
        gpu_queue.task_done()
        return

    # Free the GPU immediately so another job can start
    gpu_queue.put(gpu_id)
    gpu_queue.task_done()
    
    found_plys = glob.glob(os.path.join(model_out_dir, "**", "checkpoints", "0000.ply"), recursive=True)
    
    if not found_plys:
        print(f"!! SKIPPING ASSOC: No PLY found for {skin_name} in {model_out_dir}")
        return
        
    ply_file_path = found_plys[0]
    assoc_out_path = os.path.join(model_out_dir, "assoc.txt")

    print(f"[Start Assoc]    {skin_name} (CPU)")
    
    try:
        cmd_assoc = [
            ASSOC_BINARY,
            config["obj_path"],  # Input OBJ
            ply_file_path,       # Input PLY
            assoc_out_path       # Output TXT
        ]
        
        subprocess.run(cmd_assoc, check=True)
        print(f"[End Assoc]      {skin_name} -> Saved assoc.txt")
        
    except subprocess.CalledProcessError as e:
        print(f"!! ASSOC FAILED for {skin_name}: {e}")

def main():
    # Parse Arguments
    args = parse_args()
    
    active_gpus = get_gpu_list(args.gpus)
    
    # Setup Configuration
    base_root = args.base_root.rstrip("/")

    rgb_root = os.path.join(base_root, "rgb")
    obj_path = os.path.join(base_root, "objs", "0000.obj")
    
    output_root = os.path.join(base_root, "binding")

    config = {
        "base_root": base_root,
        "output_root": output_root,
        "obj_path": obj_path,
        "gpus": active_gpus
    }

    # Compile C++ Tool
    compile_cpp()
    
    # Setup GPU Queue
    gpu_queue = Queue()
    for gpu in active_gpus:
        gpu_queue.put(gpu)
    
    # Scan for Tasks
    tasks = []
    if not os.path.exists(rgb_root):
        print(f"Error: Data directory '{rgb_root}' does not exist.")
        return

    print(f"Scanning {rgb_root}...")
    clothing_dirs = glob.glob(os.path.join(rgb_root, "*"))
    
    for clothing_dir in clothing_dirs:
        if os.path.isdir(clothing_dir):
            skin_dirs = glob.glob(os.path.join(clothing_dir, "*"))
            for skin_dir in skin_dirs:
                if os.path.isdir(skin_dir):
                    # Pass config and queue to every task
                    tasks.append((clothing_dir, skin_dir, gpu_queue, config))

    print(f"Found {len(tasks)} tasks. Starting execution on {len(active_gpus)} GPUs: {active_gpus}\n")

    # Execute Pipeline
    with ThreadPoolExecutor(max_workers=len(active_gpus)) as executor:
        executor.map(run_pipeline, tasks)
        
    print("\nAll pipeline tasks completed.")

if __name__ == "__main__":
    main()