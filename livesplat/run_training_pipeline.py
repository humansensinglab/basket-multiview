import os
import subprocess
import sys
import json
import argparse
import socket
import threading
import time
import pynvml
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# --- Config ---
IPC_PORT = 5005
MATCH_SCRIPT = "scripts/match_binding_to_person.py"
TRAIN_SCRIPT = "scripts/train_dynamic_skeleton.py"
BG_TRAIN_SCRIPT = "scripts/train_static_opt.py"
OBJ_TRAIN_SCRIPT = "scripts/train_dynamic_ball.py"
MERGE_SCRIPT = "scripts/combine_scenes.py"
TASK_PER_GPU = 4
FRAME_WORKERS = 3
 
# State tracking
frame_tracker = defaultdict(set) # {frame_idx: {player1, player2}}
active_players = set()
active_objects = set()
pipeline_lock = threading.Lock()
stop_event = threading.Event()

def get_gpu_list(gpu_args):
    if "all" in [str(g).lower() for g in gpu_args]:
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            return list(range(count))
        except pynvml.NVMLError:
            return [0]
    return [int(g) for g in gpu_args]

# -------------------------------------------------------------------
# UDP SERVER LOOP
# -------------------------------------------------------------------
def ipc_server_loop(output_root, total_objects_count, scene_path, dataset_type, add_bg=False, gpu=0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', IPC_PORT))
    sock.settimeout(2.0)
    print(f"[Server] Listening for completion signals on port {IPC_PORT}...")

    final_output_dir = os.path.join(output_root, "combined")
    os.makedirs(final_output_dir, exist_ok=True)
    if add_bg: total_objects_count += 1

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(1024)
            msg = json.loads(data.decode('utf-8'))
            
            p_id = msg['object_key']
            frame = msg['frame_idx']
            
            with pipeline_lock:
                frame_tracker[frame].add(p_id)
                current_done = len(frame_tracker[frame])
                
                # Check if tasks is done for this frame
                if current_done >= total_objects_count:
                    print(f"[Merger] All {current_done} players finished Frame {frame}. Merging...")
                    
                    # Collect PLY paths
                    input_plys = []
                    for p in active_players:
                        # Construct path based on training script output logic
                        ply_path = os.path.join(output_root, "dynamic", p, f"{frame}", "checkpoints", f"{frame:04d}.ply")
                        input_plys.append(ply_path)
                    for o in active_objects:
                        # Construct path based on training script output logic
                        ply_path = os.path.join(output_root, "dynamic", o, f"{frame}", "checkpoints", f"{frame:04d}.ply")
                        input_plys.append(ply_path)
                    
                    if add_bg:
                        bg_ply_path = os.path.join(output_root, "static", f"{frame}", "checkpoints", f"{frame:04d}.ply")
                        input_plys.append(bg_ply_path)
                        
                    output_ply = os.path.join(final_output_dir, f"{frame}", f"{frame:04d}.ply")
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    # Trigger Merge Script
                    subprocess.run([
                        "python", MERGE_SCRIPT,
                        "--output_path", output_ply,
                        "--input_plys", *input_plys,
                        "--render", 
                        "--source_path", scene_path, 
                        "--dataset_type", dataset_type,
                    ], env=env, check=True)
                    
                    # Cleanup memory for this frame
                    del frame_tracker[frame]
                    
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[Server Error] {e}")

# -------------------------------------------------------------------
# TASKS
# -------------------------------------------------------------------
def task_match(args):
    pid, scene, bind_root, output_root, dtype, q, results = args
    gpu = q.get()
    try:
        out_json = f"{output_root}/dynamic/{pid}/binding_match.json"
        cmd = ["python", MATCH_SCRIPT, "--source_path", scene, "--binding_root", bind_root, 
               "--object_key", pid, "--dataset_type", dtype, "--output_json", out_json]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        subprocess.run(cmd, env=env, check=True)
        with open(out_json) as f: results[pid] = json.load(f)['best_match']
    except Exception as e: print(f"Match failed {pid}: {e}")
    finally: q.put(gpu); q.task_done()

def task_train(args, script=TRAIN_SCRIPT, type="dynamic"):
    pid, ply, scene, out_root, dtype, q, threads, opt, worker_id, frame_workers = args
    gpu = q.get()
    model_path = os.path.join(out_root, type)
    try:
        frame_tag = f"worker {worker_id}/{frame_workers}" if frame_workers > 1 else "all frames"
        print(f"[Train] Launching {pid} ({frame_tag}) on GPU {gpu}")
        cmd = [
            "python", script,
            "--source_path", scene, "--model_path", model_path,
            "--object_key", pid, "--first_frame_ply", ply,
            "--dataset_type", dtype,
            "--ipc_port", str(IPC_PORT)
        ]
        if frame_workers > 1:
            cmd += ["--frame_worker_id", str(worker_id), "--frame_workers", str(frame_workers)]
        if opt.noisy_skeleton and type == "dynamic": cmd.append("--noisy_skeleton")
        if opt.use_static_graph and type == "static": cmd += ["--use_precomputed"]
        if opt.eval: cmd.append("--eval")
        if opt.render: cmd.append("--render")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        subprocess.run(cmd, env=env, check=True)
    except Exception as e: print(f"Train failed {pid}: {e}")
    finally: q.put(gpu); q.task_done()




# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--binding_root", required=True)
    parser.add_argument("--asset_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--first_frame_bg", default=None)
    parser.add_argument("--noisy_skeleton", action='store_true')
    parser.add_argument("--use_static_graph", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--gpus", nargs='+', default=["all"])
    args = parser.parse_args()

    gpus = get_gpu_list(args.gpus)
    frame_workers = FRAME_WORKERS 
    print(f"Frame workers per track: {frame_workers}")

    q = Queue()
    for _ in range(TASK_PER_GPU):
        for g in gpus: q.put(g)

    # Calculate CPU threads allocation
    total_cores = os.cpu_count() or 4
    threads_per_job = max(1, total_cores // len(gpus))
    print(f"Detected {total_cores} CPU cores. Allocating {threads_per_job} threads per job.")

    # 1. Discovery
    skel_root = os.path.join(args.scene_path, "skeleton")
    players = [os.path.basename(p) for p in os.listdir(skel_root) if os.path.isdir(os.path.join(skel_root, p))]
    active_players.update(players)
    
    obs_root = os.path.join(args.scene_path, "objs")
    objs = [os.path.basename(p) for p in os.listdir(obs_root) if os.path.isdir(os.path.join(obs_root, p))]
    active_objects.update(objs)
    first_frame_obj_ply = "binding/0/checkpoints/0000.ply"
    print(f"Discovered Players: {players}")
    print(f"Discovered Objects: {objs}")
    
    # 2. Matching
    print("--- Phase 1: Parallel Matching ---")
    matches = {}
    with ThreadPoolExecutor(len(gpus) * TASK_PER_GPU) as exe:
        futures = [exe.submit(task_match, (p, args.scene_path, args.binding_root, args.output_root, "basket_mv", q, matches)) for p in players]
        for f in as_completed(futures): f.result()

    valid_players = [p for p in players if p in matches]
    active_players.intersection_update(valid_players)
    print(f"Valid Players: {valid_players}")

    # 3. Start Listener
    # Each player/object sends one IPC signal per frame regardless of how many
    # frame-chunk workers cover that frame, so the per-frame total count is unchanged.
    server_thread = threading.Thread(target=ipc_server_loop, args=(args.output_root, len(valid_players) + len(objs), args.scene_path, "basket_mv", args.first_frame_bg is not None, gpus[0]), daemon=True)
    server_thread.start()

    # 4. Training — parallelised over both players/objects AND frame chunks
    print(f"--- Phase 2: Parallel Training ({frame_workers} frame worker(s) per track) ---")
    with ThreadPoolExecutor(len(gpus) * TASK_PER_GPU) as exe:
        futures = [
            exe.submit(task_train,
                       (p, matches[p]['ply_path'], args.scene_path, args.output_root,
                        "basket_mv", q, threads_per_job, args, wid, frame_workers))
            for p in valid_players
            for wid in range(frame_workers)
        ]
        futures += [
            exe.submit(task_train,
                       (o, os.path.join(args.asset_root, o, first_frame_obj_ply),
                        args.scene_path, args.output_root,
                        "basket_mv", q, threads_per_job, args, wid, frame_workers),
                       script=OBJ_TRAIN_SCRIPT)
            for o in objs
            for wid in range(frame_workers)
        ]
        if args.first_frame_bg is not None:
            futures += [
                exe.submit(task_train,
                           ("background", args.first_frame_bg, args.scene_path,
                            args.output_root, "basket_mv", q, threads_per_job,
                            args, wid, frame_workers),
                           script=BG_TRAIN_SCRIPT, type="static")
                for wid in range(frame_workers)
            ]
        for f in as_completed(futures): f.result()

    print("Training complete. Stopping server...")
    stop_event.set()
    server_thread.join()

if __name__ == "__main__":
    main()