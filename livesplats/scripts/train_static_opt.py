import os
import sys
from argparse import ArgumentParser
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from losses import dynamic_loss as dl
from cs_renderers.accelerated_renderer import AccceleratedRenderer as acc
import datasets.dataset_factory as dataset_factory
from arguments import GroupParams, ModelParams, OptimizationParams, PipelineParams
from cs_renderers import cs_renderer_factory, get_cs_renderer_config
from utils.confs import DynamicTraningConf
from losses import get_loss_config, loss_factory
from utils.camera_utils import get_minicam
from scene_hs import GaussianModel
from utils.dynamic_utils import (
    get_timestamps, render_samples, create_output,
    prepare_output_and_logger, 
    send_data_to_device
) 
from utils.general_utils import safe_state
from torchvision.utils import save_image
import torchvision.utils as vutils
from utils.ipc_notifier import notify_frame_done

# Function to handle 1 frame of training
def training_frame(
    model_params: GroupParams,
    dynamic_training_params: DynamicTraningConf,
    pipeline_params: GroupParams,
    gaussian_model: GaussianModel,
    cur_dataset: dataset_factory.BaseDataset,
    frame_idx: int,
    num_total_frames: int,
    precomp_data,
    render_buffers: dict,
    use_cuda_graph: bool,
    grad_buffers: dict,
    minicam: list,
    stream,
    loss_object,
    renderer,
    background_in,
    record_graphs: bool,
    graphs: list,
    use_load_balancing: bool = False,
    scale_reg: float = 0,
    optimize_initial_model: bool = True,
    sfm_path: str = None,
    use_precomputed: bool = True,
):

    # Frame Initialization
    spatial_lr_scale = 0.01
    view_data = {}
    num_views = len(cur_dataset)
    num_views = cur_dataset.__len__()
    if model_params.eval:
        train_view_inds = [x for x in np.arange(num_views) if x % 10 != 0]
    else:
        train_view_inds = list(range(num_views))

    if not model_params.initialized:
        if sfm_path is not None:
            gaussian_model.load_sfm(sfm_path, spatial_lr_scale)

        else:
            # Create a random gaussian model if no previous model exists
            num_points = 100000
            center = cur_dataset.nerfpp_norm["translate"]
            length = cur_dataset.nerfpp_norm["radius"]
            gaussian_model.create_random(
                center=center,
                length=length,
                spatial_lr_scale=spatial_lr_scale,
                num_points=num_points,
            )

        model_params.initialized = True

    # Set up current optimization parameters
    cur_opt_params =  dynamic_training_params.optimization.get_opt_conf('static_opt')
    gaussian_model.training_setup(
        optimization_params=cur_opt_params,
        scheduler_params=dynamic_training_params.optimization.get_scheduler_conf('static'),
        densification_params=dynamic_training_params.densification,
    )

    # One iteration = optimizing one batch (no matter the batch size)
    num_iters = cur_opt_params.iterations
    iteration = 0

    # Batch size needs to be taken into account for num_iters
    progress_bar = tqdm(range(0, num_iters), desc="Training progress")
    
    path_to_output_folder = create_output(
        frame_idx, 
        model_params, 
        cur_dataset, 
        gaussian_model)

    # Dictionary to store Precomp-Graph parameters
    accel_params = {}
    # list that will store the graphs when created in record stage
    cuda_graphs = []
    
    frame_start = torch.cuda.Event(enable_timing=True)
    frame_end = torch.cuda.Event(enable_timing=True)
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # While loop to iterate through the frames
    frame_start.record()
    while iteration < num_iters:

        if frame_idx == 0 and not optimize_initial_model:
            break
        
        iter_start.record()
        view_idx = train_view_inds[iteration % len(train_view_inds)]
        gaussian_model.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussian_model.oneupSHdegree()
        
        cur_data = cur_dataset.__getitem__(view_idx)
        send_data_to_device(cur_data)

        if frame_idx == 0 or not use_precomputed:
            
            accel_params = {
                "store": False,
                "precomp": False,
                "ranges": torch.empty(0, dtype=torch.int8, device="cuda"),
                "gs_list": torch.empty(0, dtype=torch.int8, device="cuda"),
                "n_rend": 0,
                "n_buck": 0,
                "use_cuda_graph": False,
                "render_buffers": render_buffers,
                "minicam": minicam[view_idx],
                "background": background_in,
                "tiles_id": torch.empty(0, dtype=torch.int8, device="cuda"),
                "num_tiles": 0,
                "pixel_tr": torch.empty(0, dtype=torch.int8, device="cuda"),
                "render_output": grad_buffers["render_output"],
                "use_load_balancing": use_load_balancing,
            }

            rendered_out = renderer.render(
                gaussian_model=gaussian_model,
                pipeline_params=pipeline_params,
                accel_params=accel_params,
            )

            loss_dict = loss_object.compute_loss(
                gt_dict=cur_data,
                pred=rendered_out["render"],
                gaussian_model=gaussian_model,
                frameID=frame_idx,
                scale_reg=scale_reg,
            )

            loss_object.backward(loss_dict)

            # Optimizer step
            if iteration < num_iters:
                gaussian_model.optimizer.step()
                gaussian_model.optimizer.zero_grad(set_to_none=True)
                
            if pipeline_params.debug_static:
                if view_idx in pipeline_params.debug_cameras:
                    save_debug_image(
                        rendered_out["render"],
                        f"debug_precomp:{use_precomputed}_graph:{use_cuda_graph}_lb:{use_load_balancing}/frame{frame_idx:04d}_view{view_idx:02d}_it{iteration:05d}.png",
                    )

        else:

            grad_buffers["dL_ddc"][:] = 0
            grad_buffers["dL_dsh"][:] = 0
            grad_buffers["dL_dcolors"][:] = 0

            # If frame >0, use precomputed rasterizer. flag 'precomp' to true and pass preallocated buffers and tensors.
            accel_params = {
                "store": False,
                "precomp": True,
                "ranges": precomp_data[view_idx]["ranges"],
                "gs_list": precomp_data[view_idx]["gs_list"],
                "n_rend": precomp_data[view_idx]["num_rendered"],
                "n_buck": precomp_data[view_idx]["num_buckets"],
                "use_cuda_graph": use_cuda_graph,
                "render_buffers": render_buffers,
                "minicam": minicam[view_idx],
                "background": background_in,
                "dL_ddc": grad_buffers["dL_ddc"],
                "dL_dsh": grad_buffers["dL_dsh"],
                "dL_dcolors": grad_buffers["dL_dcolors"],
                "render_output": grad_buffers["render_output"],
                "tiles_id": precomp_data[view_idx]["tiles_id"],
                "num_tiles": precomp_data[view_idx]["num_tiles"],
                "pixel_tr": precomp_data[view_idx]["pixel_tr"],
                "use_load_balancing": use_load_balancing,
            }
            # Records in frame 1 only
            if record_graphs:

                if iteration < num_views and iteration < num_iters:

                    # -------------------------------------------------
                    # 1) WARMUP (normal run, NOT captured)
                    # -------------------------------------------------
                    with torch.no_grad():
                        grad_buffers["render_output"][:] = 0

                    _ = renderer.render(
                        gaussian_model=gaussian_model,
                        pipeline_params=pipeline_params,
                        accel_params=accel_params,
                    )

                    loss_dict = loss_object.compute_loss(
                        gt_dict=cur_data,
                        pred=grad_buffers["render_output"],
                        gaussian_model=gaussian_model,
                        frameID=frame_idx,
                    )

                    loss_object.backward(loss_dict)
                    gaussian_model.optimizer.zero_grad(set_to_none=False)

                    torch.cuda.synchronize()

                    # -------------------------------------------------
                    # 2) CAPTURE (same step, now recorded)
                    # -------------------------------------------------

                    g = torch.cuda.CUDAGraph()
                    cuda_graphs.append(g)

                    with torch.cuda.graph(g):

                        with torch.no_grad():
                            grad_buffers["render_output"][:] = 0

                        _ = renderer.render(
                            gaussian_model=gaussian_model,
                            pipeline_params=pipeline_params,
                            accel_params=accel_params,
                        )

                        loss_dict = loss_object.compute_loss(
                            gt_dict=cur_data,
                            pred=grad_buffers["render_output"],
                            gaussian_model=gaussian_model,
                            frameID=frame_idx,
                        )

                        loss_object.backward(loss_dict)

                        gaussian_model.optimizer.step()
                        gaussian_model.optimizer.zero_grad(set_to_none=False)

                    iteration += 1
                    continue

                else:
                    return cuda_graphs

            with torch.no_grad():
                grad_buffers["render_output"][:] = 0

            if not use_cuda_graph:

                _ = renderer.render(
                    gaussian_model=gaussian_model,
                    pipeline_params=pipeline_params,
                    accel_params=accel_params,
                )

                loss_dict = loss_object.compute_loss(
                    gt_dict=cur_data,
                    pred=grad_buffers["render_output"],
                    gaussian_model=gaussian_model,
                    frameID=frame_idx,
                )

                loss_object.backward(loss_dict)

                gaussian_model.optimizer.step()
                gaussian_model.optimizer.zero_grad(set_to_none=False)

                if pipeline_params.debug_static:
                    if view_idx in pipeline_params.debug_cameras:
                        save_debug_image(
                            grad_buffers["render_output"],
                            f"debug_precomp:{use_precomputed}_graph:{use_cuda_graph}_lb:{use_load_balancing}/frame{frame_idx:04d}_view{view_idx:02d}_it{iteration:05d}.png",
                        )

            else:
                # Graph->Replay graph for frames i; i>1
                grad_buffers["dL_ddc"][:] = 0
                grad_buffers["dL_dsh"][:] = 0
                grad_buffers["dL_dcolors"][:] = 0

                graphs[view_idx].replay()

                if pipeline_params.debug_static:
                    torch.cuda.synchronize()
                    if view_idx in pipeline_params.debug_cameras:
                        save_debug_image(
                            grad_buffers["render_output"],
                            f"debug_precomp:{use_precomputed}_graph:{use_cuda_graph}_lb:{use_load_balancing}/frame{frame_idx:04d}_view{view_idx:02d}_it{iteration:05d}.png",
                        )

        with torch.no_grad():

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Frame": f"{frame_idx+1}/{num_total_frames}",
                    }
                )
                progress_bar.update(10)
            if iteration == num_iters:
                progress_bar.close()

        iteration += 1

    
    try:
        frame_end.record()
        torch.cuda.synchronize()
        iter_time = frame_start.elapsed_time(frame_end)
        print(f"Total training time: {iter_time}ms")
    except Exception as e:
        print(f"Warning: Could not calculate final training time (likely GPU sync issue): {e}")
        
    print("Saving checkpoints...")
    checkpoint_path = os.path.join(path_to_output_folder, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    gaussian_model.save_ply(os.path.join(checkpoint_path, f"{frame_idx:04d}.ply"))
    
    # Notify the merger that this frame is done
    if hasattr(model_params, 'ipc_port') and model_params.ipc_port > 0:
        notify_frame_done(
            port=model_params.ipc_port,
            object_key=model_params.object_key,
            frame_idx=frame_idx
        )
        
    # Frame training ends. If frame==0, store data.
    if frame_idx == 0:
        buffer_sizes = {}

        accel_params = {
            "store": True,
            "precomp": False,
            "ranges": torch.empty(0, dtype=torch.int8, device="cuda"),
            "gs_list": torch.empty(0, dtype=torch.int8, device="cuda"),
            "n_rend": 0,
            "n_buck": 0,
            "use_cuda_graph": False,
            "render_buffers": render_buffers,
            "minicam": minicam[0],
            "background": background_in,
            "tiles_id": torch.empty(0, dtype=torch.int8, device="cuda"),
            "num_tiles": 0,
            "pixel_tr": torch.empty(0, dtype=torch.int8, device="cuda"),
            "render_output": grad_buffers["render_output"],
            "use_load_balancing": use_load_balancing,
        }

        for view_idx in range(0, num_views):

            accel_params["minicam"] = minicam[view_idx]

            rendered_out = renderer.render(
                gaussian_model=gaussian_model,
                pipeline_params=pipeline_params,
                accel_params=accel_params,
            )

            torch.cuda.synchronize()
            (
                _,
                ranges,
                gs_list,
                num_bucket,
                num_rendered,
                img_size,
                geom_size,
                sample_size,
                tiles_id,
                num_tiles,
                pixel_tr,
            ) = (
                rendered_out["render"],
                rendered_out["ranges"],
                rendered_out["gaussian_list"],
                rendered_out["num_bucket"],
                rendered_out["num_rendered"],
                rendered_out["img_buffer"],
                rendered_out["geom_buffer"],
                rendered_out["sample_buffer"],
                rendered_out["tiles_id"],
                rendered_out["num_tiles"],
                rendered_out["pixel_tr"],
            )

            view_data[view_idx] = {
                "ranges": ranges,
                "gs_list": gs_list,
                "num_buckets": num_bucket,
                "num_rendered": num_rendered,
                "tiles_id": tiles_id,
                "num_tiles": num_tiles,
                "pixel_tr": pixel_tr,
            }
            if view_idx == 0:
                buffer_sizes["img_buffer_size"] = img_size
                buffer_sizes["geom_buffer_size"] = geom_size
                max_sample_buffer_size = sample_size

            max_sample_buffer_size = max(max_sample_buffer_size, sample_size)
            if view_idx == num_views - 1:
                buffer_sizes["sample_buffer_size"] = max_sample_buffer_size

        return view_data, gaussian_model, buffer_sizes, True

def save_debug_image(img_chw: torch.Tensor, path: str):
    # img_chw: [C,H,W] float (0..1-ish)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with torch.no_grad():
        vutils.save_image(img_chw.clamp(0, 1), path)
        
def training(
    model_params,
    dynamic_training_params,
    loss_params,
    renderer_params,
    pipeline_params,
    sh_degree=3,
    using_graphs=True,
    using_loadbalancing=True,
    initial_gaussian_model=None,
    use_precomp_acceleration=True,
):

    timestamps = get_timestamps(model_params.source_path, subsample_step=1)

    if len(timestamps) == 0:
        raise ValueError(f"No timestamps found in {model_params.source_path}")
    if using_graphs and len(timestamps) < 2:
        using_graphs = False

    gaussian_model = GaussianModel(sh_degree)
    model_params.initialized = False

    prepare_output_and_logger(model_params)

    loss_object = loss_factory(type=loss_params["type"], conf=loss_params["conf"])
    renderer = cs_renderer_factory(
        type=renderer_params["type"], conf=renderer_params["conf"]
    )
    background = acc.get_background(renderer)

    cur_t = timestamps[0]
    conf = build_dataset_conf(model_params, cur_t)
    cur_dataset = dataset_factory.dataset_factory(
        type_dataset=model_params.dataset_type, conf=conf
    )
    num_views = len(cur_dataset)

    out_root = model_params.model_path
    state = init_runtime_state(num_views)

    init_views(state, cur_dataset)
    
    if initial_gaussian_model is not None:
        spatial_lr_scale = cur_dataset.nerfpp_norm["radius"]
        gaussian_model.load_ply(initial_gaussian_model, spatial_lr_scale)
        model_params.initialized = True
        state["gaussian_model"] = gaussian_model

    # FRAME_0
    print(f"Running Frame 0 to get Precompued Data...")

    run_frame0(
        model_params=model_params,
        dynamic_training_params=dynamic_training_params,
        pipeline_params=pipeline_params,
        loss_object=loss_object,
        renderer=renderer,
        background=background,
        state=state,
        cur_dataset=cur_dataset,
        timestamps=timestamps,
        initial_gaussian_model=initial_gaussian_model,
        gaussian_model=gaussian_model,
        use_load_balancing=using_loadbalancing,
    )

    print("Preallocating Memory for Tensors & Buffers...")
    # Preallocated Buffers. Will be used even with no graph
    ensure_graph_resources(state, cur_dataset, sh_degree)

    if using_graphs and use_precomp_acceleration:

        print(f"Recording Graphs...")
        record_graphs(
            model_params=model_params,
            dynamic_training_params=dynamic_training_params,
            pipeline_params=pipeline_params,
            loss_object=loss_object,
            renderer=renderer,
            background=background,
            state=state,
            cur_dataset=cur_dataset,
            timestamps=timestamps,
            use_load_balancing=using_loadbalancing,
            record_frame_idx=1,
            use_precomputed=use_precomp_acceleration,
        )
 
    # Apply frame worker interleaving to main training loop
    fw  = getattr(model_params, 'frame_workers',   1)
    wid = getattr(model_params, 'frame_worker_id', 0)
    worker_timestamps = list(enumerate(timestamps))
    if fw > 1:
        worker_timestamps = worker_timestamps[wid::fw]
        frame_ids = [i for i, _ in worker_timestamps]
        print(f"[Frame filter] Worker {wid}/{fw}: {len(worker_timestamps)} frames {frame_ids[:4]}{'...' if len(frame_ids) > 4 else ''}")

    for t_idx, frame in worker_timestamps:
        
        conf['t_idx'] = t_idx
        conf['timestamp'] = frame
        cur_dataset = dataset_factory.dataset_factory(model_params.dataset_type, conf.copy())
        
        if initial_gaussian_model is not None:
            spatial_lr_scale = cur_dataset.nerfpp_norm["radius"]
            gaussian_model.load_ply(initial_gaussian_model, spatial_lr_scale)
            model_params.initialized = True
            state["gaussian_model"] = gaussian_model

        print(f"Train frame: {frame}...")
        training_frame(
            model_params=model_params,
            dynamic_training_params=dynamic_training_params,
            pipeline_params=pipeline_params,
            gaussian_model=state["gaussian_model"],
            cur_dataset=cur_dataset,
            frame_idx=frame,
            num_total_frames=len(timestamps),
            precomp_data=state["precomp_data"],
            render_buffers=state["render_buffers"],
            use_cuda_graph=using_graphs,
            grad_buffers=state["grad_buffers"],
            minicam=state["minicam"],
            stream=state["stream"],
            loss_object=loss_object,
            renderer=renderer,
            background_in=background,
            record_graphs=False,
            graphs=state["graphs"] if using_graphs else None,
            use_load_balancing=using_loadbalancing,
            use_precomputed=use_precomp_acceleration,
        )


def build_dataset_conf(model_params, timestamp, img_scale_factor=0.5, spatial_scale_factor=1.0):
    conf = dict(dataset_factory.DATASET_CONF)  # important: copy
    conf.update(
        {
            "data_path": model_params.source_path,
            "timestamp": timestamp,
            "require_all_img_data_available": False,
            "first_k_cams": model_params.first_k_cams,
            "dataset_caching": model_params.dataset_caching,
            "spatial_scale_factor": spatial_scale_factor,
            "img_scale_factor": img_scale_factor,
            "object_key": model_params.object_key,
        }
    )
    return conf


def build_output_dirs(source_path):
    sequence = os.path.basename(os.path.dirname(os.path.normpath(source_path)))
    out_root = os.path.join("outputs", sequence)
    precomp_dir = os.path.join(out_root, "precomp")
    ckpt_dir = os.path.join(out_root, "checkpoints")
    os.makedirs(precomp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    return out_root, precomp_dir, ckpt_dir


def init_runtime_state(num_views):
    """
    Stateful containers that persist across frames.
    NOTE: Their identity should not change after init (especially for graphs).
    """
    return {
        "num_views": num_views,
        "cur_data": [
            {} for _ in range(num_views)
        ],  # view cache dicts (tensors allocated once inside update_view_cache)
        "minicam": [None] * num_views,  # per-view camera objects (typically static)
        "stream": torch.cuda.Stream(),
        # frame0 outputs:
        "precomp_data": None,
        "buffer_sizes": None,
        "gaussian_model": None,
        # graph-related:
        "render_buffers": None,
        "grad_buffers": None,
        "graphs": None,
        "graphs_recorded": False,
    }


def init_views(state, cur_dataset):
    for v in range(len(cur_dataset)):
        cam = cur_dataset[v]["cam_info"]
        state["minicam"][v] = get_minicam(
            width=cam["width"], height=cam["height"], K=cam["K"], w2c=cam["w2c"]
        )


def run_frame0(
    *,
    model_params,
    dynamic_training_params,
    pipeline_params,
    loss_object,
    renderer,
    background,
    state,
    cur_dataset,
    timestamps,
    initial_gaussian_model=None,
    gaussian_model=None,
    use_load_balancing=True,
):
    """
    Runs the special frame-0 initialization step (NOT graphable).
    Produces precomp_data + buffer_sizes, saves them, and saves point_cloud0.ply.

    Required state keys:
      - state["cur_data"]: list[dict] (view cache dicts)
      - state["minicam"]: list
      - state["stream"]: torch.cuda.Stream
      - state["loss_dict"]: dict
    """
    # Allocate minimal gradients dict needed by frame0 path
    W = cur_dataset[0]["cam_info"]["width"]
    H = cur_dataset[0]["cam_info"]["height"]
    grad_buffers_frame0 = {"render_output": torch.zeros((3, H, W), device="cuda")}

    # Decide whether to optimize initial model on frame 0
    if initial_gaussian_model is None:
        train_initial_frame = True
        if gaussian_model is None:
            gaussian_model = state.get("gaussian_model", None)
            if gaussian_model is None:
                raise ValueError(
                    "gaussian_model must be provided or stored in state['gaussian_model']"
                )
    else:
        train_initial_frame = False
        model_params.initialized = True
        gaussian_model.load_ply(initial_gaussian_model, 1)

    # Run frame-0 training (special path)
    precomp_data, gaussian_model, buffer_sizes, _ = training_frame(
        model_params=model_params,
        dynamic_training_params=dynamic_training_params,
        pipeline_params=pipeline_params,
        gaussian_model=gaussian_model,
        cur_dataset=cur_dataset,
        frame_idx=0,
        num_total_frames=len(timestamps),
        precomp_data=None,
        render_buffers=None,
        use_cuda_graph=False,
        grad_buffers=grad_buffers_frame0,
        minicam=state["minicam"],
        stream=state["stream"],
        loss_object=loss_object,
        renderer=renderer,
        background_in=background,
        record_graphs=False,
        graphs=None,
        use_load_balancing=use_load_balancing,
        optimize_initial_model=train_initial_frame,
    )

    # Stash results in state for later phases (graph record + loop)
    state["precomp_data"] = precomp_data
    state["buffer_sizes"] = buffer_sizes
    state["gaussian_model"] = gaussian_model

def ensure_graph_resources(state, cur_dataset, sh_degree):
    """
    Allocates:
      - state["render_buffers"] (renderer scratch buffers)
      - state["grad_buffers"] (preallocated tensors for gradients/output)
    once, based on state["buffer_sizes"], state["gaussian_model"], and image resolution.

    IMPORTANT INVARIANT:
      - For CUDA graph replay to be valid, shapes must not change:
        P (#gaussians), H/W, buffer_sizes.
      - If your training changes P (densify/prune), you must either disable that in graph mode
        or re-record graphs when P changes.
    """
    if state["buffer_sizes"] is None or state["gaussian_model"] is None:
        raise RuntimeError(
            "ensure_graph_resources requires state['buffer_sizes'] and state['gaussian_model'] set (run frame0 first)."
        )

    if state["render_buffers"] is None:
        bs = state["buffer_sizes"]
        state["render_buffers"] = {
            "img_buffer": torch.zeros(
                bs["img_buffer_size"], dtype=torch.uint8, device="cuda"
            ),
            "geom_buffer": torch.zeros(
                bs["geom_buffer_size"], dtype=torch.uint8, device="cuda"
            ),
            "sample_buffer": torch.zeros(
                bs["sample_buffer_size"], dtype=torch.uint8, device="cuda"
            ),
        }

    if state["grad_buffers"] is None:
        sh_rest = (sh_degree + 1) ** 2 - 1
        P = state["gaussian_model"].get_xyz.shape[0]
        W = cur_dataset[0]["cam_info"]["width"]
        H = cur_dataset[0]["cam_info"]["height"]
        state["grad_buffers"] = {
            "dL_ddc": torch.zeros((P, 1, 3), device="cuda"),
            "dL_dsh": torch.zeros((P, sh_rest, 3), device="cuda"),
            "dL_dcolors": torch.zeros((P, 3), device="cuda"),
            "render_output": torch.zeros((3, H, W), device="cuda"),
        }


def record_graphs(
    *,
    model_params,
    dynamic_training_params,
    pipeline_params,
    loss_object,
    renderer,
    background,
    state,
    cur_dataset,
    timestamps,
    use_load_balancing=True,
    record_frame_idx=1,
    use_precomputed=True,
):
    """
    Records CUDA graphs for the steady-state per-frame step.

    Why record_frame_idx defaults to 1:
      - Some code paths branch on (frame_idx == 0). You want the steady-state path.
      - You can still feed frame0 data (cur_data already holds it); frame_idx is just a flag.

    Stores state["graphs"] and sets state["graphs_recorded"] = True.
    """

    graphs = training_frame(
        model_params=model_params,
        dynamic_training_params=dynamic_training_params,
        pipeline_params=pipeline_params,
        gaussian_model=state["gaussian_model"],
        cur_dataset=cur_dataset,
        frame_idx=record_frame_idx,
        num_total_frames=len(timestamps),
        precomp_data=state["precomp_data"],
        render_buffers=state["render_buffers"],
        use_cuda_graph=True,
        grad_buffers=state["grad_buffers"],
        minicam=state["minicam"],
        stream=state["stream"],
        loss_object=loss_object,
        renderer=renderer,
        background_in=background,
        record_graphs=True,
        graphs=None,
        use_load_balancing=use_load_balancing,
        use_precomputed=use_precomputed,
    )

    state["graphs"] = graphs
    state["graphs_recorded"] = True


if __name__ == "__main__":

    parser = ArgumentParser(description="Training script parameters")

    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6031)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--websockets", default=True, action="store_true")
    parser.add_argument("--ipc_port", type=int, default=-1, help="Port to send frame completion signals to")
    parser.add_argument("--frame_worker_id", type=int, default=0, help="Index of this frame worker (0-based)")
    parser.add_argument("--frame_workers", type=int, default=1, help="Total number of parallel frame workers")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    model_params_processed = model_params.extract(args)
    model_params_processed.ipc_port = args.ipc_port
    model_params_processed.frame_worker_id = args.frame_worker_id
    model_params_processed.frame_workers = args.frame_workers
    dynamic_training_params = DynamicTraningConf()
    pipeline_params_processed = pipeline_params.extract(args)

    test_iters = args.test_iterations
    save_iters = args.save_iterations
    checkpoint_iterations = args.checkpoint_iterations
    starting_checkpoint = args.start_checkpoint
    debug_from = args.debug_from

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    # For static frame by frame
    loss_params = get_loss_config(type="static_opt")
    renderer_params = get_cs_renderer_config(type="accel-precomp")

    training(
        model_params=model_params_processed,
        dynamic_training_params=dynamic_training_params,
        loss_params=loss_params,
        renderer_params=renderer_params,
        pipeline_params=pipeline_params_processed,
            using_graphs=pipeline_params_processed.use_cuda_graph,
        using_loadbalancing=pipeline_params_processed.use_load_balancing,
        initial_gaussian_model=model_params_processed.first_frame_ply,
        use_precomp_acceleration=pipeline_params_processed.use_precomputed,
    )

    # All done
    print("\nTraining complete.")
