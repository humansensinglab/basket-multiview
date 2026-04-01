import os, torch
import sys
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import datasets.dataset_factory as dataset_factory
from arguments import (GroupParams, ModelParams, OptimizationParams,
                       PipelineParams)
from cs_renderers import cs_renderer_factory, get_cs_renderer_config
from utils.confs import DynamicTraningConf
from losses import get_loss_config, loss_factory
from scene_hs import GaussianModel
from utils.dynamic_utils import (create_output, get_timestamps,
                                 prepare_output_and_logger, render_samples,
                                 send_data_to_device, training_report,
                                 read_obj_file)

from utils.general_utils import safe_state
from collections import defaultdict
from utils.ipc_notifier import notify_frame_done 

def training_frame(
    model_params : GroupParams, 
    dynamic_training_params : DynamicTraningConf,
    loss_params:dict,
    renderer_params:dict,
    pipeline_params : GroupParams, 
    gaussian_model : GaussianModel, 
    cur_dataset: dataset_factory.BaseDataset,
    frame_idx : int,
    num_total_frames : int,
    test_iters : list,
    ):

    spatial_lr_scale = cur_dataset.nerfpp_norm["radius"]

    if not model_params.initialized: 
        if model_params.first_frame_ply:
            gaussian_model.load_ply(model_params.first_frame_ply, spatial_lr_scale)
        else:
            num_points = 100000
            center = cur_dataset.nerfpp_norm['translate']
            lenght = cur_dataset.nerfpp_norm['radius']
            gaussian_model.create_random(
                center=center, 
                lenght=lenght, 
                spatial_lr_scale=spatial_lr_scale,
                num_points=num_points)
            
        model_params.initialized = True
    
    cur_opt_params = dynamic_training_params.optimization.get_opt_conf('dynamic')
    gaussian_model.training_setup(
        optimization_params=cur_opt_params,
        scheduler_params=dynamic_training_params.optimization.get_scheduler_conf('dynamic'),
        densification_params=dynamic_training_params.densification)

    loss_object = loss_factory(type=loss_params['type'], conf=loss_params['conf'])
    renderer = cs_renderer_factory(type=renderer_params['type'], conf=renderer_params['conf'])        
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # One iteration = optimizing one batch (no matter the batch size)
    num_iters = cur_opt_params.iterations
    
    iteration = 1
    ema_dict_for_log = defaultdict(int)
    
    # Batch size needs to be taken into account for num_iters
    progress_bar = tqdm(range(0, num_iters), desc="Training progress")

    # Data split
    num_images = cur_dataset.__len__()
    if model_params.eval:
        train_view_inds = [x for x in np.arange(num_images) if x % 10 != 0]
    else:
        train_view_inds = list(range(num_images))
    
    path_to_output_folder = create_output(  
        frame_idx, 
        model_params, 
        cur_dataset, 
        gaussian_model)

    gaussian_model._xyz_reg = gaussian_model._xyz.clone().detach().requires_grad_(False)
    gaussian_model._scaling_reg = gaussian_model.get_scaling.clone().detach().requires_grad_(False)
    
    gaussian_model.training_setup(
        optimization_params=cur_opt_params,
        scheduler_params=dynamic_training_params.optimization.get_scheduler_conf('dynamic'),
        densification_params=dynamic_training_params.densification)    

    while iteration <= num_iters:
        iter_start.record()
        cur_img_idx = train_view_inds[iteration % len(train_view_inds)]
        
        gaussian_model.update_learning_rate(iteration)
        
        cur_data = cur_dataset.__getitem__(cur_img_idx)
        send_data_to_device(cur_data)
        bg_color=renderer.get_background()
        
        rendered_out = renderer.render(
            sample=cur_data, 
            gaussian_model=gaussian_model, 
            pipeline_params=pipeline_params,
            bg_color=bg_color)
        
        visibility_filter = rendered_out["visibility_filter"]
        
        loss_dict = {}
        if visibility_filter.sum() > 0.1 * gaussian_model._xyz.shape[0] and torch.sum(cur_data['alpha'] > 0) > 0: 
            loss_dict = loss_object.compute_loss(
                gt_dict=cur_data,
                pred_dict=rendered_out,
                gaussian_model=gaussian_model,
                masked=True,
                reg=True,
                bg_color=bg_color,
            )
            loss_object.backward(loss_dict)
        
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            for k in loss_dict:
                if k in ["total_loss", "psnr", "ssim"]:
                    ema_dict_for_log[k] = 0.4 * loss_dict[k] + 0.6 * ema_dict_for_log[k]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_dict_for_log['total_loss']:.{7}f}",
                                          "psnr": f"{ema_dict_for_log['psnr']:.{2}f}",
                                          "ssim": f"{ema_dict_for_log['ssim']:.{4}f}", 
                                          "num_points": f"{gaussian_model.get_xyz.shape[0]}",
                                          "Frame": f"{frame_idx+1}/{num_total_frames}"})
                progress_bar.update(10)
            if iteration == num_iters:
                progress_bar.close()
        
            training_report(
                gaussian_model, 
                iteration, 
                cur_dataset, 
                test_iters, 
                renderer.render, 
                pipeline_params)
        
            # Optimizer step
            if iteration < num_iters:
                gaussian_model.optimizer.step()
                gaussian_model.optimizer.zero_grad()
                
        iteration += 1
        
    # Save the current gaussian to the dynamic gausssian sequence
    print("Saving checkpoints...")
    checkpoint_path = os.path.join(path_to_output_folder, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    gaussian_model.save_ply(os.path.join(checkpoint_path, f"{frame_idx:04d}.ply"))
    
    print("Rendering samples...")
    if model_params.render:
        render_samples(
            gaussian_model, 
            cur_dataset, 
            renderer.render, 
            pipeline_params, 
            frame_idx, 
            path_to_output_folder
        )
        
    # Notify the merger that this frame is done
    if hasattr(model_params, 'ipc_port') and model_params.ipc_port > 0:
        notify_frame_done(
            port=model_params.ipc_port,
            object_key=model_params.object_key,
            frame_idx=frame_idx
        )
    
    for i in range(cur_dataset.__len__()):
        send_data_to_device(cur_dataset[i], 'cpu')

def training(
    model_params, 
    dynamic_training_params,
    loss_params,
    renderer_params,
    pipeline_params,
    starting_checkpoint=None,
    test_iters=None,
    ):

    # Read timestamps
    timestamps = get_timestamps(model_params.source_path, subsample_step=1)
    paired_timestamps = [(i, timestamp) for i, timestamp in enumerate(timestamps)]

    # Restrict to this worker's interleaved frames when running with frame-level parallelism.
    # Worker wid processes frames wid, wid+fw, wid+2*fw, ...
    fw  = getattr(model_params, 'frame_workers',   1)
    wid = getattr(model_params, 'frame_worker_id', 0)
    if fw > 1:
        paired_timestamps = paired_timestamps[wid::fw]
        frame_ids = [i for i, _ in paired_timestamps]
        print(f"[Frame filter] Worker {wid}/{fw}: {len(paired_timestamps)} frames {frame_ids[:4]}{'...' if len(frame_ids) > 4 else ''}")
    
    model_params.model_path = Path(model_params.model_path).resolve() / model_params.object_key
    dynamic_obj_paths = Path(model_params.source_path) / "objs" / model_params.object_key

    # Initialize Gaussians
    gaussian_model = GaussianModel(1)
    model_params.initialized = False
    starting_t = -1
    if starting_checkpoint:
        (model_params, starting_t) = torch.load(starting_checkpoint)
        starting_idx = -1
        for t_idx, timestamp in paired_timestamps:
            if timestamp < starting_t:
                starting_idx = t_idx
            else:
                break
        
        gaussian_model.restore(model_params, dynamic_training_params, starting_idx)
        model_params.initialized = True
        
    prepare_output_and_logger(model_params)    

    # Dataset parallel loading
    conf = dataset_factory.DATASET_CONF 
    conf['data_path'] = model_params.source_path
    conf['require_all_img_data_available'] = False
    conf['first_k_cams'] = model_params.first_k_cams
    conf['dataset_caching'] = model_params.dataset_caching
    conf['parallel'] = False
    conf['use_pickle'] = False
    conf['mesh_present'] = False
    conf['spatial_scale_factor'] = 1.0
    conf['img_scale_factor'] = 0.5
    conf['object_key'] = model_params.object_key if model_params.object_key else None
    
    if not model_params.first_frame_obj:
        model_params.first_frame_obj = str(model_params.first_frame_ply).replace("binding/0/checkpoints", "objs").replace(".ply", ".obj")
    
    for t_idx, timestamp in paired_timestamps:
        conf['t_idx'] = t_idx
        conf['timestamp'] = timestamp
        cur_dataset = dataset_factory.dataset_factory(model_params.dataset_type, conf.copy())
        
        if model_params.first_frame_ply:
            spatial_lr_scale = cur_dataset.nerfpp_norm["radius"]
            gaussian_model.load_ply(model_params.first_frame_ply, spatial_lr_scale, scale=0.01)
            binding_mean = torch.mean(gaussian_model._xyz, dim=0, keepdims=True).cuda()
            model_params.initialized = True
        
        # Load animated skeleton
        animated_mean = torch.mean(torch.tensor(read_obj_file(dynamic_obj_paths / f'ball_{t_idx}.obj')), dim=0, keepdims=True).cuda() * 0.01
        if model_params.axis_flip:
            print("Applying axis flip to animated skeleton")
            animated_mean[:, 0] = -animated_mean[:, 0]
            animated_mean[:, 1] = -animated_mean[:, 1]

        with torch.no_grad():
            gaussian_model._xyz += (animated_mean - binding_mean)
        
        training_frame(
            model_params=model_params,
            dynamic_training_params=dynamic_training_params,
            loss_params=loss_params,
            renderer_params=renderer_params,
            pipeline_params=pipeline_params,
            gaussian_model=gaussian_model,
            cur_dataset=cur_dataset,
            frame_idx=t_idx,
            num_total_frames=len(timestamps),
            test_iters=test_iters,
        )
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=7009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", default=True, action="store_true")
    parser.add_argument("--ipc_port", type=int, default=-1, help="Port to send frame completion signals to")
    parser.add_argument("--render", action="store_true", default=False, help="save renders after training")
    parser.add_argument("--axis_flip", action="store_true", default=False, help="flip the axis when load obj")
    parser.add_argument("--frame_worker_id", type=int, default=0, help="Index of this frame worker (0-based)")
    parser.add_argument("--frame_workers", type=int, default=1, help="Total number of parallel frame workers")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    model_params_processed = model_params.extract(args)
    model_params_processed.ipc_port = args.ipc_port
    model_params_processed.render = args.render
    model_params_processed.axis_flip = args.axis_flip
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
    loss_params = get_loss_config(type='dynamic')
    renderer_params = get_cs_renderer_config(type='dynamic')

    training(
        model_params=model_params_processed, 
        dynamic_training_params=dynamic_training_params,
        loss_params=loss_params,
        renderer_params=renderer_params,
        pipeline_params=pipeline_params_processed,
        starting_checkpoint=starting_checkpoint,
        test_iters=test_iters,
        )

    # All done
    print("\nTraining complete.")
