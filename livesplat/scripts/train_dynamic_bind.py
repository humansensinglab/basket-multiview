import os, torch
import sys
from argparse import ArgumentParser
from typing import Dict
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
from post_proc import get_post_proc_config, post_proc_factory
from scene_hs import GaussianModel
from cs_renderers import network_gui_ws
from utils.dynamic_utils import (create_output, get_timestamps,
                                 prepare_output_and_logger, render_samples,
                                 send_data_to_device, training_report)
from scene_hs.skeleton_global import SkeletonGlobal
from utils.general_utils import safe_state
from collections import defaultdict
from torchvision.utils import save_image


def training_frame(
    model_params : GroupParams, 
    dynamic_training_params : DynamicTraningConf,
    loss_params: Dict,
    renderer_params: Dict,
    pipeline_params : GroupParams, 
    gaussian_model : GaussianModel, 
    motion_info: Dict,
    cur_dataset: dataset_factory.BaseDataset,
    frame_idx : int,
    num_total_frames : int,
    test_iters : list,
    ):

    spatial_lr_scale = cur_dataset.nerfpp_norm["radius"] 

    if not model_params.initialized:
        if model_params.first_frame_obj:
            gaussian_model.load_obj(model_params.first_frame_obj, spatial_lr_scale)
        elif model_params.first_frame_ply:
            gaussian_model.load_ply(model_params.first_frame_ply, spatial_lr_scale)
        else:
            num_points = 1000000
            center = cur_dataset.nerfpp_norm['translate']
            lenght = cur_dataset.nerfpp_norm['radius']
            gaussian_model.create_random(
                center=center, 
                lenght=lenght, 
                spatial_lr_scale=spatial_lr_scale,
                num_points=num_points)
            
        model_params.initialized = True
    
    cur_opt_params = dynamic_training_params.optimization.get_opt_conf('static')
    gaussian_model.training_setup(
        optimization_params=cur_opt_params,
        scheduler_params=dynamic_training_params.optimization.get_scheduler_conf('static'),
        densification_params=dynamic_training_params.densification)

    loss_object = loss_factory(type=loss_params['type'], conf=loss_params['conf'])
    renderer = cs_renderer_factory(type=renderer_params['type'], conf={'background_mode': 'random'})
    viz_renderer = cs_renderer_factory(type='default', conf={'background_mode': 'black'})      

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # One iteration = optimizing one batch (no matter the batch size)
    num_iters = cur_opt_params.iterations
    
    iteration = 1
    ema_dict_for_log = defaultdict(int)
    
    # Batch size needs to be taken into account for num_iters
    progress_bar = tqdm(range(0, num_iters), desc="Training progress")

    num_images = cur_dataset.__len__()
    
    path_to_output_folder = create_output(
        frame_idx, 
        model_params, 
        cur_dataset, 
        gaussian_model)

    skl_path = os.path.join(model_params.source_path, "skls/0000.skl")
    skl_joints = SkeletonGlobal(skl_path).joints if os.path.isfile(skl_path) else None

    while iteration <= num_iters:

        iter_start.record()
        cur_img_idx = iteration % num_images
        
        cur_data = cur_dataset.__getitem__(cur_img_idx)
                
        gaussian_model.update_learning_rate(iteration)
        
        if iteration % 1000 == 0:
            gaussian_model.oneupSHdegree()
    
        send_data_to_device(cur_data)
        bg_color=renderer.get_background()
        
        rendered_out = renderer.render(
            sample=cur_data, 
            gaussian_model=gaussian_model, 
            pipeline_params=pipeline_params,
            bg_color=bg_color
        )
        
        viewspace_point_tensor = rendered_out["viewspace_points"]
        visibility_filter = rendered_out["visibility_filter"]
        radii = rendered_out["radii"]
                
        loss_dict = loss_object.compute_loss(
            gt_dict=cur_data,
            pred_dict=rendered_out,
            gaussian_model=gaussian_model,
            motion_info=motion_info,
            bg_color=bg_color,
            skl_joints=skl_joints
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
            
            # Densification
            if iteration < dynamic_training_params.densification.densify_until_iter and not model_params.global_illumination:
                # Keep track of max radii in image-space for pruning
                gaussian_model.max_radii2D[visibility_filter] = torch.max(
                    gaussian_model.max_radii2D[visibility_filter], 
                    radii[visibility_filter])
                
                gaussian_model.add_densification_stats(
                    viewspace_point_tensor, 
                    visibility_filter)

                if iteration > dynamic_training_params.densification.densify_from_iter and iteration % dynamic_training_params.densification.densification_interval == 0:
                    size_threshold = 20 if iteration > dynamic_training_params.densification.opacity_reset_interval else None
                    gaussian_model.densify_and_prune(
                        dynamic_training_params.densification.densify_grad_threshold, 
                        0.005, 
                        cur_dataset.nerfpp_norm["radius"], 
                        size_threshold)
                
                if iteration % dynamic_training_params.densification.opacity_reset_interval == 0 or \
                    (model_params.white_background and iteration == dynamic_training_params.densification.densify_from_iter):
                    gaussian_model.reset_opacity()    
        
            # Optimizer step
            if iteration < num_iters:
                gaussian_model.optimizer.step()
                gaussian_model.optimizer.zero_grad(set_to_none = True)
            
            if iteration == (num_iters - 500):
                gaussian_model.filter_by_2d_masks(cur_dataset)
                
        iteration += 1
        
    # Save the current gaussian to the dynamic gausssian sequence
    print("Saving checkpoints...")
    checkpoint_path = os.path.join(path_to_output_folder, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    gaussian_model.save_ply(os.path.join(checkpoint_path, f"{frame_idx:04d}.ply"))
    
    # render samples
    print("Rendering samples...")
    render_samples(
        gaussian_model, 
        cur_dataset, 
        viz_renderer.render, 
        pipeline_params, 
        frame_idx, 
        path_to_output_folder
    )
    

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
    timestamps = get_timestamps(model_params.source_path if model_params.rgb_source_path == '' else model_params.rgb_source_path, subsample_step=1)
    
    # Initialize Gaussians
    gaussian_model = GaussianModel(1)
    model_params.initialized = False
    starting_t = -1
    if starting_checkpoint:
        (model_params, starting_t) = torch.load(starting_checkpoint)
        starting_idx = -1
        for cur_idx, cur_t in enumerate(timestamps):
            if cur_t < starting_t:
                starting_idx = cur_idx
            else:
                break
         
        gaussian_model.restore(model_params, dynamic_training_params, starting_idx)
        model_params.initialized = True
        
    prepare_output_and_logger(model_params)    

    motion_info = None
    
    for cur_idx, cur_t in enumerate(timestamps):
        
        if cur_t < starting_t: 
            continue
        
        conf = dataset_factory.DATASET_CONF 
        conf['data_path'] = model_params.source_path
        conf['rgb_source_path'] = model_params.rgb_source_path
        conf['timestamp'] = cur_t
        conf['t_idx'] = cur_idx
        conf['require_all_img_data_available'] = False
        conf['first_k_cams'] = model_params.first_k_cams
        conf['dataset_caching'] = model_params.dataset_caching
        conf['parallel'] = False
        conf['use_pickle'] = False
        conf['mesh_present'] = False
        conf['spatial_scale_factor'] = 1
        conf['img_scale_factor'] = 1.0
        conf['load_depth'] = False
        
        cur_dataset = dataset_factory.dataset_factory(
            type_dataset= model_params.dataset_type, conf=conf)

        
        training_frame(
            model_params=model_params,
            dynamic_training_params=dynamic_training_params,
            loss_params=loss_params,
            renderer_params=renderer_params,
            pipeline_params=pipeline_params,
            gaussian_model=gaussian_model,
            motion_info=motion_info,
            cur_dataset=cur_dataset,
            frame_idx=cur_idx,
            num_total_frames=len(timestamps),
            test_iters=test_iters,
        )
        
        break # bind pose only have one frame

        
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

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    model_params_processed = model_params.extract(args)
    dynamic_training_params = DynamicTraningConf()
    pipeline_params_processed = pipeline_params.extract(args)
    
    test_iters = args.test_iterations
    save_iters = args.save_iterations
    checkpoint_iterations = args.checkpoint_iterations
    starting_checkpoint = args.start_checkpoint
    debug_from = args.debug_from
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # For static frame by frame
    loss_params = get_loss_config(type='default')
    
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
