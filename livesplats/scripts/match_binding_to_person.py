import os
import sys
import torch
import json
import glob
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from cs_renderers.cs_renderers_factory import get_cs_renderer_config
from scene_hs import GaussianModel
from utils.dynamic_utils import (
    indices_from_file, 
    read_skin_file, 
    SkeletonGlobal, 
    transform_gaussians_by_skeleton
)
from cs_renderers import cs_renderer_factory
from losses import l1_loss, ssim
import datasets.dataset_factory as dataset_factory
from arguments import ModelParams, PipelineParams

def find_binding_candidates(binding_root):
    """
    Scans the binding root for all valid PLY files.
    Structure assumed: binding_root/gender/clothing_X/skin_Y/0/checkpoints/0000.ply
    """
    candidates = []
    # Search for all 0000.ply files deep inside the binding root
    ply_files = glob.glob(os.path.join(binding_root, "**", "0000.ply"), recursive=True)
    
    for ply_path in ply_files:
        parts = ply_path.split(os.sep)
        try:
            # Assumes structure: .../gender/clothing_X/skin_Y/0/checkpoints/0000.ply
            skin_name = parts[-4] 
            clothing_name = parts[-5]
            gender = parts[-7]
            
            candidates.append({
                "name": f"{gender}/{clothing_name}/{skin_name}",
                "ply_path": ply_path,
                "gender": gender,
                "clothing": clothing_name,
                "skin": skin_name
            })
        except IndexError:
            continue
            
    return candidates

def evaluate_match(
    model_params, 
    object_key, 
    binding_ply, 
    dataset, 
    device="cuda",
    num_views=5
):
    """
    Renders the binding gaussian in the scene pose and compares with GT.
    Averages loss across multiple views.
    """
    dataset_root = Path(model_params.source_path)
    
    if model_params.dataset_type == 'basket_mv':   
        binding_ply_path = Path(binding_ply)
        assoc_path = binding_ply_path.parents[2] / "assoc.txt"
        
        binding_skl_path = binding_ply_path.parents[5] / "skls" / "0000.skl" 
        skin_weight_path = binding_ply_path.parents[5] / "skinning_weights.skin"
        
        dynamic_skl_path = dataset_root / "skeleton" / object_key / "0000.skl"
        
        skel_scale = 1 / 100.0
    else:
        raise NotImplementedError("Only basket_mv implemented for now")

    if not assoc_path.exists():
        print(f"Skipping {binding_ply}: Missing assoc.txt at {assoc_path}")
        return float('inf')

    gaussians = GaussianModel(1) 
    spatial_lr_scale = dataset.nerfpp_norm["radius"]
    gaussians.load_ply(str(binding_ply), spatial_lr_scale=spatial_lr_scale)
    
    try:
        # Load indices mapping Gaussians to Vertices
        gs2vert = indices_from_file(assoc_path).to(device)
        
        # Load Binding Skeleton (Rest Pose)
        binding_skeleton = SkeletonGlobal(
            str(binding_skl_path), 
            device=device, 
            skel_scale=skel_scale
        )
        
        # Load Dynamic Skeleton (Target Pose)
        animated_skeleton = SkeletonGlobal(
            str(dynamic_skl_path), 
            device=device, 
        )
        
        # Load Skinning Weights
        vert_skn_weights, vert_skn_weights_inds = read_skin_file(
            skn_path=str(skin_weight_path),
            skl_path=str(binding_skl_path)
        )
        
        # Map weights to Gaussians
        gs_skn_weights = vert_skn_weights[gs2vert]
        gs_skn_weights_inds = vert_skn_weights_inds[gs2vert]
        
    except Exception as e:
        print(f"Error loading skeleton/weights for {binding_ply}: {e}")
        return float('inf')

    # Transform Gaussians (LBS)
    chains = animated_skeleton.precomp_chains()
    transform_gaussians_by_skeleton(
        gaussians, 
        binding_skeleton, 
        animated_skeleton, 
        gs_skn_weights, 
        gs_skn_weights_inds, 
        chains
    )

    # Setup simple renderer
    renderer_params = get_cs_renderer_config(type='dynamic')
    renderer = cs_renderer_factory(type=renderer_params['type'], conf=renderer_params['conf'])
    bg_color = renderer.get_background()
    pipeline_params = PipelineParams(ArgumentParser()) # Defaults

    total_loss_accum = 0.0
    valid_views_count = 0
    
    # Iterate through views (up to num_views or dataset length)
    views_to_process = min(num_views, len(dataset))
    
    for i in range(views_to_process):
        view = dataset.__getitem__(i)
        
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view[k] = v.to(device)

        # Render
        render_pkg = renderer.render(
            sample=view,
            gaussian_model=gaussians,
            pipeline_params=pipeline_params,
            bg_color=bg_color
        )
        
        image = render_pkg["render"]
        gt_image = view["im"]
        alpha = view["alpha"]
        mask = alpha > 0.5
        _, h, w = gt_image.shape
        bg = bg_color.T.unsqueeze(2).repeat(1, h, w)
        gt_image = gt_image * mask + (~mask) * bg
        
        # Compute Metrics (Lower is better)
        l1 = l1_loss(image, gt_image)
        score_ssim = ssim(image, gt_image)
        
        # Combined score 
        current_loss = (1.0 - 0.8) * l1 + 0.8 * (1.0 - score_ssim)
        total_loss_accum += current_loss.item()
        valid_views_count += 1
    
    if valid_views_count == 0:
        return float('inf')
        
    return total_loss_accum / valid_views_count


def main():
    parser = ArgumentParser(description="Match Binding Gaussians to Scene Person")
    
    # Standard Model Params
    lp = ModelParams(parser)
    
    parser.add_argument('--binding_root', type=str, required=True, 
                        help="Root directory containing all binding checkpoints")
    parser.add_argument('--output_json', type=str, default="matching_results.json")
    parser.add_argument('--num_views', type=int, default=1, 
                        help="Number of views to use for matching (default: 5)")
    
    args = parser.parse_args(sys.argv[1:])
    model_params = lp.extract(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Searching for candidates in {args.binding_root}...")
    candidates = find_binding_candidates(args.binding_root)
    print(f"Found {len(candidates)} binding candidates.")
    
    if len(candidates) == 0:
        print("No candidates found. Check --binding_root path.")
        return

    print(f"Loading scene data for {model_params.object_key}...")
    
    # Configure dataset factory args manually 
    dataset_conf = dataset_factory.DATASET_CONF.copy()
    dataset_conf['data_path'] = model_params.source_path
    dataset_conf['object_key'] = model_params.object_key
    dataset_conf['dataset_caching'] = False
    # Load enough cameras for the requested views
    dataset_conf['first_k_cams'] = args.num_views 
    dataset_conf['t_idx'] = 0 # First timestamp/frame
    dataset_conf['timestamp'] = 0
    dataset_conf['spatial_scale_factor'] = 1.0
    dataset_conf['img_scale_factor'] = 1.0
    
    # Initialize dataset
    dataset = dataset_factory.dataset_factory(
        model_params.dataset_type, 
        dataset_conf
    )

    results = {}
    best_score = float('inf')
    best_candidate = None

    print(f"\n--- Starting Evaluation using {args.num_views} views ---")
    for cand in tqdm(candidates):
        loss = evaluate_match(
            model_params, 
            model_params.object_key, 
            cand['ply_path'], 
            dataset, 
            device,
            num_views=args.num_views
        )
        
        print(f"Candidate: {cand['name']} | Avg Loss: {loss:.5f}")
        
        results[cand['name']] = loss
        
        if loss < best_score:
            best_score = loss
            best_candidate = cand

    print("\n--- Matching Complete ---")
    if best_candidate:
        print(f"Best Match: {best_candidate['name']}")
        print(f"Best Loss:  {best_score:.5f}")
        
        # Save output
        output_data = {
            "object_key": model_params.object_key,
            "best_match": best_candidate,
            "all_scores": results
        }
        
        if not os.path.exists(os.path.dirname(args.output_json)):
            os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
            
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        print(f"Results saved to {args.output_json}")
    else:
        print("Failed to find any valid matches.")

if __name__ == "__main__":
    main()