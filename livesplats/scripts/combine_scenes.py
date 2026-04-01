import sys
import os
import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from scene_hs import GaussianModel
from utils.dynamic_utils import render_samples
from cs_renderers import cs_renderer_factory, get_cs_renderer_config
import datasets.dataset_factory as dataset_factory
from arguments import PipelineParams


class GaussianData:
    def __init__(self, ply_path=None):
        self.xyz = np.empty((0, 3))
        self.f_dc = np.empty((0, 3))
        self.f_rest = np.empty((0, 0))
        self.opacities = np.empty((0, 1))
        self.scales = np.empty((0, 3))
        self.rotations = np.empty((0, 4))
        
        if ply_path:
            self.load_ply(ply_path)

    def load_ply(self, path):
        try:
            plydata = PlyData.read(path)
            v = plydata['vertex']
            
            self.xyz = np.stack((v['x'], v['y'], v['z']), axis=-1)
            self.f_dc = np.stack((v['f_dc_0'], v['f_dc_1'], v['f_dc_2']), axis=-1)
            
            # Dynamically find f_rest
            f_rest_names = [p.name for p in v.properties if p.name.startswith("f_rest_")]
            f_rest_names = sorted(f_rest_names, key=lambda x: int(x.split('_')[-1]))
            
            if f_rest_names:
                self.f_rest = np.stack([v[n] for n in f_rest_names], axis=-1)
            else:
                self.f_rest = np.empty((self.xyz.shape[0], 0))
                
            self.opacities = v['opacity'][..., np.newaxis]
            self.scales = np.stack((v['scale_0'], v['scale_1'], v['scale_2']), axis=-1)
            self.rotations = np.stack((v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']), axis=-1)
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise e

def merge_gaussians(gaussian_list):
    merged = GaussianData()
    if not gaussian_list:
        return merged
    min_sh = min([g.f_rest.shape[1] for g in gaussian_list])
    merged.xyz = np.concatenate([g.xyz for g in gaussian_list], axis=0)
    merged.f_dc = np.concatenate([g.f_dc for g in gaussian_list], axis=0)
    merged.f_rest = np.concatenate([g.f_rest[:, :min_sh] for g in gaussian_list], axis=0)
    merged.opacities = np.concatenate([g.opacities for g in gaussian_list], axis=0)
    merged.scales = np.concatenate([g.scales for g in gaussian_list], axis=0)
    merged.rotations = np.concatenate([g.rotations for g in gaussian_list], axis=0)
    return merged

def save_composed_data(composed_data, output_path):
    count = composed_data.xyz.shape[0]
    
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    
    num_f_rest = composed_data.f_rest.shape[1]
    for i in range(num_f_rest):
        dtype_list.append((f'f_rest_{i}', 'f4'))
        
    dtype_list.append(('opacity', 'f4'))
    for i in range(3): dtype_list.append((f'scale_{i}', 'f4'))
    for i in range(4): dtype_list.append((f'rot_{i}', 'f4'))
        
    elements = np.empty(count, dtype=dtype_list)
    
    elements['x'] = composed_data.xyz[:, 0]
    elements['y'] = composed_data.xyz[:, 1]
    elements['z'] = composed_data.xyz[:, 2]
    elements['nx'] = np.zeros(count, dtype='f4')
    elements['ny'] = np.zeros(count, dtype='f4')
    elements['nz'] = np.zeros(count, dtype='f4')
    elements['f_dc_0'] = composed_data.f_dc[:, 0]
    elements['f_dc_1'] = composed_data.f_dc[:, 1]
    elements['f_dc_2'] = composed_data.f_dc[:, 2]
    
    for i in range(num_f_rest):
        elements[f'f_rest_{i}'] = composed_data.f_rest[:, i]
        
    elements['opacity'] = composed_data.opacities.flatten()
    for i in range(3): elements[f'scale_{i}'] = composed_data.scales[:, i]
    for i in range(4): elements[f'rot_{i}'] = composed_data.rotations[:, i]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el], text=False).write(output_path)

def render_scene(ply_path, source_path, dataset_type, frame_idx):
    """
    Loads the merged PLY and renders it using the project's pipeline.
    """
    print(f"Rendering frame {frame_idx} from {ply_path}...")
    
    pp_parser = ArgumentParser()
    pp = PipelineParams(pp_parser)
    pipeline_params = pp.extract(pp_parser.parse_args([]))
    
    dataset_conf = dataset_factory.DATASET_CONF.copy()
    dataset_conf['data_path'] = source_path
    dataset_conf['dataset_caching'] = False
    dataset_conf['t_idx'] = frame_idx
    dataset_conf['timestamp'] = frame_idx
    dataset_conf['spatial_scale_factor'] = 1.0
    dataset_conf['img_scale_factor'] = 0.5
    
    try:
        dataset = dataset_factory.dataset_factory(dataset_type, dataset_conf)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    gaussians = GaussianModel(1) # Assuming sh_degree 1
    gaussians.load_ply(ply_path, dataset.nerfpp_norm["radius"])
    
    renderer_params = get_cs_renderer_config(type='dynamic')
    renderer = cs_renderer_factory(type=renderer_params['type'], conf=renderer_params['conf'])
    
    output_folder = os.path.dirname(ply_path)
    render_samples(
        gaussians, 
        dataset, 
        renderer.render, 
        pipeline_params, 
        frame_idx, 
        output_folder, 
        show_vs=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_plys", nargs='+', required=True, help="List of PLY files to merge")
    parser.add_argument("--output_path", required=True, help="Output PLY path")
    
    parser.add_argument("--render", action="store_true", help="Render samples after merging")
    parser.add_argument("--source_path", type=str, help="Path to scene data (needed for rendering)")
    parser.add_argument("--dataset_type", type=str, default="basket_mv", help="Dataset type")
    
    args = parser.parse_args()

 
    data_list = []
    for p in args.input_plys:
        data_list.append(GaussianData(p))
        
    merged = merge_gaussians(data_list)
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    save_composed_data(merged, args.output_path)


    # Render (Optional)
    if args.render:
        if not args.source_path:
            print("Skipping render: --source_path is required.")
        else:
            try:
                filename = os.path.basename(args.output_path)
                frame_idx = int(os.path.splitext(filename)[0])
                
                render_scene(args.output_path, args.source_path, args.dataset_type, frame_idx)
            except Exception as e:
                print(f"Error during rendering: {e}")
