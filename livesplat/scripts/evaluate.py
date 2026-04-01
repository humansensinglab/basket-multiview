import os
import sys
import subprocess as sp
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import numpy as np
import json
import imageio.v2 as imageio
import lpips
from PIL import Image
import concurrent.futures
import torch.nn.functional as F
from os.path import join
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from scene_hs.skeleton_global import SkeletonGlobal, skeleton_error
from utils.image_utils import psnr, psnr_masked
from losses._utils import fast_ssim, ssim

class EvalPathConfig:
    def __init__(
        self,
        output_path,
        dynamic_dir_rel="dynamic",
        renders_dir_rel="combined",
        frame_dir_pattern="{frame_idx}",
        renders_subdir="sample_renders",
        render_pattern="render_{cam_idx}.png",
        gt_pattern="gt_{cam_idx}.png",
        video_dir_rel="videos",
        video_pattern="{mode}_cam_{cam_idx}.mp4",
        skl_path_pattern="{frame_idx}/checkpoints/{frame_idx:04d}.skl",
        group_player_dir_pattern="player_{player_id}",
        group_skl_path_pattern="skeleton/player_{player_id}/{frame_id:04d}.skl",
        group_skl_data_pattern="player_{player_id}/dynamic/skls/{frame_id:04d}.skl",
    ):
        self.output_path = output_path
        self.dynamic_dir_rel = dynamic_dir_rel
        self.renders_dir_rel = renders_dir_rel if renders_dir_rel is not None else dynamic_dir_rel
        self.frame_dir_pattern = frame_dir_pattern
        self.renders_subdir = renders_subdir
        self.render_pattern = render_pattern
        self.gt_pattern = gt_pattern
        self.video_dir_rel = video_dir_rel
        self.video_pattern = video_pattern
        self.skl_path_pattern = skl_path_pattern
        self.group_player_dir_pattern = group_player_dir_pattern
        self.group_skl_path_pattern = group_skl_path_pattern
        self.group_skl_data_pattern = group_skl_data_pattern

    @property
    def dynamic_dir(self):
        return os.path.join(self.output_path, self.dynamic_dir_rel)

    @property
    def renders_root(self):
        return os.path.join(self.output_path, self.renders_dir_rel)

    @property
    def video_dir(self):
        return os.path.join(self.output_path, self.video_dir_rel)

    def frame_dir(self, frame_idx):
        return os.path.join(self.renders_root, self.frame_dir_pattern.format(frame_idx=frame_idx))

    def group_skl_data_path(self, data_dir, player_id, frame_id):
        return join(data_dir, self.group_skl_data_pattern.format(player_id=player_id, frame_id=frame_id))

    def renders_dir(self, frame_idx):
        return os.path.join(self.frame_dir(frame_idx), self.renders_subdir)

    def render_path(self, frame_idx, cam_idx):
        return os.path.join(self.renders_dir(frame_idx), self.render_pattern.format(cam_idx=cam_idx, frame_idx=frame_idx))

    def gt_path(self, frame_idx, cam_idx):
        return os.path.join(self.renders_dir(frame_idx), self.gt_pattern.format(cam_idx=cam_idx, frame_idx=frame_idx))

    def video_path(self, mode, cam_idx):
        return os.path.join(self.video_dir, self.video_pattern.format(mode=mode, cam_idx=cam_idx))

    def skl_path(self, frame_idx):
        return os.path.join(self.dynamic_dir, self.skl_path_pattern.format(frame_idx=frame_idx))

    def group_player_dynamic_dir(self, player_id):
        return os.path.join(self.dynamic_dir, self.group_player_dir_pattern.format(player_id=player_id))

    def group_player_skl_path(self, player_id, frame_idx):
        return os.path.join(
            self.group_player_dynamic_dir(player_id),
            self.group_skl_path_pattern.format(frame_idx=frame_idx)
        )

def read_img(path):
    img = torch.tensor(np.array(Image.open(path).convert('RGB')), dtype=torch.float32, device='cuda:0').permute(2,0,1).contiguous() / 255
    return img

def write_video(video_dir, path_cfg, test_cam_inds, num_frames, mode, overwrite=True):
    for cam_id in tqdm(test_cam_inds, desc=f"Creating {mode} videos", ncols=100):
        video_path = path_cfg.video_path(mode, cam_id)
        if os.path.isfile(video_path):
            continue
        frames = []
        for frame_idx in range(num_frames):
            img_path = path_cfg.render_path(frame_idx, cam_id) if mode == "render" else path_cfg.gt_path(frame_idx, cam_id)
            frames.append(imageio.imread(img_path))
        imageio.mimwrite(video_path, frames, fps=30, format='FFMPEG', codec='libx264')

@torch.no_grad()
def eval_img_metrics(path_cfg, test_views, num_frames):

    # Get image paths
    render_paths = []
    gt_paths = []
    for frame_idx in range(num_frames):
        for cam_idx in test_views:
            render_path = path_cfg.render_path(frame_idx, cam_idx)
            gt_path = path_cfg.gt_path(frame_idx, cam_idx)
            render_paths.append(render_path)
            gt_paths.append(gt_path)
    num_imgs = len(render_paths)
    
    # Compute PSNR and SSIM
    def compute_psnr_ssim(render_path, gt_path, pgbr):
        render_img = read_img(render_path)
        gt_img = read_img(gt_path)
        cur_psnr = psnr(render_img, gt_img).mean().item()
        cur_ssim = fast_ssim(render_img, gt_img).item()
        pgbr.update(1)
        return cur_psnr, cur_ssim
    psnr_list = []
    ssim_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        task_list = []
        pgbr = tqdm(total=len(render_paths), ncols=100, desc="Computing PSNR and SSIM")
        for render_path, gt_path in zip(render_paths, gt_paths):
            task_list.append(executor.submit(compute_psnr_ssim, render_path, gt_path, pgbr))
        for future in concurrent.futures.as_completed(task_list):
            psnr_list.append(future.result()[0])
            ssim_list.append(future.result()[1])
        pgbr.close()
    
    # Compute LPIPS
    def load_imgs(render_path, gt_path):
        render_img = read_img(render_path)
        gt_img = read_img(gt_path)
        return [render_img, gt_img]
    lpips_list = []
    lpips_model = lpips.LPIPS(net='vgg').cuda().eval()
    bz = 1
    start = 0
    end = min(start + bz, num_imgs)
    pgbr = tqdm(total=num_imgs, ncols=100, desc="Computing LPIPS")
    while start < num_imgs:
        render_imgs = []
        gt_imgs = []
        task_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=bz) as executor:
            for render_path, gt_path in zip(render_paths[start:end], gt_paths[start:end]):
                task_list.append(executor.submit(load_imgs, render_path, gt_path))
            for future in concurrent.futures.as_completed(task_list):
                render_imgs.append(future.result()[0])
                gt_imgs.append(future.result()[1])
        render_imgs = torch.stack(render_imgs, dim=0)
        gt_imgs = torch.stack(gt_imgs, dim=0)
        lpips_list.append(np.array(lpips_model(gt_imgs, render_imgs).cpu().detach().flatten()))
        pgbr.update(end-start)
        start = end
        end = min(end + bz, num_imgs)
    pgbr.close()
    lpips_list = np.concatenate(lpips_list, axis=0)
    
    return {
        'psnr_avg': np.mean(psnr_list),
        'ssim_avg': np.mean(ssim_list),
        'lpips_avg': float(np.mean(lpips_list)),
    }

def eval_masked_psnr(path_cfg, test_views, num_frames, dataset_dir):
    "Evaluate PSNR only in masked regions of players and the basketball."
    
    def compute_masked_psnr(render_path, gt_path, semantic_path, pgbr):
        render_img = read_img(render_path)
        gt_img = read_img(gt_path)
        semantic = read_img(semantic_path)
        if semantic.shape[1] != render_img.shape[1] or semantic.shape[2] != render_img.shape[2]:
            semantic = F.interpolate(semantic.unsqueeze(0), size=render_img.shape[1:]).squeeze(0)
        mask = ((semantic > 0).sum(dim=0, keepdim=True) > 0)
        if mask.sum() == 0:
            pgbr.update(1)
            return None
        pgbr.update(1)
        return psnr_masked(render_img, gt_img, mask).mean().item()
    
    masked_psnr_list = []
    pgbr = tqdm(total=num_frames*len(test_views), ncols=100, desc="Computing masked PSNR")
    for frame_idx in range(num_frames):
        for cam_idx in test_views:
            render_path = path_cfg.render_path(frame_idx, cam_idx)
            gt_path = path_cfg.gt_path(frame_idx, cam_idx)
            if 'full' in os.listdir(dataset_dir):
                semantic_path = os.path.join(dataset_dir, f'full/cams/{cam_idx+1}/CameraComponent/SemanticImage/1_{cam_idx+1}.{frame_idx:04}.png')
            else:
                semantic_path = os.path.join(dataset_dir, f'dynamic/cams/{cam_idx+1}/CameraComponent/SemanticImage/1_{cam_idx+1}.{frame_idx:04}.png')
            res = compute_masked_psnr(render_path, gt_path, semantic_path, pgbr)
            if res is not None:
                masked_psnr_list.append(res)
    
    return {
        'masked_psnr_avg': np.mean(masked_psnr_list)
    }

def eval_masked_psnr_masks_dir(path_cfg, test_views, num_frames, masks_dir):
    """Evaluate PSNR only in foreground regions using pre-computed masks.
    Mask files live at: masks_dir/cam_{cam_idx:04d}/{frame_idx:04d}.png
    Background is pure black (0,0,0); any non-black pixel is foreground.
    """

    def compute_masked_psnr(render_path, gt_path, mask_path, pgbr):
        render_img = read_img(render_path)
        gt_img = read_img(gt_path)
        mask_img = read_img(mask_path)  # (3, H, W), values in [0,1]
        if mask_img.shape[1] != render_img.shape[1] or mask_img.shape[2] != render_img.shape[2]:
            mask_img = F.interpolate(mask_img.unsqueeze(0), size=render_img.shape[1:]).squeeze(0)
        mask = ((mask_img > 0).any(dim=0, keepdim=True))  # (1, H, W) foreground
        pgbr.update(1)
        if mask.sum() == 0:
            return None
        return psnr_masked(render_img, gt_img, mask).mean().item()

    masked_psnr_list = []
    pgbr = tqdm(total=num_frames * len(test_views), ncols=100, desc="Computing masked PSNR")
    for frame_idx in range(num_frames):
        for cam_idx in test_views:
            render_path = path_cfg.render_path(frame_idx, cam_idx)
            gt_path = path_cfg.gt_path(frame_idx, cam_idx)
            mask_path = os.path.join(masks_dir, f'cam_{cam_idx:04d}/{frame_idx:04d}.png')
            res = compute_masked_psnr(render_path, gt_path, mask_path, pgbr)
            if res is not None:
                masked_psnr_list.append(res)
    pgbr.close()

    return {
        'masked_psnr_avg': np.mean(masked_psnr_list)
    }


def eval_vmaf(video_dir, test_views):
    per_view_score = []
    for view in tqdm(test_views, desc="Computing VMAF per view", ncols=100):
        a = sp.run(
            f"ffmpeg -i {video_dir}/render_cam_{view}.mp4 -i {video_dir}/gt_cam_{view}.mp4 -filter_complex libvmaf -f null -", 
            capture_output=True,
            shell=True,
            text=True)
        per_view_score.append(float(a.stderr.split("\n")[-4].split(" ")[-1]))
    
    return {
        'vmaf_avg': sum(per_view_score) / len(per_view_score),
        'vmaf_min': min(per_view_score),
        'vmaf_max': max(per_view_score),
        'vmaf_min_view': test_views[per_view_score.index(min(per_view_score))],
        'vmaf_max_view': test_views[per_view_score.index(max(per_view_score))],
    }

def eval_skl_error(path_cfg, data_dir, num_frames):    
    skl_error_list = []
    for frame_id in tqdm(range(num_frames), ncols=100, desc="Computing skl error"):
        res_skl_path = path_cfg.skl_path(frame_id)
        gt_skl_path = join(data_dir, f'dynamic/skls/{frame_id:04d}.skl')
        res_skl = SkeletonGlobal(res_skl_path, 'cpu', requires_grad=False)
        gt_skl = SkeletonGlobal(gt_skl_path, 'cpu', requires_grad=False)
        skl_error_list.append(skeleton_error(res_skl, gt_skl).item())
    
    return {
        'skl_error': np.mean(skl_error_list)
    }

def eval_group_skl_error(path_cfg, data_dir, num_frames):
    num_players = len([x for x in os.listdir(path_cfg.dynamic_dir) if x.startswith('player_')])
    skl_error_list = []
    for player_id in range(num_players):
        for frame_id in tqdm(range(num_frames), ncols=100, desc=f"Computing player_{player_id} skl error"):
            res_skl_path = path_cfg.group_player_skl_path(player_id, frame_id)
            gt_skl_path = path_cfg.group_skl_data_path(data_dir, player_id, frame_id)
            res_skl = SkeletonGlobal(res_skl_path, 'cpu', requires_grad=False)
            gt_skl = SkeletonGlobal(gt_skl_path, 'cpu', requires_grad=False)
            skl_error_list.append(skeleton_error(res_skl, gt_skl).item())
    
    return {
        'skl_error': np.mean(skl_error_list)
    }
    
if __name__ == '__main__':
    
    """
    python scripts/evaluate.py -s output/running_black_jw
    """
    
    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--vmaf', action='store_true', default=False)
    parser.add_argument('--img_metrics', action='store_true', default=False)
    parser.add_argument('--masked_psnr', action='store_true', default=False)
    parser.add_argument('--source_path', type=str, default='')
    parser.add_argument('--skl_error', action='store_true', default=False)
    parser.add_argument('--group_skl_error', action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    path_cfg = EvalPathConfig(output_path=args.output_path)
    cam_inds = sorted([int(x[7:-4]) for x in os.listdir(path_cfg.renders_dir(0)) if x.startswith('render_')])
    num_frames = len([x for x in os.listdir(path_cfg.renders_root) if x.isnumeric()])

    test_cam_inds = [x for x in range(max(cam_inds)+1) if x%10==0]
    print(f'num_frames: {num_frames} \n'+ \
            f'cam_inds: {cam_inds} \n'+ \
            f'test_cam_inds: {test_cam_inds}')
    
    metrics = {}
    
    if args.vmaf:
        os.makedirs(path_cfg.video_dir, exist_ok=True)
        write_video(path_cfg.video_dir, path_cfg, test_cam_inds, num_frames, 'render')    
        write_video(path_cfg.video_dir, path_cfg, test_cam_inds, num_frames, 'gt')
    if args.img_metrics:
        metrics.update(eval_img_metrics(path_cfg, test_cam_inds, num_frames))
    if args.masked_psnr:
        metrics.update(eval_masked_psnr_masks_dir(path_cfg, test_cam_inds, num_frames, os.path.join(args.source_path, "masks")))
    
    if args.skl_error:
        metrics.update(eval_skl_error(path_cfg, args.source_path, num_frames))
    if args.group_skl_error:
        metrics.update(eval_group_skl_error(path_cfg, args.source_path, num_frames))

    # Display
    for k, v in metrics.items():
        print(f"{k}\t: {v}")
    
    # Dump to json file
    json_path = os.path.join(args.output_path, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics dumped to {json_path}.")