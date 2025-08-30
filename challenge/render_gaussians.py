import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

from general_utils import load_and_render_gaussians
from video_utils import create_video_from_images


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('torchvision').setLevel(logging.WARNING)
    
    return logger


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render Gaussian splatting scenes and create videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--pointcloud_path', type=str, help='Path to the directory containing Gaussian PLY files')
    parser.add_argument('--scene_name', type=str, default="atc_1", choices=["atc_1", "atc_2", "atc_3", "atc_4", "atc_4_female", "def_1", "def_2", "int_1"], help='Name of the sequence')
    parser.add_argument('--output_dir', default="predictions", type=str, help='Directory to save rendered outputs')
    parser.add_argument('--phase', type=str, help='Challenge split (interpolation or VR)', choices=["interp", "vr"])
    parser.add_argument('--camera_path', type=str, help='Path to folder containing test cameras in JSON')

    # Optional arguments
    parser.add_argument('--sh-degree', type=int, default=3, help='Spherical harmonics degree')
    parser.add_argument('--bg-color', type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=('R', 'G', 'B'), help='Background color (RGB values 0-1)')
    
    # Do not change the default values while making a submission
    parser.add_argument('--width', type=int, default=960, help='Output image width')
    parser.add_argument('--height', type=int, default=540, help='Output image height')
    parser.add_argument('--video-fps', type=int, default=30, help='Frames per second for output video')
    parser.add_argument('--video-codec', type=str, default='mp4v', choices=['mp4v', 'h264', 'hevc'], help='Video codec to use')
    parser.add_argument('--no-cleanup', action='store_true', help='Keep rendered images after creating video')
    parser.add_argument('--no-video', action='store_true', help='Skip video creation, only render images')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()


def get_frame_list(pointcloud_path: Path) -> List[int]:
    """Get list of frame numbers to render."""
    # Find all Gaussian PLY files
    ply_files = sorted(pointcloud_path.glob("gaussians_*.ply"))
    
    if not ply_files:
        raise FileNotFoundError(f"No Gaussian PLY files found in {pointcloud_path}")
    
    # Extract frame numbers
    frame_numbers = []
    for ply_file in ply_files:
        try:
            frame_num = int(ply_file.stem.split('_')[-1])
            frame_numbers.append(frame_num)
        except (ValueError, IndexError):
            continue
    
    if not frame_numbers:
        raise ValueError("No valid frame numbers found in PLY files")
        
    return sorted(frame_numbers)


def render_frame(
    frame_num: int,
    pointcloud_path: Path,
    metadata: Dict,
    args: argparse.Namespace,
    logger: logging.Logger
) -> Optional[torch.Tensor]:
    """
    Render a single frame.
    
    Returns:
        Rendered image tensor or None if rendering failed
    """
    ply_path = pointcloud_path / f"gaussians_{frame_num:04d}.ply"
    
    if not ply_path.exists():
        logger.warning(f"PLY file not found: {ply_path}")
        return None
    
    try:
        # Get camera parameters for this frame
        frame_metadata = metadata[frame_num]
        w2c = np.array(frame_metadata["w2c"])
        k = np.array(frame_metadata["K"])
        
        # Render the frame
        rendered_image = load_and_render_gaussians(
            ply_path=str(ply_path),
            K=k,
            R=w2c[:3, :3].T,
            T=w2c[:3, 3],
            image_width=args.width,
            image_height=args.height,
            sh_degree=args.sh_degree,
            bg_color=tuple(args.bg_color),
        )
        
        return rendered_image
        
    except KeyError:
        logger.warning(f"No camera metadata for frame {frame_num}")
        return None
    except Exception as e:
        logger.error(f"Failed to render frame {frame_num}: {e}")
        return None


def render_sequence(
    pointcloud_path: Path,
    output_path: Path,
    metadata: Dict,
    args: argparse.Namespace,
    logger: logging.Logger
) -> List[Path]:
    """
    Render all frames in the sequence.
    
    Returns:
        List of paths to rendered images
    """
    frame_numbers = get_frame_list(pointcloud_path)
    logger.info(f"Found {len(frame_numbers)} frames to render")
    
    logger.info(f"Rendering {len(frame_numbers)} frames")
    
    rendered_paths = []
    failed_frames = []
    
    # Progress bar for rendering
    with tqdm(frame_numbers, desc="Rendering frames", unit="frame") as pbar:
        for frame_num in pbar:
            rendered_image = render_frame(
                frame_num, pointcloud_path, metadata, args, logger
            )
            
            if rendered_image is not None:
                # Save the rendered image
                output_file = output_path / f"{frame_num:06d}.png"
                save_image(rendered_image, str(output_file))
                rendered_paths.append(output_file)
                
                pbar.set_postfix({'last_frame': frame_num})
            else:
                failed_frames.append(frame_num)
    
    if failed_frames:
        logger.warning(f"Failed to render {len(failed_frames)} frames: {failed_frames[:10]}...")
    
    logger.info(f"Successfully rendered {len(rendered_paths)} frames")
    
    return rendered_paths


def main():
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    logger.info("Starting Gaussian splatting renderer")
    logger.debug(f"Arguments: {vars(args)}")
    
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        pointcloud_path = Path(args.pointcloud_path) / args.scene_name / "point_cloud"

        output_path = Path(args.output_dir) / args.phase / args.scene_name
        output_path.mkdir(parents=True, exist_ok=True)

        camera_file = Path(args.camera_path) / args.phase / f"{args.scene_name}.json"
        with open(camera_file, 'r') as f:
            metadata = json.load(f)
        
        # Render all frames
        rendered_paths = render_sequence(
            pointcloud_path, output_path, metadata, args, logger
        )
        
        if not rendered_paths:
            logger.error("No frames were successfully rendered")
            sys.exit(1)
        
        if not args.no_video:
            video_path = output_path.parent / f"{args.scene_name}.mp4"
            
            logger.info(f"Creating video: {video_path}")
            success = create_video_from_images(
                image_dir=output_path,
                output_path=video_path,
                fps=args.video_fps,
                codec=args.video_codec,
                logger=logger
            )
            
            if success:
                logger.info(f"Video created successfully: {video_path}")
                
                if not args.no_cleanup:
                    shutil.rmtree(output_path)
            else:
                logger.error("Failed to create video")
                sys.exit(1)
        
        logger.info("Rendering complete")

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()