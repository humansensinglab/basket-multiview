import cv2
import logging
import subprocess

from pathlib import Path
from typing import Optional, List


def create_video_from_images(
    image_dir: Path,
    output_path: Path,
    fps: int = 30,
    codec: str = 'mp4v',
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Create a video from a directory of images.
    
    Args:
        image_dir: Directory containing image files
        output_path: Path for the output video file
        fps: Frames per second for the video
        codec: Video codec to use ('mp4v', 'h264', 'hevc')
        logger: Logger instance for output
        
    Returns:
        True if video was created successfully, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Get list of image files
    image_files = sorted(image_dir.glob("*.png"))
    if not image_files:
        image_files = sorted(image_dir.glob("*.jpg"))
    
    if not image_files:
        logger.error(f"No image files found in {image_dir}")
        return False
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Try using ffmpeg first (if available)
    if try_ffmpeg_creation(image_dir, output_path, fps, logger):
        return True
    
    # Fallback to OpenCV
    logger.info("Falling back to OpenCV for video creation")
    logger.warning(">>Its highly recommended that you use ffmpeg<<")
    return create_video_opencv(image_files, output_path, fps, codec, logger)


def try_ffmpeg_creation(
    image_dir: Path,
    output_path: Path,
    fps: int,
    logger: logging.Logger
) -> bool:
    """
    Try to create video using ffmpeg (faster and better quality).
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if ffmpeg is available
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            return False
        
        # Get the pattern for image files
        image_files = sorted(image_dir.glob("*.png"))
        if not image_files:
            return False
        
        # Check if files are sequentially numbered
        first_file = image_files[0].name
        if first_file.replace('.png', '').isdigit() or '_' in first_file:
            # Use pattern matching for sequential files
            pattern = str(image_dir / "%06d.png")
        else:
            # Create a temporary file list
            list_file = image_dir / "filelist.txt"
            with open(list_file, 'w') as f:
                for img_file in image_files:
                    f.write(f"file '{img_file.name}'\n")
            pattern = None
        
        # Base ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-framerate', str(fps),
        ]
        
        if pattern:
            cmd.extend(['-i', pattern])
        else:
            cmd.extend([
                '-f', 'concat',
                '-safe', '0',
                '-i', str(list_file)
            ])
        
        cmd.extend([
            '-c:v', 'libx264',
            '-crf', '0',
            '-pix_fmt', 'yuv444p',
            '-preset', 'veryslow',
            str(output_path)
        ])
        
        logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
        
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Clean up temporary file list if created
        if pattern is None and list_file.exists():
            list_file.unlink()
        
        if result.returncode == 0:
            logger.info("Video created successfully with ffmpeg")
            return True
        else:
            logger.debug(f"ffmpeg error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        logger.debug("ffmpeg not found")
        return False
    except Exception as e:
        logger.debug(f"ffmpeg failed: {e}")
        return False


def create_video_opencv(
    image_files: List[Path],
    output_path: Path,
    fps: int,
    codec: str,
    logger: logging.Logger
) -> bool:
    """
    Create video using OpenCV.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read first image to get dimensions
        first_image = cv2.imread(str(image_files[0]))
        if first_image is None:
            logger.error(f"Failed to read first image: {image_files[0]}")
            return False
        
        height, width, _ = first_image.shape
        logger.info(f"Video dimensions: {width}x{height}")
        
        # Set up video writer
        fourcc_codes = {
            'mp4v': cv2.VideoWriter_fourcc(*'mp4v'),
            'h264': cv2.VideoWriter_fourcc(*'H264'),
            'hevc': cv2.VideoWriter_fourcc(*'HEVC'),
        }
        
        fourcc = fourcc_codes.get(codec, cv2.VideoWriter_fourcc(*'mp4v'))
        
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        if not out.isOpened():
            logger.error("Failed to open video writer")
            return False
        
        # Write frames
        logger.info("Writing frames to video...")
        for i, image_file in enumerate(image_files):
            frame = cv2.imread(str(image_file))
            if frame is None:
                logger.warning(f"Failed to read image: {image_file}")
                continue
            
            out.write(frame)
            
            # Log progress every 10%
            if (i + 1) % max(1, len(image_files) // 10) == 0:
                progress = (i + 1) / len(image_files) * 100
                logger.debug(f"Progress: {progress:.0f}%")
        
        out.release()
        logger.info(f"Video saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create video with OpenCV: {e}")
        return False


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    frame_skip: int = 1,
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save extracted frames
        frame_skip: Extract every Nth frame (1 = all frames)
        logger: Logger instance
        
    Returns:
        List of paths to extracted frames
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video has {frame_count} frames")
    
    extracted_paths = []
    frame_idx = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            output_path = output_dir / f"{extracted_count:06d}.png"
            cv2.imwrite(str(output_path), frame)
            extracted_paths.append(output_path)
            extracted_count += 1
        
        frame_idx += 1
    
    cap.release()
    logger.info(f"Extracted {extracted_count} frames")
    
    return extracted_paths
