import logging
from pathlib import Path
from typing import Tuple, Union

import torch
import numpy as np
from libgs.data.types import TensorSpace
from libgs.model.gaussian import GaussianModel
from libgs.renderer import Renderer, RendererConfig
from libgs.data.utils.graphics import focal2fov, getProjectionMatrix, getWorld2View2


class GaussianRenderer:
    def __init__(
        self,
        sh_degree: int = 3,
        compute_cov3D_python: bool = False,
        convert_SHs_python: bool = False,
        debug: bool = False,
        device: str = "cuda"
    ):
        """
        Initialize renderer
        
        Args:
            sh_degree: Spherical harmonics degree
            compute_cov3D_python: Use Python for covariance computation
            convert_SHs_python: Use Python for SH conversion
            debug: Enable debug mode
            device: Device to use for rendering
        """
        self.sh_degree = sh_degree
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        self.renderer_config = RendererConfig(
            compute_cov3D_python=compute_cov3D_python,
            convert_SHs_python=convert_SHs_python,
            debug=debug
        )
        
        self.gaussians = None
        self.renderer = None
        self.current_ply_path = None
    
    def load_ply(self, ply_path: Union[str, Path]) -> None:
        """
        Load Gaussian model
        
        Args:
            ply_path: Path to the PLY file
        """
        ply_path = Path(ply_path)
        
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")
        
        # Only reload if different file
        if self.current_ply_path != ply_path:
            self.gaussians = GaussianModel(self.sh_degree)
            self.gaussians.load_ply(str(ply_path))
            self.renderer = Renderer(self.renderer_config, self.gaussians)
            self.current_ply_path = ply_path
            self.logger.debug(f"Loaded PLY: {ply_path}")
    
    def render(
        self,
        K: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        image_width: int,
        image_height: int,
        bg_color: Tuple[float, float, float] = (0, 0, 0),
        znear: float = 0.01,
        zfar: float = 100.0,
        scaling_modifier: float = 1.0
    ) -> torch.Tensor:
        """
        Render the loaded Gaussian model from a given viewpoint.
        
        Args:
            K: 3x3 camera intrinsic matrix
            R: 3x3 rotation matrix (world to camera)
            T: 3x1 translation vector (world to camera)
            image_width: Output image width
            image_height: Output image height
            bg_color: Background color as RGB tuple (0-1 range)
            znear: Near clipping plane
            zfar: Far clipping plane
            scaling_modifier: Scaling factor for Gaussians
            
        Returns:
            Rendered image as tensor (C, H, W)
        """
        if self.gaussians is None or self.renderer is None:
            raise RuntimeError("No PLY file loaded. Call load_ply() first.")
        
        # Validate dims
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got {K.shape}")
        if R.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got {R.shape}")
        if T.shape not in [(3,), (3, 1)]:
            raise ValueError(f"T must be 3x1 or (3,), got {T.shape}")
        
        # Calculate FOV from intrinsics
        fovx = focal2fov(K[0, 0], image_width)
        fovy = focal2fov(K[1, 1], image_height)
        
        # Create view and projection matrices
        world_view_transform = torch.from_numpy(
            getWorld2View2(R, T.flatten())
        ).transpose(0, 1).to(self.device).float()
        
        projection_matrix = getProjectionMatrix(
            znear, zfar, fovx, fovy
        ).transpose(0, 1).to(self.device).float()
        
        # Create viewpoint
        viewpoint = TensorSpace(
            image=torch.zeros(3, image_height, image_width).to(self.device),
            fovx=fovx,
            fovy=fovy,
            world_view_transform=world_view_transform,
            full_proj_transform=world_view_transform @ projection_matrix,
            camera_center=world_view_transform.inverse()[3, :3],
        )
        
        # Set background color
        bg = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        
        # Render
        with torch.no_grad():
            render_output = self.renderer(
                viewpoint, 
                bg, 
                scaling_modifier=scaling_modifier
            )
            rendered_image = render_output["render"]
        
        return rendered_image


def load_and_render_gaussians(
    ply_path: str,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    image_width: int,
    image_height: int,
    sh_degree: int = 3,
    bg_color: Tuple[float, float, float] = (0, 0, 0),
    znear: float = 0.01,
    zfar: float = 100.0,
    scaling_modifier: float = 1.0,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Load Gaussian splats from PLY and render from a given camera viewpoint.
        
    Args:
        ply_path: Path to the PLY file containing Gaussian parameters
        K: 3x3 camera intrinsic matrix
        R: 3x3 rotation matrix (world to camera)
        T: 3x1 translation vector (world to camera)
        image_width: Output image width
        image_height: Output image height
        sh_degree: Spherical harmonics degree (default 3)
        bg_color: Background color as RGB tuple (0-1 range)
        znear: Near clipping plane
        zfar: Far clipping plane
        scaling_modifier: Scaling factor for Gaussians
        device: Device to use for rendering
    
    Returns:
        Rendered image as tensor (C, H, W)
    
    Example:
        >>> K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        >>> R = np.eye(3)  # Identity rotation
        >>> T = np.array([0, 0, 5])  # 5 units back
        >>> img = load_and_render_gaussians("model.ply", K, R, T, 1920, 1080)
    """
    try:
        # Create renderer objetc
        renderer = GaussianRenderer(
            sh_degree=sh_degree,
            device=device
        )
        
        # Load PLY file
        renderer.load_ply(ply_path)
        
        # Render image
        rendered_image = renderer.render(
            K=K,
            R=R,
            T=T,
            image_width=image_width,
            image_height=image_height,
            bg_color=bg_color,
            znear=znear,
            zfar=zfar,
            scaling_modifier=scaling_modifier
        )
        
        return rendered_image
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to render {ply_path}: {e}")
        raise

