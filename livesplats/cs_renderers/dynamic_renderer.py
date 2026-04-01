import math
import numpy as np
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer)

from cs_renderers._utils import BaseRenderer
from utils.camera_utils import get_minicam
 
from scene_hs.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

class DynamicRenderer(BaseRenderer):
    
    def __init__(self, conf:dict):
        super().__init__(conf=conf)
        self.conf = conf
        
    def render(self, sample:dict, gaussian_model, pipeline_params, view_index=None, full_bwmatrix=None, full_bwmatrix2=None, pixelParity=None, bg_color=None) ->dict:
        
        minicam = get_minicam(
            width=sample['cam_info']['width'], 
            height=sample['cam_info']['height'], 
            K=sample['cam_info']['K'], 
            w2c=sample['cam_info']['w2c'],
        )
        
        if bg_color is None:
            bg_color = self.get_background()

        rendered_im_out = self._render(
            viewpoint_camera=minicam,
            gaussians_model=gaussian_model,
            pipeline_params=pipeline_params,
            bg_color=bg_color)
        
        return rendered_im_out

    def get_background(self):
        
        if self.conf['background_mode'] == 'white':
            background_color = np.array([[1., 1., 1.]])
        elif self.conf['background_mode'] == 'black':
            background_color = np.array([[0., 0., 0.]])
        elif self.conf['background_mode'] == 'random':
            background_color = np.random.rand(1,3) 
        
        background = torch.tensor(
            background_color, 
            dtype=torch.float32).cuda()

        return background
 
    def _render(self,
        viewpoint_camera, 
        gaussians_model : GaussianModel,
        pipeline_params, 
        bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        override_color = None):
        
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        """
    
        # Create zero tensor. We will use it to make pytorch return gradients 
        # of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(
            gaussians_model.get_xyz, 
            dtype=gaussians_model.get_xyz.dtype, 
            requires_grad=True, 
            device="cuda") + 0
        
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=gaussians_model.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipeline_params.debug,
            antialiasing=pipeline_params.antialiasing
        )

        means3D = gaussians_model.get_xyz
        means2D = screenspace_points
        opacity = gaussians_model.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will 
        # be computed from scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipeline_params.compute_cov3D_python:
            cov3D_precomp = gaussians_model.get_covariance(scaling_modifier)
        else:
            scales = gaussians_model.get_scaling
            rotations = gaussians_model.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired 
        # to precompute colors from SHs in Python, do it. If not, then SH -> 
        # -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipeline_params.convert_SHs_python:
                
                shs_view = gaussians_model.get_features.transpose(1, 2).view(
                    -1, 3, (gaussians_model.max_sh_degree+1)**2)
                
                dir_pp = (gaussians_model.get_xyz - viewpoint_camera.camera_center.repeat(
                    gaussians_model.get_features.shape[0], 1))
                
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(gaussians_model.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                
            else:
                # dc, shs = gaussians_model.get_features_dc, gaussians_model.get_features_rest
                shs = gaussians_model.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        rendered_image, radii, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            # dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        ret_val = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "cov3D": cov3D_precomp
            }
    
        return ret_val