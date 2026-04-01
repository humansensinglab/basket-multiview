import math
import numpy as np
import torch
from diff_gaussian_rasterization_accelerated import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from cs_renderers._utils import BaseRenderer
from utils.camera_utils import get_minicam

from scene_hs.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


class AccceleratedRenderer(BaseRenderer):

    def __init__(self, conf: dict):
        super().__init__(conf=conf)
        self.conf = conf

    def render(self, gaussian_model, pipeline_params, accel_params) -> dict:

        store_pre = accel_params["store"]
        precomp = accel_params["precomp"]
        ranges = accel_params["ranges"]
        gs_list = accel_params["gs_list"]
        n_rend = accel_params["n_rend"]
        n_buck = accel_params["n_buck"]
        use_cuda_graph = accel_params["use_cuda_graph"]
        minicam = accel_params["minicam"]
        background = accel_params["background"]
        use_load_balancing = accel_params["use_load_balancing"]

        if precomp:
            img_buffer = accel_params["render_buffers"]["img_buffer"]
            geom_buffer = accel_params["render_buffers"]["geom_buffer"]
            sample_buffer = accel_params["render_buffers"]["sample_buffer"]
            dL_ddc = accel_params["dL_ddc"]
            dL_dsh = accel_params["dL_dsh"]
            dL_dcolors = accel_params["dL_dcolors"]
            output_img = accel_params["render_output"]

            if use_load_balancing:
                tiles_id = accel_params["tiles_id"]
                num_tiles = accel_params["num_tiles"]
                pixel_tr = accel_params["pixel_tr"]
            else:
                tiles_id = torch.empty(0, dtype=torch.int8, device="cuda")
                num_tiles = 0
                pixel_tr = torch.empty(0, dtype=torch.int8, device="cuda")

        else:
            img_buffer = torch.empty(0, dtype=torch.int8, device="cuda")
            geom_buffer = torch.empty(0, dtype=torch.int8, device="cuda")
            sample_buffer = torch.empty(0, dtype=torch.int8, device="cuda")
            dL_ddc = torch.empty((0, 1, 3), device="cuda")
            dL_dsh = torch.empty((0, 1, 3), device="cuda")
            dL_dcolors = torch.empty((0, 3), device="cuda")
            output_img = accel_params["render_output"]
            tiles_id = torch.empty(0, dtype=torch.int8, device="cuda")
            num_tiles = 0
            pixel_tr = torch.empty(0, dtype=torch.int8, device="cuda")

        return self._render(
            minicam,
            gaussian_model,
            pipeline_params,
            background,
            store_pre,
            precomp,
            ranges,
            gs_list,
            n_buck,
            n_rend,
            use_cuda_graph,
            dL_ddc,
            dL_dsh,
            dL_dcolors,
            img_buffer,
            geom_buffer,
            sample_buffer,
            tiles_id,
            num_tiles,
            pixel_tr,
            output_img,
            1.0,
            None,
            use_load_balancing,
        )

    def get_background(self):
        if self.conf["background_mode"] == "white":
            background_color = [1, 1, 1]
        elif self.conf["background_mode"] == "black":
            background_color = [0, 0, 0]
        elif self.conf["background_mode"] == "random":
            background_color = np.random.rand(1, 3)

        return torch.tensor(background_color, dtype=torch.float32).cuda()

    def _render(
        self,
        viewpoint_camera,
        pc: GaussianModel,
        pipe,
        bg_color: torch.Tensor,
        store_ordering: bool,
        using_precomp: bool,
        ranges: torch.Tensor,
        gs_list: torch.Tensor,
        num_buck: int,
        num_rend: int,
        use_cuda_graph: bool,
        dL_ddc=None,
        dL_dsh=None,
        dL_dcolors=None,
        img_buffer=None,
        geom_buffer=None,
        sample_buffer=None,
        tiles_id=None,
        num_tiles=0,
        pixel_tr=None,
        output_img=None,
        scaling_modifier=1.0,
        override_color=None,
        use_load_balancing=None,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        screenspace_points = (
            torch.zeros_like(
                pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

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
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
            store_ordering=store_ordering,
            using_precomp=using_precomp,
            gaussian_list=gs_list,
            ranges=ranges,
            num_buckets=num_buck,
            num_rendered=num_rend,
            use_cuda_graph=use_cuda_graph,
            img_buffer=img_buffer,
            geom_buffer=geom_buffer,
            sample_buffer=sample_buffer,
            dl_dsh=dL_dsh,
            dl_ddc=dL_ddc,
            dl_dcolors=dL_dcolors,
            tiles_id=tiles_id,
            num_tiles=num_tiles,
            pixel_tr=pixel_tr,
            output_img=output_img,
            use_load_balancing=use_load_balancing,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        scales = None
        rotations = None
        cov3D_precomp = None

        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(
                    -1, 3, (pc.max_sh_degree + 1) ** 2
                )
                dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                    pc.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                dc, shs = pc._features_dc, pc._features_rest
            colors_precomp = override_color

        (
            rendered_image,
            radii,
            gaussian_list,
            ranges,
            _bucket,
            num_rendered,
            num_bucket,
            img_buffer_size,
            geom_buffer_size,
            sample_buffer_size,
            tiles_lb,
            num_tiles,
            pixel_tr,
        ) = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": 0 if using_precomp else (radii > 0).nonzero(),
            "radii": radii,
            "ranges": ranges,
            "gaussian_list": gaussian_list,
            "num_bucket": num_bucket,
            "num_rendered": num_rendered,
            "img_buffer": img_buffer_size,
            "geom_buffer": geom_buffer_size,
            "sample_buffer": sample_buffer_size,
            "tiles_id": tiles_lb,
            "num_tiles": num_tiles,
            "pixel_tr": pixel_tr,
        }
