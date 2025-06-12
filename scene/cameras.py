#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
from PIL import Image

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image=None, invdepthmap=None,
                 image_name="", uid=0,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 train_test_exp=False, is_test_dataset=False, is_test_view=False,
                 image_path=None, depth_path="", preload=True, is_nerf_synthetic=False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.image_path = image_path
        self.depth_path = depth_path
        self.preload = preload
        self.is_nerf_synthetic = is_nerf_synthetic
        self.resolution = resolution
        self.train_test_exp = train_test_exp
        self.is_test_dataset = is_test_dataset
        self.is_test_view = is_test_view
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.alpha_mask = None
        self.original_image = None
        self.image_width = resolution[0]
        self.image_height = resolution[1]
        self.invdepthmap = None
        self.depth_mask = None
        self.depth_reliable = False

        self.depth_params = depth_params

        if self.preload:
            self.load_data(image, invdepthmap)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load_data(self, image=None, invdepthmap=None):
        if image is None and self.image_path is not None:
            image = Image.open(self.image_path)

        if image is not None:
            resized_image_rgb = PILtoTorch(image, self.resolution)
            gt_image = resized_image_rgb[:3, ...]
            if resized_image_rgb.shape[0] == 4:
                self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
            else:
                self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

            if self.train_test_exp and self.is_test_view:
                if self.is_test_dataset:
                    self.alpha_mask[..., : self.alpha_mask.shape[-1] // 2] = 0
                else:
                    self.alpha_mask[..., self.alpha_mask.shape[-1] // 2 :] = 0

            self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

        if invdepthmap is None and self.depth_path:
            try:
                if self.is_nerf_synthetic:
                    invdepthmap = cv2.imread(self.depth_path, -1).astype(np.float32) / 512
                else:
                    invdepthmap = cv2.imread(self.depth_path, -1).astype(np.float32) / float(2 ** 16)
            except Exception:
                invdepthmap = None

        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            invdepthmap = cv2.resize(invdepthmap, self.resolution)
            invdepthmap[invdepthmap < 0] = 0
            self.depth_reliable = True

            if self.depth_params is not None:
                if self.depth_params["scale"] < 0.2 * self.depth_params["med_scale"] or self.depth_params["scale"] > 5 * self.depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0

                if self.depth_params["scale"] > 0:
                    invdepthmap = invdepthmap * self.depth_params["scale"] + self.depth_params["offset"]

            if invdepthmap.ndim != 2:
                invdepthmap = invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(invdepthmap[None]).to(self.data_device)

    def release_data(self):
        self.original_image = None
        self.alpha_mask = None
        self.invdepthmap = None
        self.depth_mask = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

