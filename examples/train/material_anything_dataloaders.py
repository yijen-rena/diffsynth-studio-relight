# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import glob
import json
import imageio.v2 as imageio

import torch
import numpy as np
import cv2

import util

from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
###############################################################################
# NERF image based dataset (synthetic)
###############################################################################

def _load_img(path):
    files = glob.glob(path + '.*')
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    img = util.load_image_raw(files[0])
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        # img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

def generate_directional_embeddings(shape=(32, 64)):
    """
    Generate directional embeddings for an environment map based on the given image resolution.
    
    Args:
    shape (tuple): A tuple of two integers (height, width) representing the resolution of the image.
    
    Returns:
    numpy.ndarray: A (height, width, 3) array where each (u, v) pixel contains the corresponding 
                   directional vector (x, y, z).
    """
    height, width = shape

    # Generate a grid of u and v values representing pixel indices
    u = np.linspace(0, 1, width, endpoint=False)
    v = np.linspace(0, 1, height, endpoint=False)

    # Create 2D grids for u and v using np.meshgrid
    U, V = np.meshgrid(u, v)

    # Compute the spherical angles (θ and φ) for all pixels at once
    theta = np.pi * V  # θ goes from 0 to π (polar angle)
    phi = 2 * np.pi * U  # φ goes from 0 to 2π (azimuth angle)

    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = -np.sin(theta) * np.sin(phi)
    z = -np.cos(theta)

    # Stack the x, y, and z arrays along a new dimension to create the directional embeddings
    embeddings = np.stack((x, y, z), axis=-1)

    return embeddings

def generate_plucker_rays(T, shape, fov, sensor_size=(1.0, 1.0)):
    """
    Generate Plücker rays for each pixel of the image based on a camera transformation matrix.

    Args:
    T (numpy.ndarray): A 4x4 transformation matrix representing the camera pose.
    H (int): Height of the image.
    W (int): Width of the image.
    focal_length_x (float): The focal length of the camera in the X direction.
    focal_length_y (float): The focal length of the camera in the Y direction.
    sensor_size (tuple): Physical size of the sensor in world units (width, height).

    Returns:
    numpy.ndarray: A (6, H, W) array where the first 3 elements are the direction vector (d)
                   and the last 3 are the moment vector (m) for each pixel.
    """
    # Extract the rotation matrix (3x3) and translation vector (3x1) from the transformation matrix
    R = T[:3, :3]  # Rotation part
    t = T[:3, 3]   # Translation part (camera position in world space)
    H, W = shape
    H //= 8
    W //= 8
    focal_length_x, focal_length_y = fov

    # Generate pixel grid in image space
    i = np.linspace(-sensor_size[1] / 2, sensor_size[1] / 2, H)  # Y coordinates
    j = np.linspace(-sensor_size[0] / 2, sensor_size[0] / 2, W)  # X coordinates

    # Create 2D meshgrid for pixel centers
    J, I = np.meshgrid(j, i)

    # Compute normalized camera rays (camera space, assuming pinhole camera model)
    rays_d_cam = np.stack([
        J / focal_length_x,  # Scale by focal length in X
        I / focal_length_y,  # Scale by focal length in Y
        np.full_like(J, 1.0)  # Z is fixed to 1.0 for normalized camera rays
    ], axis=-1)  # Shape: (H, W, 3)

    # Normalize ray directions
    rays_d_cam /= np.linalg.norm(rays_d_cam, axis=-1, keepdims=True)  # Normalize to unit vectors

    # Transform ray directions to world space using the rotation matrix
    rays_d_world = -np.matmul(R, rays_d_cam.reshape(-1, 3).T).T.reshape(H, W, 3)  # Shape: (H, W, 3)

    # Moment vector for each pixel is computed as t x d (cross product of translation and direction)
    rays_m_world = np.cross(t, rays_d_world, axisa=0, axisb=2)  # Shape: (H, W, 3)

    # Combine direction vectors and moment vectors into a single array of shape (6, H, W)
    plucker_rays = np.stack([
        rays_d_world[..., 0], rays_d_world[..., 1], rays_d_world[..., 2],
        rays_m_world[..., 0], rays_m_world[..., 1], rays_m_world[..., 2]
    ], axis=0)
    # origins = np.tile(t[:, np.newaxis, np.newaxis], (1, H, W))  # Shape: (3, H, W)
    return plucker_rays#, origins

def env_map_to_cam_to_world_by_convention(envmap: np.ndarray, c2w, convention):
    R = c2w[:3,:3]
    H, W = envmap.shape[:2]
    theta, phi = np.meshgrid(np.linspace(-0.5*np.pi, 1.5*np.pi, W), np.linspace(0., np.pi, H))
    viewdirs = np.stack([-np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)],
                           axis=-1).reshape(H*W, 3)    # [H, W, 3]
    viewdirs = (R.T @ viewdirs.T).T.reshape(H, W, 3)
    viewdirs = viewdirs.reshape(H, W, 3)
    # This is correspond to the convention of +Z at left, +Y at top
    # -np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)
    coord_y = ((np.arccos(viewdirs[..., 1].clip(-1, 1))/np.pi*(H-1)+H)%H).astype(np.float32)
    coord_x = (((np.arctan2(viewdirs[...,0], -viewdirs[...,2])+np.pi)/2/np.pi*(W-1)+W)%W).astype(np.float32)
    envmap_remapped = cv2.remap(envmap, coord_x, coord_y, cv2.INTER_LINEAR)
    if convention == 'ours':
        return envmap_remapped
    if convention == 'physg':
        # change convention from ours (Left +Z, Up +Y) to physg (Left -Z, Up +Y)
        envmap_remapped_physg = np.roll(envmap_remapped, W//2, axis=1)
        return envmap_remapped_physg
    if convention == 'nerd':
        # change convention from ours (Left +Z-X, Up +Y) to nerd (Left +Z+X, Up +Y)
        envmap_remapped_nerd = envmap_remapped[:,::-1,:]
        return envmap_remapped_nerd

    assert convention == 'invrender', convention
    # change convention from ours (Left +Z-X, Up +Y) to invrender (Left -X+Y, Up +Z)
    theta, phi = np.meshgrid(np.linspace(1.0 * np.pi, -1.0 * np.pi, W), np.linspace(0., np.pi, H))
    viewdirs = np.stack([np.cos(theta) * np.sin(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(phi)], axis=-1)    # [H, W, 3]
    # viewdirs = np.stack([-viewdirs[...,0], viewdirs[...,2], viewdirs[...,1]], axis=-1)
    coord_y = ((np.arccos(viewdirs[..., 1])/np.pi*(H-1)+H)%H).astype(np.float32)
    coord_x = (((np.arctan2(viewdirs[...,0], -viewdirs[...,2])+np.pi)/2/np.pi*(W-1)+W)%W).astype(np.float32)
    envmap_remapped_Inv = cv2.remap(envmap_remapped, coord_x, coord_y, cv2.INTER_LINEAR)
    return envmap_remapped_Inv

def preceptual_quantizer(x):
    x = np.clip(x, 0, 10000)
    m1 = 0.1593017578125
    c1 = 0.8359375
    m2 = 78.84375
    c2 = 18.8515625
    c3 = 18.6875
    return ((c1 + c2 * x ** m1) / (1 + c3 * x ** m1)) ** m2

def hlg_oetf(x):
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    return np.where(x <= 1/12, np.sqrt(3 * x), a * np.log((12*x - b).clip(1e-5, np.inf)) + c)

def rotate_image_fast(image, degrees):
    H, W, C = image.shape
    degrees = degrees % 360  # Ensure degrees is within 360
    pixel_loc = round(degrees * W / 360)  # Calculate the pixel location to rotate
    image = np.hstack((image[:, -pixel_loc:, :], image[:, :-pixel_loc, :]))
    return image

class DatasetNERFMultiRelight(Dataset):
    def __init__(self, cfg_path, args, transform, env_transform, examples=None):
        self.examples = examples
        self.base_dir = os.path.dirname(cfg_path)
        # self.limiter = 10 if "train" in cfg_path else 3
        self.args = args
        if "transforms_test" in cfg_path:
            self.is_test = True
        else:
            self.is_test = False
        # Load config / transforms
        self.cfg = json.load(open(cfg_path, 'r'))
        self.n_objs = len(self.cfg['frames'])
        self.n_envmaps = len(self.cfg['frames'][0])
        self.n_views = len(self.cfg['frames'][0][0]["views"])
        self.k_views = 4 #how many views to use
        if self.k_views > self.n_views:
            self.k_views = self.n_views

        # Determine resolution & aspect ratio
        self.resolution = imageio.imread(os.path.join(self.base_dir, self.cfg["frames"][0][0]["views"][0]['file_path'] + ".png")).shape[0:2]
        self.aspect = self.resolution[1] / self.resolution[0]

        self.transform = transform
        self.env_transform = env_transform
        print("DatasetNERFMultiRelight: loading %d objects with shape [%d, %d]" % (self.n_objs, self.resolution[0], self.resolution[1]))
        self.fovx   = self.cfg['camera_angle_x']
        self.fovy   = util.fovx_to_fovy(self.fovx, self.aspect)
        self.proj   = util.perspective(self.fovy, self.aspect, 0.1, 1000.0)

        # self.dir_embeds = torch.tensor(generate_directional_embeddings(imageio.imread(os.path.join(self.args.envmap_path, self.cfg['frames'][0][0]['env'])).shape[:2]), dtype=torch.float32).permute(2, 0, 1)
        self.dir_embeds = torch.tensor(generate_directional_embeddings(), dtype=torch.float32).permute(2, 0, 1)


    def _parse_frame(self, cfg, idx, cam_near_far=[0.1, 1000.0]):
        proj   = self.proj
        #TODO: make this generalizable to take in parameter of k views
        imgs, albedos, orms, masks, mvps, envs, envs_darker, envs_brighter, dir_embeds, pluckers = [], [], [], [], [], [], [], [], [], []

        if self.is_test:
            env_idxs = [0, 1]
        else:
            # env_idxs = np.random.choice(self.n_envmaps, 2, replace=False)
            env_idxs = np.random.choice(len(self.cfg['frames'][idx]), 2, replace=False)
        # print("env_idxs", env_idxs, idx, self.is_test, self.n_envmaps)
        view_idxs = np.random.choice(len(cfg['frames'][idx][0]['views']), self.k_views, replace=False)
        for i_idx, env_idx in enumerate(env_idxs):
            obj_views = cfg['frames'][idx][env_idx]['views']
            env_darker, env_brighter = None, None
            for v_idx, view_idx in enumerate(view_idxs):

                view = obj_views[view_idx]
                img    = _load_img(os.path.join(self.base_dir, view['file_path'])).permute(2, 0, 1)
                albedo = _load_img(os.path.join(self.base_dir, view['file_path_albedo'])).permute(2, 0, 1)
                orm    = _load_img(os.path.join(self.base_dir, view['file_path_orm'])).permute(2, 0, 1)
                frame_transform = torch.tensor(view['transform_matrix'], dtype=torch.float32)
                mv = torch.linalg.inv(frame_transform)
                mv = mv @ util.rotate_x(-np.pi / 2)

                try:
                    if "scale" in cfg['frames'][idx][env_idx]:
                        env = imageio.imread(os.path.join(self.args.envmap_path, cfg['frames'][idx][env_idx]['env']))[..., :3] * cfg['frames'][idx][env_idx]["scale"]
                    elif ".exr" in cfg['frames'][idx][env_idx]["env"]: #laval hack
                        env = imageio.imread(os.path.join(self.args.envmap_path, cfg['frames'][idx][env_idx]['env']))[..., :3] * 150
                    else:
                        env = imageio.imread(os.path.join(self.args.envmap_path, cfg['frames'][idx][env_idx]['env']))[..., :3]
                except Exception as e:
                    print(e)
                    env = imageio.imread(os.path.join("/ocean/projects/cis240022p/ylitman/VSCode/RelightingDiffusion/data/fullres_light_probes", cfg['frames'][idx][env_idx]['env']))[..., :3]
                    

                # print("init", env.min(), env.max())
                # env_norm = (env - np.min(env)) / (np.max(env) - np.min(env))
                if "rotation" in cfg['frames'][idx][env_idx]:
                    if cfg['frames'][idx][env_idx]["rotation"] > 0:
                        env = rotate_image_fast(env, cfg['frames'][idx][env_idx]["rotation"])
                if "flip" in cfg['frames'][idx][env_idx]:
                    if cfg['frames'][idx][env_idx]["flip"]:
                        env = env[..., ::-1, :]
                if self.args.transform_envmap:
                    env = env_map_to_cam_to_world_by_convention(env, torch.linalg.inv(mv).numpy(), 'ours')
                env_darker = (np.log10(env + 1) / np.log10(env.max())).clip(0, 1)
                env_brighter = hlg_oetf(env_darker).clip(0, 1)

                env_darker = self.env_transform(env_darker)
                env_brighter = self.env_transform(env_brighter)
                # d_embeds = torch.tensor(generate_directional_embeddings(env.shape[:2]))
                # mask = orm[..., 3:]
                # img = img[..., :3] * mask
                # albedo = albedo[..., :3] * mask
                # orm = orm[..., :3] * mask
                mask = img[3:, ...]
                img = img[:3, ...] * mask
                albedo = albedo[:3, ...] * mask
                orm = orm[:3, ...] * mask


                img = self.transform(img)
                albedo = self.transform(albedo)
                orm = self.transform(orm)
                mask = transforms.Resize((self.args.resolution, self.args.resolution), antialias=True)(mask)
                # mask = transforms.Resize((self.args.resolution, self.args.resolution), antialias=True)(F.to_tensor(mask))
                frame_pluckers = torch.tensor(generate_plucker_rays(frame_transform, img.shape[1:3], [self.fovx, self.fovy]))

                imgs.append(img)
                albedos.append(albedo)
                orms.append(orm)
                masks.append(mask)
                # envs.append(env)
                envs_darker.append(env_darker)
                envs_brighter.append(env_brighter)
                dir_embeds.append(self.dir_embeds)

                pluckers.append(frame_pluckers)

                mvp = proj @ mv
                # t = mvp[:3, 3]
                t = frame_transform[:3, 3]
                r = torch.linalg.norm(t)
                theta = torch.arccos(t[2] / r)
                # Compute azimuth angle (ϕ)
                phi = torch.arctan2(t[1], t[0])
                mvp = torch.tensor([theta, torch.sin(phi), torch.cos(phi), r])
                mvps.append(mvp)

        # [2, b, v, c, h, w]
        imgs = torch.stack(imgs) 
        albedos = torch.stack(albedos)
        orms = torch.stack(orms)
        masks = torch.stack(masks)
        mvps = torch.stack(mvps)
        # envs = torch.stack(envs)
        envs_darker = torch.stack(envs_darker)
        envs_brighter = torch.stack(envs_brighter)
        dir_embeds = torch.stack(dir_embeds)
        pluckers = torch.stack(pluckers)
        imgs = imgs.reshape(2, self.k_views, *imgs.shape[1:])
        albedos = albedos.reshape(2, self.k_views, *albedos.shape[1:])
        orms = orms.reshape(2, self.k_views, *orms.shape[1:])
        masks = masks.reshape(2, self.k_views, *masks.shape[1:])
        mvps = mvps.reshape(2, self.k_views, *mvps.shape[1:])
        # envs = envs.reshape(2, self.k_views, *envs.shape[1:])
        envs_darker = envs_darker.reshape(2, self.k_views, *envs_darker.shape[1:])
        envs_brighter = envs_brighter.reshape(2, self.k_views, *envs_brighter.shape[1:])
        dir_embeds = dir_embeds.reshape(2, self.k_views, *dir_embeds.shape[1:])
        pluckers = pluckers.reshape(2, self.k_views, *pluckers.shape[1:])
        return imgs, albedos, orms, masks, mvps, envs_darker, envs_brighter, dir_embeds, pluckers

    def getMesh(self):
        return None

    def __len__(self):
        return self.n_objs if self.examples is None else self.examples
        # return int(self.n_objs*self.n_views*self.n_envmaps/self.k_views) if self.examples is None else self.examples

    def __getitem__(self, itr):
        # img      = []
        img, albedo, orm, mask, mvp, envs_darker, envs_brighter, dir_embeds, pluckers = self._parse_frame(self.cfg, itr % self.n_objs)

        return {
            # 'mv' : mv,
            'T' : mvp,
            # 'campos' : campos,
            'resolution' : [self.args.resolution, self.args.resolution],
            'spp' : 16,
            'img' : img,
            'albedo' : albedo,
            'orm' : orm,
            'mask' : mask,
            'dir_embeds' : dir_embeds,
            'pluckers' : pluckers,
            # 'envs' : envs
            'envs_darker' : envs_darker,
            'envs_brighter' : envs_brighter,
            # 'envmap' : envmap
        }


    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']

        src_batch = {
            'image': torch.cat([b['img'][0] for b in batch]),       # Even elements
            'albedo': torch.cat([b['albedo'][0] for b in batch]),
            'orm': torch.cat([b['orm'][0] for b in batch]),
            'mask': torch.cat([b['mask'][0] for b in batch]),
            # 'envs': torch.cat([b['envs'][0] for b in batch]),
            'envs_darker': torch.cat([b['envs_darker'][0] for b in batch]),
            'envs_brighter': torch.cat([b['envs_brighter'][0] for b in batch]),
            'dir_embeds': torch.cat([b['dir_embeds'][0] for b in batch]),
            'pluckers': torch.cat([b['pluckers'][0] for b in batch]),
            'T': torch.cat([b['T'][0] for b in batch]),
        }

        target_batch = {
            'image': torch.cat([b['img'][1] for b in batch]),       # Odd elements
            'albedo': torch.cat([b['albedo'][1] for b in batch]),
            'orm': torch.cat([b['orm'][1] for b in batch]),
            'mask': torch.cat([b['mask'][1] for b in batch]),
            # 'envs': torch.cat([b['envs'][1] for b in batch]),
            'envs_darker': torch.cat([b['envs_darker'][1] for b in batch]),
            'envs_brighter': torch.cat([b['envs_brighter'][1] for b in batch]),
            'dir_embeds': torch.cat([b['dir_embeds'][1] for b in batch]),
            'pluckers': torch.cat([b['pluckers'][1] for b in batch]),
            'T': torch.cat([b['T'][1] for b in batch]),
        }

        out_batch = {
            'src_batch': src_batch,
            'target_batch': target_batch,
            'resolution': iter_res,
            'spp': iter_spp,
        }

        return out_batch
