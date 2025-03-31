import numpy as np
import time
import torch
import cv2
import torch.nn.functional as F
from PIL import Image
import os
from pathlib import Path
import argparse
from glob import glob
from tqdm import tqdm
import traceback
import torchvision
from torchvision import transforms
import random
import shutil
import json

from torch.utils.data import Dataset
from kornia import create_meshgrid
import imageio
import math

import pdb

ENV_MAP_PATH = "/ocean/projects/cis240022p/ylitman/VSCode/RelightingDiffusion/data/fullres_light_probes"

def fovx_to_fovy(fov_x, aspect_ratio):
    fov_x_rad = math.radians(fov_x)
    fov_y_rad = 2 * math.atan(math.tan(fov_x_rad / 2) * aspect_ratio)
    fov_y = math.degrees(fov_y_rad)
    return fov_y

def perspective(fovy, aspect, near, far):
    fovy_rad = math.radians(fovy)
    f = 1.0 / math.tan(fovy_rad / 2.0)
    result = np.zeros((4, 4), dtype=np.float32)
    result[0, 0] = f / aspect
    result[1, 1] = f
    result[2, 2] = (far + near) / (near - far)
    result[2, 3] = (2.0 * far * near) / (near - far)
    result[3, 2] = -1.0
    result = torch.from_numpy(result)
    return result

def rotate_x(angle):
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle, dtype=torch.float32)
    
    device = angle.device

    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    matrix = torch.eye(4, dtype=torch.float32, device=device)
    
    matrix[1, 1] = cos_theta
    matrix[1, 2] = -sin_theta
    matrix[2, 1] = sin_theta
    matrix[2, 2] = cos_theta
    
    return matrix

def _load_img(path):
    files = glob(path + '.*')
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    img = imageio.imread(files[0])
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        # img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

def rotate_image_fast(image, degrees):
    H, W, C = image.shape
    degrees = degrees % 360  # Ensure degrees is within 360
    pixel_loc = round(degrees * W / 360)  # Calculate the pixel location to rotate
    image = np.hstack((image[:, -pixel_loc:, :], image[:, :-pixel_loc, :]))
    return image

def generate_plucker_rays(T, shape, fov = (35, 35), sensor_size=(1.0, 1.0)):
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
    return plucker_rays

def generate_directional_embeddings(shape, world2cam=None, normalize=True):
    height, width = shape

    u = np.linspace(0, 1, width, endpoint=False)
    v = np.linspace(0, 1, height, endpoint=False)
    U, V = np.meshgrid(u, v)

    theta = np.pi * V 
    phi = 2 * np.pi * U 

    x = np.sin(theta) * np.cos(phi)
    y = -np.sin(theta) * np.sin(phi)
    z = -np.cos(theta)

    embeddings = np.stack((x, y, z), axis=-1)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)

    if normalize:
        embeddings = (embeddings + 1) / 2

    if world2cam is not None:
        R = world2cam[:3, :3]
        embeddings_flat = embeddings.reshape(-1, 3)
        embeddings_flat = embeddings_flat @ R.T
        embeddings = embeddings_flat.reshape(height, width, 3)

    return embeddings

def env_map_to_cam_to_world_by_convention(envmap: np.ndarray, c2w):
    import cv2
    R = c2w[:3,:3]
    H, W = envmap.shape[:2]
    theta, phi = np.meshgrid(np.linspace(-0.5*np.pi, 1.5*np.pi, W), np.linspace(0., np.pi, H))
    viewdirs = np.stack([-np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)],
                        axis=-1).reshape(H*W, 3)    # [H, W, 3]
    viewdirs = (R.T @ viewdirs.T).T.reshape(H, W, 3)
    viewdirs = viewdirs.reshape(H, W, 3)
    # This corresponds to the convention of +Z at left, +Y at top
    # -np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)
    coord_y = ((np.arccos(viewdirs[..., 1].clip(-1, 1))/np.pi*(H-1)+H)%H).astype(np.float32)
    coord_x = (((np.arctan2(viewdirs[...,0], -viewdirs[...,2])+np.pi)/2/np.pi*(W-1)+W)%W).astype(np.float32)
    envmap_remapped = cv2.remap(envmap, coord_x, coord_y, cv2.INTER_LINEAR)

    return envmap_remapped

##### Below are functions for preprocessing environment map, copied from Neural Gaffer #####

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
def read_hdr(path, return_type='np'):
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    if not path.endswith('.png'):
        path = path + '.png'
        
    try:
        with open(path, 'rb') as h:
            buffer_ = np.frombuffer(h.read(), np.uint8)
        bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(f"Error reading HDR file {path}: {e}")
        return None
    
    if return_type == 'np':
        return rgb
    elif return_type == 'torch':
        return torch.from_numpy(rgb)
    else:
        raise ValueError(f"Invalid return type: {return_type}")

def generate_envir_map_dir(envmap_h, envmap_w):
    lat_step_size = np.pi / envmap_h
    lng_step_size = 2 * np.pi / envmap_w
    theta, phi = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, envmap_h), 
                                torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, envmap_w)], indexing='ij')

    sin_theta = torch.sin(torch.pi / 2 - theta)  # [envH, envW]
    light_area_weight = 4 * torch.pi * sin_theta / torch.sum(sin_theta)  # [envH, envW]
    assert 0 not in light_area_weight, "There shouldn't be light pixel that doesn't contribute"
    light_area_weight = light_area_weight.to(torch.float32).reshape(-1) # [envH * envW, ]


    view_dirs = torch.stack([   torch.cos(phi) * torch.cos(theta), 
                                torch.sin(phi) * torch.cos(theta), 
                                torch.sin(theta)], dim=-1).view(-1, 3)    # [envH * envW, 3]
    light_area_weight = light_area_weight.reshape(envmap_h, envmap_w)
    
    return light_area_weight, view_dirs

def get_light(hdr_rgb, incident_dir, hdr_weight=None):

    envir_map = hdr_rgb
    envir_map = envir_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
    if torch.isnan(envir_map).any():
        os.system('echo "nan in envir_map"')
    if hdr_weight is not None:
        hdr_weight = hdr_weight.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
    incident_dir = incident_dir.clip(-1, 1)
    theta = torch.arccos(incident_dir[:, 2]).reshape(-1) - 1e-6 # top to bottom: 0 to pi
    phi = torch.atan2(incident_dir[:, 1], incident_dir[:, 0]).reshape(-1) # left to right: pi to -pi

    #  x = -1, y = -1 is the left-top pixel of F.grid_sample's input
    query_y = (theta / np.pi) * 2 - 1 # top to bottom: -1-> 1
    query_y = query_y.clip(-1, 1)
    query_x = - phi / np.pi # left to right: -1 -> 1
    query_x = query_x.clip(-1, 1)
    grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0).float() # [1, 1, 2, N]
    if abs(grid.max()) > 1 or abs(grid.min()) > 1:
        os.system('echo "grid out of range"')
    
    light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)

    if torch.isnan(light_rgbs).any():
        os.system('echo "nan in light_rgbs"')
    return light_rgbs    


def process_im(im):
    im = im.convert("RGB")
    image_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256), antialias=True),  # 256, 256
            transforms.ToTensor(), # for PIL to Tensor [0, 255] -> [0.0, 1.0] and H×W×C-> C×H×W
            transforms.Normalize([0.5], [0.5]) # x -> (x - 0.5) / 0.5 == 2 * x - 1.0; [0.0, 1.0] -> [-1.0, 1.0]
        ]
    )
    return image_transforms(im)

def get_aligned_RT(cam2world):
    world2cam = np.linalg.inv(cam2world)
    aligned_RT = world2cam[:3, :]
    return aligned_RT


def reinhard_tonemap(hdr_image):
    """
    Basic Reinhard global operator.
    """
    # Convert to luminance (perceived brightness)
    luminance = 0.2126 * hdr_image[...,0] + \
                0.7152 * hdr_image[...,1] + \
                0.0722 * hdr_image[...,2]
    
    # Apply tone mapping to luminance
    L_mapped = luminance / (1 + luminance)
    
    # Preserve color ratios
    result = np.zeros_like(hdr_image)
    for i in range(3):
        result[...,i] = (hdr_image[...,i] / (luminance + 1e-6)) * L_mapped
    
    return np.clip(result, 0, 1)

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5 # 1xHxWx2

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)  # (H, W, 3)

    return directions

def get_ray_d(input_RT):
    sensor_width = 32

    # Get camera focal length
    focal_length = 35

    # Get image resolution
    resolution_x = 256

    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Compute focal length in pixels
    focal_length_px_x = focal_length * resolution_x / sensor_width

    focal = focal_length_px_x
    
    directions = get_ray_directions(resolution_x, resolution_x, [focal, focal])  # [H, W, 3]
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    
    w2c = input_RT
    w2c = np.vstack([w2c, [0, 0, 0, 1]])  # [4, 4]
    c2w = np.linalg.inv(w2c)
    pose = c2w @ blender2opencv
    c2w = torch.FloatTensor(pose)  # [4, 4]
    w2c = torch.linalg.inv(c2w)  # [4, 4]
    # Read ray data
    _, rays_d = get_rays(directions, c2w)

    return rays_d

def get_envir_map_light(envir_map, incident_dir):

    envir_map = envir_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
    phi = torch.arccos(incident_dir[:, 2]).reshape(-1) - 1e-6
    theta = torch.atan2(incident_dir[:, 1], incident_dir[:, 0]).reshape(-1)
    # normalize to [-1, 1]
    query_y = (phi / np.pi) * 2 - 1
    query_x = - theta / np.pi
    grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
    light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)

    return light_rgbs

def rotate_and_preprocess_envir_map(envir_map, aligned_RT, rotation_idx=0, total_view=120, visualize=False, output_dir=None):
    # envir_map: [H, W, 3]
    # aligned_RT: numpy.narray [3, 4] w2c
    # the coordinate system follows Blender's convention
    
    # c_x_axis, c_y_axis, c_z_axis = aligned_RT[0, :3], aligned_RT[1, :3], aligned_RT[2, :3]
    env_h, env_w = envir_map.shape[0], envir_map.shape[1]
 
    light_area_weight, view_dirs = generate_envir_map_dir(env_h, env_w)
    
    axis_aligned_transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) # Blender's convention
    axis_aligned_R = axis_aligned_transform @ aligned_RT[:3, :3] # [3, 3]
    view_dirs_world = view_dirs @ axis_aligned_R # [envH * envW, 3]
    
    # rotate the envir map along the z-axis
    rotated_z_radius = (-2 * np.pi * rotation_idx / total_view) 
    # [3, 3], left multiplied by the view_dirs_world
    rotation_maxtrix = np.array([[np.cos(rotated_z_radius), -np.sin(rotated_z_radius), 0],
                                [np.sin(rotated_z_radius), np.cos(rotated_z_radius), 0],
                                [0, 0, 1]])
    view_dirs_world = view_dirs_world @ rotation_maxtrix        
    
    rotated_hdr_rgb = get_light(envir_map, view_dirs_world)
    rotated_hdr_rgb = rotated_hdr_rgb.reshape(env_h, env_w, 3)
    
    rotated_hdr_rgb = np.array(rotated_hdr_rgb, dtype=np.float32)

    # ldr
    # envir_map_ldr = rotated_hdr_rgb.clip(0, 1)
    # envir_map_ldr = envir_map_ldr ** (1/2.2)
    envir_map_ldr = reinhard_tonemap(rotated_hdr_rgb)
    
    # hdr
    # envir_map_hdr = np.log1p(10 * rotated_hdr_rgb)
    
    # log
    envir_map_log = np.log(rotated_hdr_rgb + 1) / np.max(rotated_hdr_rgb)

    # dir
    envir_map_dir = generate_directional_embeddings((env_h, env_w), world2cam=aligned_RT)

    if visualize:
        envir_map_ldr_viz = np.uint8(envir_map_ldr * 255)
        envir_map_ldr_viz = Image.fromarray(envir_map_ldr_viz)
        envir_map_ldr_viz.save(os.path.join(output_dir, f"envir_map_ldr.png"))
        
        envir_map_log_viz = np.uint8(envir_map_log * 255)
        envir_map_log_viz = Image.fromarray(envir_map_log_viz)
        envir_map_log_viz.save(os.path.join(output_dir, f"envir_map_log.png"))
        
        envir_map_dir_viz = np.uint8(envir_map_dir * 255)
        envir_map_dir_viz = Image.fromarray(envir_map_dir_viz)
        envir_map_dir_viz.save(os.path.join(output_dir, f"envir_map_dir.png"))

    
    envir_map_ldr = torch.from_numpy(envir_map_ldr).permute(2, 0, 1)
    envir_map_log = torch.from_numpy(envir_map_log).permute(2, 0, 1)
    envir_map_dir = torch.from_numpy(envir_map_dir).permute(2, 0, 1)

    return envir_map_ldr, envir_map_log, envir_map_dir

def visualize_rotated_envir_map(envir_map_path, camera_pose_path, output_dir, cam2world=None):
    os.makedirs(output_dir, exist_ok=True)
    envir_map_name = os.path.splitext(os.path.basename(envir_map_path))[0]
    
    envir_map_hdr = read_hdr(envir_map_path)
    
    with open(camera_pose_path, 'r') as f:
        camera_poses = json.load(f)
    cam2world = np.array(camera_poses['frame_0'])
    
    envir_map_remapped = env_map_to_cam_to_world_by_convention(envir_map_hdr, cam2world)
    envir_map_ldr = reinhard_tonemap(envir_map_remapped)
    envir_map_ldr_img = Image.fromarray((envir_map_ldr * 255).astype(np.uint8)).convert('RGB')
    
    envir_map_ldr_img.save(os.path.join(output_dir, f"{envir_map_name}_ldr.png"))


def _clip_0to1_warn_torch(tensor_0to1):
    """Enforces [0, 1] on a tensor/array that should be already [0, 1].
    """
    msg = "Some values outside [0, 1], so clipping happened"
    if isinstance(tensor_0to1, torch.Tensor):
        if torch.min(tensor_0to1) < 0 or torch.max(tensor_0to1) > 1:
            tensor_0to1 = torch.clamp(
                tensor_0to1, min=0, max=1)
    elif isinstance(tensor_0to1, np.ndarray):
        if tensor_0to1.min() < 0 or tensor_0to1.max() > 1:
            tensor_0to1 = np.clip(tensor_0to1, 0, 1)
    else:
        raise NotImplementedError(f'Do not support dtype {type(tensor_0to1)}')
    return tensor_0to1

def linear2srgb_torch(tensor_0to1):
    if isinstance(tensor_0to1, torch.Tensor):
        pow_func = torch.pow
        where_func = torch.where
    elif isinstance(tensor_0to1, np.ndarray):
        pow_func = np.power
        where_func = np.where
    else:
        raise NotImplementedError(f'Do not support dtype {type(tensor_0to1)}')

    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_0to1 = _clip_0to1_warn_torch(tensor_0to1)

    tensor_linear = tensor_0to1 * srgb_linear_coeff
    
    tensor_nonlinear = srgb_exponential_coeff * (
        pow_func(tensor_0to1 + 1e-6, 1 / srgb_exponent)
    ) - (srgb_exponential_coeff - 1)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = where_func(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb



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
            'T' : mvp,
            'campos' : campos,
            'resolution' : [self.args.resolution, self.args.resolution],
            'spp' : 16,
            'img' : img,
            'mask' : mask,
            'dir_embeds' : dir_embeds,
            'envmap' : envmap
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


class DatasetTextAndEnvmapToImage(Dataset):
    # def __init__(self, dataset_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False):
    def __init__(self, config_path, args, examples=None):
        self.examples = examples
        self.base_dir = os.path.dirname(config_path)
        self.args = args
        self.steps_per_epoch = args.steps_per_epoch
        self.height = args.height
        self.width = args.width
        
        # metadata = pd.read_csv(os.path.join(self.base_dir, "metadata.csv"))
        # self.text = metadata["text"].to_list()
        
        if "transforms_test" in config_path:
            self.is_test = True
        else:
            self.is_test = False
        
        # Load config / transforms
        self.config = json.load(open(config_path, 'r'))
        # frames[i][j].keys() = ['env', 'rotation', 'flip', 'scale', 'views']
        # frames[i][j]['views'][k].keys() = ['transform_matrix', 'file_path', 'file_path_albedo', 'file_path_normal', 'file_path_orm', 'file_path_depth']
        # len(frames[i][j]['views'][k]) = 8
        # frames[i][j]['views'][k]['file_path'] = f"{view_id}_{same_envmap}.png"
        
        frames = self.config['frames']
        
        self.text = [""] * len(frames)
        
        self.n_objs = len(frames)
        self.n_envmaps = len(frames[0])
        self.n_views = len(frames[0][0]["views"])
        self.k_views = 4
        if self.k_views > self.n_views:
            self.k_views = self.n_views

        # Determine resolution & aspect ratio
        self.data_resolution = imageio.imread(os.path.join(self.base_dir, frames[0][0]["views"][0]['file_path'] + ".png")).shape[0:2]
        self.data_resolution = (self.data_resolution[1], self.data_resolution[0])
        self.aspect = self.data_resolution[1] / self.data_resolution[0]

        # Camera parameters
        self.fovx   = self.config['camera_angle_x']
        self.fovy   = fovx_to_fovy(self.fovx, self.aspect)
        self.proj   = perspective(self.fovy, self.aspect, 0.1, 1000.0)

        self.dir_embeds = torch.tensor(generate_directional_embeddings(
                                        shape=self.data_resolution,
                                        world2cam=None, normalize=True),
                                       dtype=torch.float32
                                       ).permute(2, 0, 1)
        
        print(f"DatasetTextAndEnvmapToImage: loading {self.n_objs} objects with shape {self.data_resolution}")


    def _parse_frame(self, config, idx, cam_near_far=[0.1, 1000.0]):
        imgs, envs, dir_embeds, mvps = [], [], [], []
        
        frames = self.config['frames']
        frame = frames[idx]

        if self.is_test:
            env_idx = 0
        else:
            env_idx = np.random.randint(len(frame))
        
        obj_views = frame[env_idx]['views']
        view_idx = np.random.randint(len(obj_views))

        # Load image
        view = obj_views[view_idx]
        img = read_hdr(os.path.join(self.base_dir, view['file_path']), return_type='torch').permute(2, 0, 1)
        
        # Camera parameters
        frame_transform = torch.tensor(view['transform_matrix'], dtype=torch.float32)
        mv = torch.linalg.inv(frame_transform)
        mv = mv @ rotate_x(-np.pi / 2)
        mvp = self.proj @ mv
        t, r = frame_transform[:3, 3], torch.linalg.norm(frame_transform[:3, 3])
        theta, phi = torch.arccos(t[2] / r), torch.arctan2(t[1], t[0])
        mvp = torch.tensor([theta, torch.sin(phi), torch.cos(phi), r])

        # Load envmap
        try:
            env = imageio.imread(os.path.join(self.args.envmap_path, frame[env_idx]['env']))[..., :3]
            if "scale" in frame[env_idx]:
                scale = frame[env_idx]["scale"]
            elif ".exr" in frame[env_idx]["env"]:
                scale = 150
            else:
                scale = 1
            env = env * scale
        except Exception as e:
            print(e)
            env = imageio.imread(os.path.join(ENV_MAP_PATH, frame[env_idx]['env']))[..., :3]
            
        # Transformations
        if "rotation" in frame[env_idx]:
            if frame[env_idx]["rotation"] > 0:
                env = rotate_image_fast(env, frame[env_idx]["rotation"])
        
        if "flip" in frame[env_idx]:
            if frame[env_idx]["flip"]:
                env = env[..., ::-1, :]
                
        env = env_map_to_cam_to_world_by_convention(env, torch.linalg.inv(mv).numpy())
        env = torch.from_numpy(env).permute(2, 0, 1)
        env = env.float() / 255.0
        env = env.half()

        # [b, v, c, h, w]
        # imgs = imgs.unsqueeze(0).repeat(self.args.batch_size, 1, 1, 1, 1)
        # mvps = mvps.unsqueeze(0).repeat(self.args.batch_size, 1, 1, 1, 1)
        # envs = envs.unsqueeze(0).repeat(self.args.batch_size, 1, 1, 1, 1)
        # dir_embeds = dir_embeds.unsqueeze(0).repeat(self.args.batch_size, 1, 1, 1, 1)
        return img, mvp, env, self.dir_embeds

    def __len__(self):
        # return self.n_objs if self.examples is None else self.examples
        return self.steps_per_epoch

    def __getitem__(self, itr):
        imgs, mvps, envs, dir_embeds = self._parse_frame(self.config, itr % self.n_objs)

        return {
            'text' : self.text[itr % self.n_objs],
            'image' : imgs,
            'envmap' : envs,
            'dir_embeds' : dir_embeds,
            'T' : mvps,
            'resolution' : (self.width, self.height),
        }
