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
import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import imageio
from glob import glob
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud,BasicPointCloud_ST
from utils.camera_utils import camera_nerfies_from_JSON
import math
import copy

import natsort

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None

class CameraInfo_st(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    near: float
    far: float
    timestamp: float
    pose: np.array
    hpdirecitons: np.array
    cxr: float
    cyr: float


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

class SceneInfo_st(NamedTuple):
    point_cloud: BasicPointCloud_ST
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose

def getNerfppNorm(cam_info, apply=False):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    cam_centers = []
    if apply:
        c2ws = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        if apply:
            c2ws.append(C2W)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal
    translate = -center
    if apply:
        c2ws = np.stack(c2ws, axis=0)
        c2ws[:, :3, -1] += translate
        c2ws[:, :3, -1] /= radius
        w2cs = np.linalg.inv(c2ws)
        for i in range(len(cam_info)):
            cam = cam_info[i]
            cam_info[i] = cam._replace(R=w2cs[i, :3, :3].T, T=w2cs[i, :3, 3])
        apply_translate = translate
        apply_radius = radius
        translate = 0
        radius = 1.
        return {"translate": translate, "radius": radius, "apply_translate": apply_translate, "apply_radius": apply_radius}
    else:
        return {"translate": translate, "radius": radius}
    
def translate_cam_info(cam_info, translate):
    for i in range(len(cam_info)):
        cam = cam_info[i]
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        C2W[:3, 3] += translate
        W2C = np.linalg.inv(C2W)
        cam_info[i] = cam._replace(R=W2C[:3, :3].T, T=W2C[:3, 3])

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, msk_folder=None):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE" or intr.model == "OPENCV" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        print('image_name:',image_name)
        print('num_frames：',num_frames)
        if msk_folder is not None and image.size[-1] == 3:
            msk_path = os.path.join(msk_folder, os.path.basename(extr.name))
            mask = Image.open(msk_path)
            image = np.concatenate([np.asarray(image), np.asarray(mask)], axis=-1)
            image = Image.fromarray(image)

        fid = int(image_name) / (num_frames - 1)
        print('fid:',fid)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                        vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchPly_st(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    times = np.vstack([vertices['t']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud_ST(points=positions, colors=colors, normals=normals, times=times)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

from plyfile import PlyData, PlyElement
def storePly_st(path, xyzt, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('t', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    xyz = xyzt[:, :3]
    normals = np.zeros_like(xyz)

    elements = np.empty(xyzt.shape[0], dtype=dtype)
    attributes = np.concatenate((xyzt, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=16, apply_cam_norm=False, recenter_by_pcl=False):
    sparse_name = "sparse" if os.path.exists(os.path.join(path, "sparse")) else "colmap_sparse"
    try:
        cameras_extrinsic_file = os.path.join(path, f"{sparse_name}/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, f"{sparse_name}/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, f"{sparse_name}/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, f"{sparse_name}/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0 or True]  # Test all files with flow
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos + test_cam_infos, apply=apply_cam_norm)

    if recenter_by_pcl:
        ply_path = os.path.join(path, f"{sparse_name}/0/points3d_recentered.ply")
    elif apply_cam_norm:
        ply_path = os.path.join(path, f"{sparse_name}/0/points3d_normalized.ply")
    else:
        ply_path = os.path.join(path, f"{sparse_name}/0/points3d.ply")
    bin_path = os.path.join(path, f"{sparse_name}/0/points3D.bin")
    txt_path = os.path.join(path, f"{sparse_name}/0/points3D.txt")
    adj_path = os.path.join(path, f"{sparse_name}/0/camera_adjustment")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        if apply_cam_norm:
            xyz += nerf_normalization["apply_translate"]
            xyz /= nerf_normalization["apply_radius"]
        if recenter_by_pcl:
            pcl_center = xyz.mean(axis=0)
            translate_cam_info(train_cam_infos, - pcl_center)
            translate_cam_info(test_cam_infos, - pcl_center)
            xyz -= pcl_center
            np.savez(adj_path, translate=-pcl_center)
        storePly(ply_path, xyz, rgb)
    elif recenter_by_pcl:
        translate = np.load(adj_path + '.npz')['translate']
        translate_cam_info(train_cam_infos, translate=translate)
        translate_cam_info(test_cam_infos, translate=translate)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", no_bg=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        frames = sorted(frames, key=lambda x: int(os.path.basename(x['file_path']).split('.')[0].split('_')[-1]))
        for idx, frame in enumerate(frames):
            if frame["file_path"].endswith('jpg') or frame["file_path"].endswith('png'):
                cam_name = os.path.join(path, frame["file_path"])
            else:
                cam_name = os.path.join(path, frame["file_path"] + extension)
            if 'time' in frame:
                frame_time = frame['time']
            else:
                frame_time = idx / len(frames)
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.join(path, frame["file_path"]))), 'rgba')):
                cam_name = os.path.join(os.path.dirname(os.path.dirname(os.path.join(path, frame["file_path"]))), 'rgba', os.path.basename(frame['file_path'])).replace('.jpg', '.png')

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] 
            if no_bg:
                norm_data[:, :, :3] = norm_data[:, :, 3:4] * norm_data[:, :, :3] + bg * (1 - norm_data[:, :, 3:4])
            
            arr = np.concatenate([arr, mask], axis=-1)

            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGBA" if arr.shape[-1] == 4 else "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], fid=frame_time))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png", no_bg=True):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension, no_bg=no_bg)
    print(f"Read Train Transforms with {len(train_cam_infos)} cameras")
    if os.path.exists(os.path.join(path, "transforms_test.json")):
        test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension, no_bg=no_bg)
    else:
        test_cam_infos = []

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)
    # nerf_normalization = {'translation': np.zeros([3], dtype=np.float32), 'radius': 1.}

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        if os.path.exists(os.path.join(path, 'rgbd')):
            import liblzfse  # https://pypi.org/project/pyliblzfse/
            def load_depth(filepath):
                with open(filepath, 'rb') as depth_fh:
                    raw_bytes = depth_fh.read()
                    decompressed_bytes = liblzfse.decompress(raw_bytes)
                    depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)
                    depth_img = depth_img.reshape((256, 192))
                return depth_img
            from utils.camera_utils import loadCam
            from collections import namedtuple
            ARGS = namedtuple('ARGS', ['resolution', 'data_device', 'load2gpu_on_the_fly'])
            args = ARGS(1, 'cpu', True)
            viewpoint_camera = loadCam(args, id, train_cam_infos[0], 1, [])
            w, h = viewpoint_camera.image_width, viewpoint_camera.image_height
            gt_depth = torch.from_numpy(load_depth(os.path.join(path, 'rgbd', '0.depth')))
            gt_depth = torch.nn.functional.interpolate(gt_depth[None, None], (h, w))[0, 0]
            far, near = viewpoint_camera.zfar, viewpoint_camera.znear
            u, v = torch.meshgrid(torch.linspace(.5, w-.5, w, device=gt_depth.device) / w * 2 - 1, torch.linspace(.5, h-.5, h, device = gt_depth.device) / h * 2 - 1, indexing='xy')
            u, v = u.reshape([-1]), v.reshape([-1])
            d = gt_depth.reshape([-1])
            nan_mask = d.isnan()
            nan_mask = torch.logical_or(nan_mask, d > 4)
            z = far / (far - near) * d - far * near / (far - near)
            uvz = torch.stack([u * d, v * d, z, d], dim=-1)
            pcl = uvz @ torch.inverse(viewpoint_camera.full_proj_transform)
            pcl = pcl[:, :3][~nan_mask]
            shs = torch.rand_like(pcl) / 255.0
            num_pts = shs.shape[0]
            pcd = BasicPointCloud(points=pcl, colors=SH2RGB(
                shs), normals=np.zeros((num_pts, 3)))
            xyz = pcl
        else:
            # Since this data set has no colmap data, we start with random points
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")

            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            # xyz = np.random.random((num_pts, 3)) * 20 - 10
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
                shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readDTUCameras(path, render_camera, object_camera):
    camera_dict = np.load(os.path.join(path, render_camera))
    images_lis = sorted(glob(os.path.join(path, 'image/*.png')))
    masks_lis = sorted(glob(os.path.join(path, 'mask/*.png')))
    n_images = len(images_lis)
    cam_infos = []
    cam_idx = 0
    for idx in range(0, n_images):
        image_path = images_lis[idx]
        image = np.array(Image.open(image_path))
        mask = np.array(imageio.imread(masks_lis[idx])) / 255.0
        image = Image.fromarray((image * mask).astype(np.uint8))
        world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
        fid = camera_dict['fid_%d' % idx] / (n_images / 12 - 1)
        image_name = Path(image_path).stem
        scale_mat = camera_dict['scale_mat_%d' % idx].astype(np.float32)
        P = world_mat @ scale_mat
        P = P[:3, :4]

        K, pose = load_K_Rt_from_P(None, P)
        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, -c, -b, pose[3:, :]], 0)

        S = np.eye(3)
        S[1, 1] = -1
        S[2, 2] = -1
        pose[1, 3] = -pose[1, 3]
        pose[2, 3] = -pose[2, 3]
        pose[:3, :3] = S @ pose[:3, :3] @ S

        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, c, b, pose[3:, :]], 0)

        pose[:, 3] *= 0.5

        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        FovY = focal2fov(K[0, 0], image.size[1])
        FovX = focal2fov(K[0, 0], image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readNeuSDTUInfo(path, render_camera, object_camera):
    print("Reading DTU Info")
    train_cam_infos = readDTUCameras(path, render_camera, object_camera)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def view_synthesis(cps, factor=10):
    frame_num = cps.shape[0]
    cps = np.array(cps)
    from scipy.spatial.transform import Slerp
    from scipy.spatial.transform import Rotation as R
    from scipy import interpolate as intp
    rots = R.from_matrix(cps[:, :3, :3])
    slerp = Slerp(np.arange(frame_num), rots)
    tran = cps[:, :3, -1]
    f_tran = intp.interp1d(np.arange(frame_num), tran.T)

    new_num = int(frame_num * factor)

    new_rots = slerp(np.linspace(0, frame_num - 1, new_num)).as_matrix()
    new_trans = f_tran(np.linspace(0, frame_num - 1, new_num)).T

    new_cps = np.zeros([new_num, 4, 4], np.float)
    new_cps[:, :3, :3] = new_rots
    new_cps[:, :3, -1] = new_trans
    new_cps[:, 3, 3] = 1
    return new_cps


def readNerfiesCameras(path, inter_valid=True): # inter_valid=False): # 
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        # train_img = dataset_json['train_ids']
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.25
    elif name.startswith('NeRF'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 1.0
    elif name.startswith('interp'):
        train_img = list(np.array(dataset_json['ids'])[[i for i in range(len(dataset_json['ids'])) if i % 4 == 0]])
        val_img = list(np.array(dataset_json['ids'])[[i for i in range(len(dataset_json['ids'])) if i % 4 == 2]])
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        train_img = dataset_json['ids']
        val_img = dataset_json['ids'][:4]
        all_img = train_img + val_img
        ratio = 0.5

    train_num = len(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    all_time = [meta_json[i]['time_id'] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    selected_time = set(all_time)

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)

    if os.path.exists(f'{path}/rgb/rgba'):
        print('Read RGBA images!')
        all_img = [f'{path}/rgb/rgba/{i}.png' for i in all_img]
        msk_path = None
    else:
        msk_path = [f'{path}/resized_mask/{int(1 / ratio)}x/{i}.png.png' for i in all_img]
        msk_path = msk_path if os.path.exists(f'{path}/resized_mask/{int(1 / ratio)}x/') else None
        all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]
    
    if inter_valid:
        cam_infos = []
        cps = []
        fids = []
        for idx in range(len(train_img)):
            image_path = all_img[idx]
            image = np.array(Image.open(image_path))
            image = Image.fromarray((image).astype(np.uint8))
            image_name = Path(image_path).stem

            if msk_path is not None:
                mask = 255 - np.array(Image.open(msk_path[idx]))[..., None]
                image = Image.fromarray(np.concatenate([np.asarray(image), mask], axis=-1).astype('uint8'))

            orientation = all_cam_params[idx]['orientation'].T
            position = -all_cam_params[idx]['position'] @ orientation
            focal = all_cam_params[idx]['focal_length']
            fid = all_time[idx]
            T = position
            R = orientation

            cp = np.eye(4)
            cp[:3, :3] = R
            cp[:3, 3] = T
            cps.append(cp)
            fids.append(fid)

            FovY = focal2fov(focal, image.size[1])
            FovX = focal2fov(focal, image.size[0])
            cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], fid=fid)
            cam_infos.append(cam_info)
        cps_valid = view_synthesis(cps=np.stack(cps), factor=5)
        fids_valid = np.linspace(0, 1, cps_valid.shape[0])
        for idx in range(cps_valid.shape[0]):
            R, T = cps_valid[idx][:3, :3], cps_valid[idx][:3, 3]
            cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], fid=fids_valid[idx])
            cam_infos.append(cam_info)
    else:
        cam_infos = []
        for idx in range(len(all_img)):
            image_path = all_img[idx]
            image = np.array(Image.open(image_path))
            image = Image.fromarray((image).astype(np.uint8))
            image_name = Path(image_path).stem

            if msk_path is not None:
                mask = 255 - np.array(Image.open(msk_path[idx]))[..., None]
                image = Image.fromarray(np.concatenate([np.asarray(image), mask], axis=-1).astype('uint8'))

            orientation = all_cam_params[idx]['orientation'].T
            position = -all_cam_params[idx]['position'] @ orientation
            focal = all_cam_params[idx]['focal_length']
            fid = all_time[idx]
            T = position
            R = orientation

            FovY = focal2fov(focal, image.size[1])
            FovX = focal2fov(focal, image.size[0])
            cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], fid=fid)
            cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale


def readNerfiesColmapCameras(path):
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.25
    elif name.startswith('NeRF'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 1.0
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        train_img = dataset_json['ids'][::4]
        val_img = dataset_json['ids'][2::4]
        all_img = train_img + val_img
        ratio = 0.5

    train_num = len(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    all_time = [meta_json[i]['time_id'] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    selected_time = set(all_time)
    
    sparse_name = "sparse" if os.path.exists(os.path.join(path, 'colmap', "sparse")) else "colmap_sparse"
    cameras_extrinsic_file = os.path.join(path, 'colmap', f"{sparse_name}/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, 'colmap', f"{sparse_name}/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    
    if os.path.exists(f'{path}/rgb/rgba'):
        img_path = f'{path}/rgb/rgba'
        msk_path = None
    else:
        img_path = f'{path}/rgb/{int(1 / ratio)}x/'
        msk_path = f'{path}/resized_mask/{int(1 / ratio)}x/'
        msk_path = msk_path if os.path.exists(msk_path) else None
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder = img_path, msk_folder=msk_path)
    name2idx = {cam.image_name: idx for idx, cam in enumerate(cam_infos_unsorted)}

    cam_infos = []
    for idx in range(len(all_img)):
        cam_infos.append(cam_infos_unsorted[name2idx[all_img[idx]]])
        cam_infos[-1]._replace(fid=all_time[idx])
    return cam_infos, train_num, 0, 1


def readNerfiesInfo(path, eval):
    print("Reading Nerfies Info")
    if os.path.exists(os.path.join(path, 'colmap')):
        cam_infos, train_num, scene_center, scene_center = readNerfiesColmapCameras(path)
        recenter_by_pcl = apply_cam_norm = True
    else:
        cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path)
        recenter_by_pcl = apply_cam_norm = False

    if eval:
        train_cam_infos = cam_infos[:train_num]
        test_cam_infos = cam_infos[train_num:]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos, apply=apply_cam_norm)

    if os.path.exists(os.path.join(path, 'colmap')):
        print('Using COLMAP for Nerfies!')
        sparse_name = "sparse" if os.path.exists(os.path.join(path, 'colmap', "sparse")) else "colmap_sparse"
        if recenter_by_pcl:
            ply_path = os.path.join(path, f"colmap/{sparse_name}/0/points3d_recentered.ply")
        elif apply_cam_norm:
            ply_path = os.path.join(path, f"colmap/{sparse_name}/0/points3d_normalized.ply")
        else:
            ply_path = os.path.join(path, f"colmap/{sparse_name}/0/points3d.ply")
        bin_path = os.path.join(path, f"colmap/{sparse_name}/0/points3D.bin")
        txt_path = os.path.join(path, f"colmap/{sparse_name}/0/points3D.txt")
        adj_path = os.path.join(path, f"colmap/{sparse_name}/0/camera_adjustment")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            if apply_cam_norm:
                xyz += nerf_normalization["apply_translate"]
                xyz /= nerf_normalization["apply_radius"]
            if recenter_by_pcl:
                pcl_center = xyz.mean(axis=0)
                translate_cam_info(train_cam_infos, - pcl_center)
                translate_cam_info(test_cam_infos, - pcl_center)
                xyz -= pcl_center
                np.savez(adj_path, translate=-pcl_center)
            storePly(ply_path, xyz, rgb)
        else:
            translate = np.load(adj_path + '.npz')['translate']
            translate_cam_info(train_cam_infos, translate=translate)
            translate_cam_info(test_cam_infos, translate=translate)
    else:
        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path):
            print(f"Generating point cloud from nerfies...")

            xyz = np.load(os.path.join(path, "points.npy"))
            xyz = (xyz - scene_center) * scene_scale
            num_pts = xyz.shape[0]
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
                shs), normals=np.zeros((num_pts, 3)))

            storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromNpy(path, npy_file, split, hold_id, num_images):
    cam_infos = []
    video_paths = sorted(glob(os.path.join(path, 'frames/*')))
    poses_bounds = np.load(os.path.join(path, npy_file))

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]

    n_cameras = poses.shape[0]
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])

    i_test = np.array(hold_id)
    video_list = i_test if split != 'train' else list(
        set(np.arange(n_cameras)) - set(i_test))

    for i in video_list:
        video_path = video_paths[i]
        c2w = poses[i]
        images_names = sorted(os.listdir(video_path))
        n_frames = num_images

        matrix = np.linalg.inv(np.array(c2w))
        R = np.transpose(matrix[:3, :3])
        T = matrix[:3, 3]

        for idx, image_name in enumerate(images_names[:num_images]):
            image_path = os.path.join(video_path, image_name)
            image = Image.open(image_path)
            frame_time = idx / (n_frames - 1)

            FovX = focal2fov(focal, image.size[0])
            FovY = focal2fov(focal, image.size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
                                        image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], fid=frame_time))

            idx += 1
    return cam_infos


def readPlenopticVideoDataset(path, eval, num_images, hold_id=[0]):
    print("Reading Training Camera")
    train_cam_infos = readCamerasFromNpy(path, 'poses_bounds.npy', split="train", hold_id=hold_id,
                                         num_images=num_images)

    print("Reading Training Camera")
    test_cam_infos = readCamerasFromNpy(
        path, 'poses_bounds.npy', split="test", hold_id=hold_id, num_images=num_images)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, 'points3D.ply')
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCMUInfo(path, split):
    camera_infos = []
    md = json.load(open(f"{path}/{split}_meta.json", 'r'))
    num_timesteps = 20  # len(md['fn'])
    for t in range(num_timesteps):
        for c in range(len(md['fn'][t])):
            w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
            image_path = f"{path}/ims/"
            image_name = md['fn'][t][c]
            im = np.array(copy.deepcopy(Image.open(f"{path}/ims/{image_name}")))
            im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
            try:
                seg = np.array(copy.deepcopy(Image.open(f"{path}/seg/{image_name.replace('.jpg', '.png')}"))).astype(np.float32)
            except:
                seg = None

            fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
            w2c = torch.tensor(w2c).cuda().float()
            w2c = w2c.transpose(0, 1)
            tanfovx = w / (2 * fx)
            tanfovy = h / (2 * fy)

            image = np.asarray(Image.open(os.path.join(image_path, image_name)))
            if seg is not None:
                image = np.concatenate([image, seg[..., None]*255], axis=-1)
            image = Image.fromarray(image.astype('uint8'))

            cam = CameraInfo(
                uid = c,
                R = w2c[:3, :3].cpu().numpy(),
                T = w2c.T[:3, 3].cpu().numpy(),
                FovY = math.atan(tanfovy) * 2,
                FovX = math.atan(tanfovx) * 2,
                image = image,
                image_path = image_path,
                image_name = image_name,
                width = w,
                height = h,
                fid = t / 150,
            )
            camera_infos.append(cam)
    return camera_infos


def readCMUSceneInfo(path, apply_cam_norm=True, recenter_by_pcl=True):
    print('Reading Training Camera')
    train_cam_infos = readCMUInfo(path, 'train')
    print('Reading Test Camera')
    test_cam_infos = readCMUInfo(path, 'test')
    nerf_normalization = getNerfppNorm(train_cam_infos, apply=apply_cam_norm)
    
    if recenter_by_pcl:
        ply_path = os.path.join(path, 'points3D_recenter.ply')
    elif apply_cam_norm:
        ply_path = os.path.join(path, 'points3D_normalize.ply')
    else:
        ply_path = os.path.join(path, 'points3D.ply')
    adj_path = os.path.join(path, "camera_adjustment")
    if not os.path.exists(ply_path):
        init_pt_cld = np.load(os.path.join(path, 'init_pt_cld.npz'))['data']
        xyz = init_pt_cld[:, :3]
        shs = init_pt_cld[:, 3:6]
        if apply_cam_norm:
            xyz += nerf_normalization["apply_translate"]
            xyz /= nerf_normalization["apply_radius"]
        if recenter_by_pcl:
            pcl_center = xyz.mean(axis=0)
            translate_cam_info(train_cam_infos, - pcl_center)
            xyz -= pcl_center
            np.savez(adj_path, translate=-pcl_center)
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros_like(xyz))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        translate = np.load(adj_path + '.npz')['translate']
        translate_cam_info(train_cam_infos, translate=translate)
    pcd = fetchPly(ply_path)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readColmapCameras_st(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=50):
    cam_infos = []

    # pose in llff. pipeline by hypereel
    originnumpy = os.path.join(os.path.dirname(os.path.dirname(images_folder)), "poses_bounds.npy")
    with open(originnumpy, 'rb') as numpy_file:
        poses_bounds = np.load(numpy_file)

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        bounds = poses_bounds[:, -2:]

        near = bounds.min() * 0.95
        far = bounds.max() * 1.05

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # 19, 3, 5

        H, W, focal = poses[0, :, -1]
        cx, cy = W / 2.0, H / 2.0

        K = np.eye(3)
        K[0, 0] = focal * W / W / 2.0
        K[0, 2] = cx * W / W / 2.0
        K[1, 1] = focal * H / H / 2.0
        K[1, 2] = cy * H / H / 2.0

        imageH = int(H // 2)  # note hard coded to half of the original image size
        imageW = int(W // 2)

    totalcamname = []
    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)

    sortedtotalcamelist = natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i  # map each cam with a number

    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]

        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        for j in range(startime, startime + int(duration)):
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image_path = image_path.replace("colmap_" + str(startime), "colmap_{}".format(j), 1)
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            image = Image.open(image_path)
            if j == startime:
                # cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=hpposes[sortednamedict[os.path.basename(extr.name)]], hpdirecitons=hpdirecitons,cxr=0.0, cyr=0.0)
                cam_info = CameraInfo_st(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                      image_name=image_name, width=width, height=height, near=near, far=far,fid=j/ duration, # add fid=1/ duration,
                                      timestamp=(j - startime) / duration, pose=1, hpdirecitons=1, cxr=0.0, cyr=0.0)

            else:
                cam_info = CameraInfo_st(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                      image_name=image_name, width=width, height=height, near=near, far=far,fid=j/ duration, # add fid=1/ duration,
                                      timestamp=(j - startime) / duration, pose=None, hpdirecitons=None, cxr=0.0,
                                      cyr=0.0)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfo_ST(path, images, eval, llffhold=8, multiview=False, duration=50):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    parentdir = os.path.dirname(path)  #

    near = 0.01  #
    far = 100  #

    starttime = os.path.basename(path).split("_")[1]  # colmap_0,   #
    assert starttime.isdigit(), "Colmap folder name must be colmap_<startime>_<duration>!"  #
    starttime = int(starttime)  #

    cam_infos_unsorted = readColmapCameras_st(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir), near=near, far=far,
                                           startime=starttime, duration=duration)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    
    print('eval:',eval)
    if eval:  #
        print('eval:',eval)
        train_cam_infos = cam_infos[duration:]

        test_cam_infos = cam_infos[:duration]
        print('train_cam_infos:', len(train_cam_infos))
        print('test_cam_infos:', len(test_cam_infos))
        uniquecheck = []
        for cam_info in test_cam_infos:
            if cam_info.image_name not in uniquecheck:
                uniquecheck.append(cam_info.image_name)
        assert len(uniquecheck) == 1

        sanitycheck = []
        for cam_info in train_cam_infos:
            if cam_info.image_name not in sanitycheck:
                sanitycheck.append(cam_info.image_name)
        for testname in uniquecheck:
            assert testname not in sanitycheck
    #else:
    #    train_cam_infos = cam_infos
    #    test_cam_infos = cam_infos[:2]  # dummy  #
    #    print('train_cam_infos2:', len(train_cam_infos))
    #    print('test_cam_infos2:', len(test_cam_infos))
    # 111
    # testidx = [0]
    # if eval:
    #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx not in testidx]
    #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in testidx]
    # else:
    #     train_cam_infos = cam_infos
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    totalply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + ".ply")

    if not os.path.exists(totalply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        totalxyz = []
        totalrgb = []
        totaltime = []
        for i in range(starttime, starttime + duration):
            thisbin_path = os.path.join(path, "sparse/0/points3D.bin").replace("colmap_" + str(starttime),
                                                                               "colmap_" + str(i), 1)
            xyz, rgb, _ = read_points3D_binary(thisbin_path)
            totalxyz.append(xyz)
            totalrgb.append(rgb)
            totaltime.append(np.ones((xyz.shape[0], 1)) * (i - starttime) / duration)
        xyz = np.concatenate(totalxyz, axis=0)
        rgb = np.concatenate(totalrgb, axis=0)
        totaltime = np.concatenate(totaltime, axis=0)
        assert xyz.shape[0] == rgb.shape[0]
        xyzt = np.concatenate((xyz, totaltime), axis=1)
        storePly_st(totalply_path, xyzt, rgb)
    try:
        pcd = fetchPly_st(totalply_path)
    except:
        pcd = None

    scene_info = SceneInfo_st(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=totalply_path)

    return scene_info

def readColmapCamerasTechnicolor(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=50):
    cam_infos = []
    totalcamname = []
    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)

    sortedtotalcamelist = natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i  # map each cam with a number

    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        for j in range(startime, startime + int(duration)):
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image_path = image_path.replace("colmap_" + str(startime), "colmap_{}".format(j), 1)

            cxr = ((intr.params[2]) / width - 0.5)
            cyr = ((intr.params[3]) / height - 0.5)

            K = np.eye(3)
            K[0, 0] = focal_length_x  # * 0.5
            K[0, 2] = intr.params[2]  # * 0.5
            K[1, 1] = focal_length_y  # * 0.5
            K[1, 2] = intr.params[3]  # * 0.5

            halfH = round(height / 2.0)
            halfW = round(width / 2.0)

            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)

            image = Image.open(image_path)

            if j == startime:
                cam_info = CameraInfo_st(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                      image_name=image_name, width=width, height=height, near=near, far=far,fid=j/ duration, #1/ duration,
                                      timestamp=(j - startime) / duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr)
            else:
                cam_info = CameraInfo_st(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                      image_name=image_name, width=width, height=height, near=near, far=far,fid=j/ duration, #1/ duration,
                                      timestamp=(j - startime) / duration, pose=None, hpdirecitons=None, cxr=cxr,
                                      cyr=cyr)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasImmersiveTestonly(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0,
                                       duration=50):
    cam_infos = []

    totalcamname = []
    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)

    sortedtotalcamelist = natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i  # map each cam with a number

    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        # T[1] = T[1] + 0.2  # y incrase by 1
        # T[2] = T[2] + 0.65
        # T[0] = T[0] + 0.65 # x by 0.65
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        for j in range(startime, startime + int(duration)):
            # image_path = os.path.join(images_folder, os.path.basename(extr.name))
            # image_name = os.path.basename(image_path).split(".")[0]
            # image_path = image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)

            parentfolder = os.path.dirname(images_folder)
            parentfolder = os.path.dirname(parentfolder)
            image_name = extr.name.split(".")[0]

            rawvideofolder = os.path.join(parentfolder, os.path.basename(image_name))

            image_path = os.path.join(rawvideofolder, str(j) + ".png")

            # image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)

            # K = np.eye(3)
            # K[0, 0] = focal_length_x * 0.5
            # K[0, 2] = intr.params[2] * 0.5
            # K[1, 1] = focal_length_y * 0.5
            # K[1, 2] = intr.params[3] * 0.5

            cxr = ((intr.params[2]) / width - 0.5)
            cyr = ((intr.params[3]) / height - 0.5)

            K = np.eye(3)
            K[0, 0] = focal_length_x  # * 0.5
            K[0, 2] = intr.params[2]  # * 0.5
            K[1, 1] = focal_length_y  # * 0.5
            K[1, 2] = intr.params[3]  # * 0.5

            # halfH = round(height / 2.0 )
            # halfW = round(width / 2.0 )
            if not os.path.exists(image_path):
                image_path = image_path.replace("_S14", "")
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)

            if image_name == "camera_0001":
                image = Image.open(image_path)
            else:
                image = None
            if j == startime:
                cam_info = CameraInfo_st(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                      image_name=image_name, width=width, height=height, near=near, far=far,fid=1/ duration,
                                      timestamp=(j - startime) / duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr)
            else:
                cam_info = CameraInfo_st(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                      image_name=image_name, width=width, height=height, near=near, far=far,fid=1/ duration,
                                      timestamp=(j - startime) / duration, pose=None, hpdirecitons=None, cxr=cxr,
                                      cyr=cyr)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasImmersive(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=50):
    cam_infos = []

    totalcamname = []
    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)

    sortedtotalcamelist = natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i  # map each cam with a number

    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # if extr.name not in ["camera_0005.png", "camera_0001.png"]:
        #         continue

        for j in range(startime, startime + int(duration)):
            # image_path = os.path.join(images_folder, os.path.basename(extr.name))
            # image_name = os.path.basename(image_path).split(".")[0]
            # image_path = image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)

            parentfolder = os.path.dirname(images_folder)
            parentfolder = os.path.dirname(parentfolder)
            image_name = extr.name.split(".")[0]

            rawvideofolder = os.path.join(parentfolder, os.path.basename(image_name))

            image_path = os.path.join(rawvideofolder, str(j) + ".png")

            # image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)

            # K = np.eye(3)
            # K[0, 0] = focal_length_x * 0.5
            # K[0, 2] = intr.params[2] * 0.5
            # K[1, 1] = focal_length_y * 0.5
            # K[1, 2] = intr.params[3] * 0.5

            cxr = ((intr.params[2]) / width - 0.5)
            cyr = ((intr.params[3]) / height - 0.5)

            K = np.eye(3)
            K[0, 0] = focal_length_x  # * 0.5
            K[0, 2] = intr.params[2]  # * 0.5
            K[1, 1] = focal_length_y  # * 0.5
            K[1, 2] = intr.params[3]  # * 0.5

            if not os.path.exists(image_path):
                image_path = image_path.replace("_S14", "")
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            image = Image.open(image_path)
            if j == startime:
                cam_info = CameraInfo_st(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                      image_name=image_name, width=width, height=height, near=near, far=far,fid=1/ duration,
                                      timestamp=(j - startime) / duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr)
            else:
                cam_info = CameraInfo_st(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                      image_name=image_name, width=width, height=height, near=near, far=far,fid=1/ duration,
                                      timestamp=(j - startime) / duration, pose=None, hpdirecitons=None, cxr=cxr,
                                      cyr=cyr)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfoTechnicolor(path, images, eval, llffhold=8, multiview=False, duration=50):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    parentdir = os.path.dirname(path)

    starttime = os.path.basename(path).split("_")[1]  # colmap_0,
    assert starttime.isdigit(), "Colmap folder name must be colmap_<startime>_<duration>!"
    starttime = int(starttime)

    near = 0.01
    far = 100
    cam_infos_unsorted = readColmapCamerasTechnicolor(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                                      images_folder=os.path.join(path, reading_dir), near=near, far=far,
                                                      startime=starttime, duration=duration)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    # for cam in cam_infos:
    #     print(cam.image_name)
    # for cam_info in cam_infos:
    #     print(cam_info.uid, cam_info.R, cam_info.T, cam_info.FovY, cam_info.image_name)
    print('eval:', eval)
    if eval:
        train_cam_infos = [_ for _ in cam_infos if "cam10" not in _.image_name]
        test_cam_infos = [_ for _ in cam_infos if "cam10" in _.image_name]
        print('len(test_cam_infos):',len(test_cam_infos))
        if len(test_cam_infos) > 0:
            uniquecheck = []
            for cam_info in test_cam_infos:
                if cam_info.image_name not in uniquecheck:
                    uniquecheck.append(cam_info.image_name)
            assert len(uniquecheck) == 1

            sanitycheck = []
            for cam_info in train_cam_infos:
                if cam_info.image_name not in sanitycheck:
                    sanitycheck.append(cam_info.image_name)
            for testname in uniquecheck:
                assert testname not in sanitycheck
        else:
            first_cam = cam_infos[0].image_name
            print("do custom loader training, select first cam as test frame: ", first_cam)
            cam_infos = natsort.natsorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
            train_cam_infos = [_ for _ in cam_infos if first_cam not in _.image_name]
            test_cam_infos = [_ for _ in cam_infos if first_cam in _.image_name]
        print('train_cam_infos:',len(train_cam_infos))
        print('test_cam_infos:', len(test_cam_infos))
    # else:
    #     train_cam_infos = cam_infos
    #     test_cam_infos = cam_infos[:4]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    totalply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + ".ply")

    if not os.path.exists(totalply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        totalxyz = []
        totalrgb = []
        totaltime = []
        for i in range(starttime, starttime + duration):
            thisbin_path = os.path.join(path, "sparse/0/points3D.bin").replace("colmap_" + str(starttime),
                                                                               "colmap_" + str(i), 1)
            xyz, rgb, _ = read_points3D_binary(thisbin_path)
            totalxyz.append(xyz)
            totalrgb.append(rgb)
            totaltime.append(np.ones((xyz.shape[0], 1)) * (i - starttime) / duration)
        xyz = np.concatenate(totalxyz, axis=0)
        rgb = np.concatenate(totalrgb, axis=0)
        totaltime = np.concatenate(totaltime, axis=0)
        assert xyz.shape[0] == rgb.shape[0]
        xyzt = np.concatenate((xyz, totaltime), axis=1)
        storePly_st(totalply_path, xyzt, rgb)
    try:
        pcd = fetchPly_st(totalply_path)
    except:
        pcd = None

    scene_info = SceneInfo_st(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=totalply_path)
    return scene_info


def readColmapSceneInfoImmersive(path, images, eval, llffhold=8, multiview=False, duration=50, testonly=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    near = 0.01
    far = 100

    starttime = os.path.basename(path).split("_")[1]  # colmap_0,
    assert starttime.isdigit(), "Colmap folder name must be colmap_<startime>_<duration>!"
    starttime = int(starttime)

    # readColmapCamerasImmersiveTestonly
    if testonly:
        cam_infos_unsorted = readColmapCamerasImmersiveTestonly(cam_extrinsics=cam_extrinsics,
                                                                cam_intrinsics=cam_intrinsics,
                                                                images_folder=os.path.join(path, reading_dir),
                                                                near=near, far=far, startime=starttime,
                                                                duration=duration)
    else:
        cam_infos_unsorted = readColmapCamerasImmersive(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                                        images_folder=os.path.join(path, reading_dir), near=near,
                                                        far=far, startime=starttime, duration=duration)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    print('eval:', eval)
    if eval:
        train_cam_infos = cam_infos[duration:]  # + cam_infos[:duration] # for demo only
        test_cam_infos = cam_infos[:duration]
        uniquecheck = []
        for cam_info in test_cam_infos:
            if cam_info.image_name not in uniquecheck:
                uniquecheck.append(cam_info.image_name)
        assert len(uniquecheck) == 1

        sanitycheck = []
        for cam_info in train_cam_infos:
            if cam_info.image_name not in sanitycheck:
                sanitycheck.append(cam_info.image_name)
        for testname in uniquecheck:
            assert testname not in sanitycheck

        print('train_cam_infos:',len(train_cam_infos))
        print('test_cam_infos:', len(test_cam_infos))
    else:
        train_cam_infos = cam_infos  # for demo without eval
        test_cam_infos = cam_infos[:duration]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    totalply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + ".ply")

    # if os.path.exists(ply_path):
    #     os.remove(ply_path)

    # if os.path.exists(totalply_path):
    #     os.remove(totalply_path)

    if not os.path.exists(totalply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        totalxyz = []
        totalrgb = []
        totaltime = []

        takeoffset = 0
        for i in range(starttime, starttime + duration):
            thisbin_path = os.path.join(path, "sparse/0/points3D.bin").replace("colmap_" + str(starttime),
                                                                               "colmap_" + str(i), 1)
            xyz, rgb, _ = read_points3D_binary(thisbin_path)

            totalxyz.append(xyz)
            totalrgb.append(rgb)
            totaltime.append(np.ones((xyz.shape[0], 1)) * (i - starttime) / duration)
        xyz = np.concatenate(totalxyz, axis=0)
        rgb = np.concatenate(totalrgb, axis=0)
        totaltime = np.concatenate(totaltime, axis=0)
        assert xyz.shape[0] == rgb.shape[0]
        xyzt = np.concatenate((xyz, totaltime), axis=1)
        storePly(totalply_path, xyzt, rgb)
    try:
        pcd = fetchPly(totalply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=totalply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Colmap_ST": readColmapSceneInfo_ST,
    # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,
    # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "DTU": readNeuSDTUInfo,  # DTU dataset used in Tensor4D [https://github.com/DSaurus/Tensor4D]
    "nerfies": readNerfiesInfo,
    # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "plenopticVideo": readPlenopticVideoDataset,
    # Neural 3D dataset in [https://github.com/facebookresearch/Neural_3D_Video]
    "CMU": readCMUSceneInfo,
    "Technicolor": readColmapSceneInfoTechnicolor,
    "Immersive": readColmapSceneInfoImmersive,
}
