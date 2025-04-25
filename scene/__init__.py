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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import * #GaussianModel,GaussianModel_ST,GaussianModel_ST2
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.camera_utils_st import cameraList_from_camInfosv2
import torch
import torch

class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0],multiview=False,duration=50, loader="colmap"):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        # args.eval :True
        if loader == "colmap" or loader == "colmapvalid":  # colmapvalid only for testing   # colmapvalid 仅用于测试
            scene_info = sceneLoadTypeCallbacks["Colmap_ST"](args.source_path, args.images, args.eval, multiview,duration=duration)
        elif loader == "MeetRoom":  # colmapvalid only for testing   # colmapvalid 仅用于测试
            scene_info = sceneLoadTypeCallbacks["MeetRoom"](args.source_path, args.images, args.eval, multiview,duration=duration)
        elif loader == "MeetRoom-3dgstream":  # colmapvalid only for testing   # colmapvalid 仅用于测试
            scene_info = sceneLoadTypeCallbacks["MeetRoom"](args.source_path, args.images, args.eval, multiview,duration=duration,type_='3dgstream')
        
        
        elif loader == "technicolor" or loader == "technicolorvalid" :
            scene_info = sceneLoadTypeCallbacks["Technicolor"](args.source_path, args.images, args.eval, multiview, duration=duration)
        elif loader == "immersive" or loader == "immersivevalid" or loader == "immersivess"  :
            scene_info = sceneLoadTypeCallbacks["Immersive"](args.source_path, args.images, args.eval, multiview, duration=duration)
        elif loader == "immersivevalidss":
            scene_info = sceneLoadTypeCallbacks["Immersive"](args.source_path, args.images, args.eval, multiview, duration=duration, testonly=True)

        
        elif os.path.exists(os.path.join(args.source_path, "sparse")) or os.path.exists(os.path.join(args.source_path, "colmap_sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            print("Found cameras_sphere.npz file, assuming DTU data set!")
            scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, "cameras_sphere.npz", "cameras_sphere.npz")
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json file, assuming Nerfies data set!")
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("Found calibration_full.json, assuming Neu3D data set!")
            scene_info = sceneLoadTypeCallbacks["plenopticVideo"](args.source_path, args.eval, 24)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found calibration_full.json, assuming Dynamic-360 data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path)
        elif os.path.exists(os.path.join(args.source_path, "train_meta.json")):
            print("Found train_meta.json, assuming CMU data set!")
            scene_info = sceneLoadTypeCallbacks["CMU"](args.source_path)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
        
        # Read flow data
        self.flow_dir = os.path.join(args.source_path, "raft_neighbouring")
        flow_list = os.listdir(self.flow_dir) if os.path.exists(self.flow_dir) else []
        flow_dirs_list = []
        for cam in scene_info.train_cameras:
            flow_dirs_list.append([os.path.join(self.flow_dir, flow_dir) for flow_dir in flow_list if flow_dir.startswith(cam.image_name+'.')])

        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # for resolution_scale in resolution_scales:
        #     print("Loading Training Cameras")
        #     self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, flow_dirs_list)
        #     print("Loading Test Cameras")
        #     self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # 根据不同的分辨率加载训练和测试相机
        for resolution_scale in resolution_scales:
            if loader in ["immersivess"]:
                print("Loading Training Cameras")
                print("Loading Training Cameras")
                assert resolution_scale == 1.0, "High frequency data only available at 1.0 scale"
                self.train_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.train_cameras,
                                                                                  resolution_scale, args, ss=True)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.test_cameras,
                                                                                 resolution_scale, args, ss=True)
            else:
                print("Loading Training Cameras")
            # 不同加载器类型下的相机加载处理
                # 不同加载器类型下的相机加载处理
                self.train_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.test_cameras, resolution_scale, args)


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
