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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCV
from utils.graphics_utils import fov2focal
# def fov2focal(fov, pixels):
#     return pixels / (2 * math.tan(fov / 2))
from kornia import create_meshgrid
# from helper_model import pix2ndc
# from helper_train import getgtisint8
import random

def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0

def getgtisint8():
    #print("get current gt", bool(int(os.getenv('gtisint8'))))
    try:
        return bool(int(os.getenv('gtisint8')))
    except:
        return False


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,fid=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", near=0.01, far=100.0,
                 timestamp=0.0, rayo=None, rayd=None, rays=None, cxr=0.0,cyr=0.0, flow_dirs=[]
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.timestamp = timestamp
        self.flow_dirs = flow_dirs


        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.fid = torch.Tensor(np.array([fid])).to(self.data_device)
        # image is real image 
        if not isinstance(image, tuple):
            if getgtisint8():
                self.original_image = (image*255).to(torch.uint8) #.to(self.data_device) #.to(self.data_device)
            else:
                if "camera_" not in image_name:
                    self.original_image = image.clamp(0.0, 1.0) #.to(self.data_device)#.to(self.data_device)
                else:
                    self.original_image = image.clamp(0.0, 1.0).half() #.to(self.data_device) #.to(self.data_device)
            
            
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
            self.gt_alpha_mask = gt_alpha_mask
            if gt_alpha_mask is not None:
                self.gt_alpha_mask = self.gt_alpha_mask.to(self.data_device)
            # if gt_alpha_mask is not None:
            #     self.original_image *= gt_alpha_mask.to(self.data_device)
            # else:
            #     self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        else:
            self.image_width = image[0]
            self.image_height = image[1]
            self.original_image = None
        


        self.zfar = 100.0
        self.znear = 0.01  

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if cyr != 0.0 :
            self.cxr = cxr
            self.cyr = cyr
            self.projection_matrix = getProjectionMatrixCV(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=cxr, cy=cyr).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

         #gaussian-opacity-fields
        #tan_fovx = np.tan(self.FoVx / 2.0)
        #tan_fovy = np.tan(self.FoVy / 2.0)
        #self.focal_y = self.image_height / (2.0 * tan_fovy)
        #self.focal_x = self.image_width / (2.0 * tan_fovx)

        if rayd is not None:
            projectinverse = self.projection_matrix.T.inverse()
            camera2wold = self.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            
            xindx = pixgrid[:,:,0] # x 
            yindx = pixgrid[:,:,1] # y
      
            
            ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

            projected = ndccamera @ projectinverse.T 
            diretioninlocal = projected / projected[:,:,3:] #v 


            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

            
            self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)                                     #rayo.permute(2, 0, 1).unsqueeze(0)
            self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)    
            

        else :
            self.rayo = None
            self.rayd = None



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


class Camerass(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", near=0.01, far=100.0, timestamp=0.0, rayo=None, rayd=None, rays=None, cxr=0.0,cyr=0.0,
                 ):
        super(Camerass, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.timestamp = timestamp
        self.fisheyemapper = None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # image is real image 

        if not isinstance(image, tuple):
           
            if getgtisint8():
                self.original_image = (image*255) #.to(torch.uint8).to(self.data_device)
            else:
                if "camera_" not in image_name:
                    self.original_image = image.clamp(0.0, 1.0) #.to(self.data_device)
                else:
                    self.original_image = image.clamp(0.0, 1.0).half() #.to(self.data_device)
                print("read one")# lazy loader already in it
                self.image_width = self.original_image.shape[2]
                self.image_height = self.original_image.shape[1]

        else:
            self.image_width = image[0] 
            self.image_height = image[1] 
            self.original_image = None
        
        self.image_width = 2 * self.image_width
        self.image_height = 2 * self.image_height # 

        self.zfar = 100.0
        self.znear = 0.01  
        self.trans = trans
        self.scale = scale

        # w2c 
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if cyr != 0.0 :
            self.cxr = cxr
            self.cyr = cyr
            self.projection_matrix = getProjectionMatrixCV(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=cxr, cy=cyr).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # gaussian-opacity-fields
        #tan_fovx = np.tan(self.FoVx / 2.0)
        #tan_fovy = np.tan(self.FoVy / 2.0)
        #self.focal_y = self.image_height / (2.0 * tan_fovy)
        #self.focal_x = self.image_width / (2.0 * tan_fovx)

        if rayd is not None:
            projectinverse = self.projection_matrix.T.inverse()
            camera2wold = self.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            
            xindx = pixgrid[:,:,0] # x 
            yindx = pixgrid[:,:,1] # y
      
            
            ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

            projected = ndccamera @ projectinverse.T 
            diretioninlocal = projected / projected[:,:,3:] # 

            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

            
            self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)                                     #rayo.permute(2, 0, 1).unsqueeze(0)
            self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)                                                                          #rayd.permute(2, 0, 1).unsqueeze(0)
        else :
            self.rayo = None
            self.rayd = None
