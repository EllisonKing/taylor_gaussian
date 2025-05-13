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
import torch
from random import randint
import random 
import sys 
import uuid
import time 
import json

import numpy as np 
import cv2
from tqdm import tqdm
import shutil

sys.path.append("./thirdparty/gaussian_splatting")

    
def getcolmapsinglen3d(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --database_path " + dbfile+ " --image_path " + inputimagefolder

    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)


    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)

   # threshold is from   https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/scripts/local_colmap_and_resize.sh#L62
    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)


    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel  + " --output_path " + folder  \
    + " --output_type COLMAP" 
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)





def getcolmapsingleimundistort(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor SiftExtraction.max_image_size 6000 --database_path " + dbfile+ " --image_path " + inputimagefolder 

    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)


 

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    #Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
   



def getcolmapsingleimdistort(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor SiftExtraction.max_image_size 6000 --database_path " + dbfile+ " --image_path " + inputimagefolder 
    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
        

def getcolmapsingletechni(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --database_path " + dbfile+ " --image_path " + inputimagefolder 

    
    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)
    

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)


    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)

    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel + " --output_path " + folder  \
    + " --output_type COLMAP "  #
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)


    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)
    
    return 
    
