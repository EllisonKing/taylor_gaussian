#!/bin/bash
GPU_ID=$1

run_pipeline() {

    # train
    CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py \
    --source_path data_spacetime/"$data_name_path"/colmap_0 \
    --model_path logs/technicolor/"$data_name" \
    --deform_type node \
    --node_num 1024 \
    --hyper_dim 8 \
    --is_blender \
    --eval \
    --gt_alpha_mask_as_scene_mask \
    --local_frame \
    --resolution 2 \
    --W 800 \
    --H 800 \
    --config arguments/"$data_name_path".py
    



    # render
    CUDA_VISIBLE_DEVICES="$GPU_ID" python render.py \
    --source_path data_spacetime/"$data_name_path"/colmap_0 \
    --model_path logs/technicolor/"$data_name" \
    --deform_type node \
    --node_num 1024 \
    --hyper_dim 8 \
    --is_blender \
    --eval \
    --gt_alpha_mask_as_scene_mask \
    --local_frame \
    --resolution 2 \
    --W 800 \
    --H 800 \
    --dataset_type $valloader
}


Neural3D_DA=("Birthday" "Train" "Painter" "Fatma")
valloader="technicolorvalid"
# run_pipeline
for data_name in "${Neural3D_DA[@]}"; 
do
    echo "Dataset: Technicolor/${data_name}"
    data_name_path="Technicolor/${data_name}"
    run_pipeline "$GPU_ID" "$data_name_path" "$data_name" "$valloader"
done

