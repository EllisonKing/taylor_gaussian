#!/bin/bash
GPU_ID=$1


run_pipeline() {
    # train
    CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py \
    --source_path data/Neural3D/"$data_name"/colmap_0 \
    --model_path logs/N3DV/"$data_name"_test \
    --deform_type node \
    --node_num 1024 \
    --hyper_dim 8 \
    --is_blender \
    --gt_alpha_mask_as_dynamic_mask \
    --gs_with_motion_mask \
    --eval \
    --local_frame \
    --resolution 2 \
    --W 800 \
    --H 800 \
    --config arguments/N3DV/"$data_name".py
    # render
    CUDA_VISIBLE_DEVICES="$GPU_ID" python render.py \
    --source_path data/Neural3D/"$data_name"/colmap_0 \
    --model_path logs/N3DV/"$data_name"_test \
    --deform_type node \
    --node_num 1024 \
    --hyper_dim 8 \
    --is_blender \
    --gt_alpha_mask_as_dynamic_mask \
    --gs_with_motion_mask \
    --eval \
    --local_frame \
    --resolution 2 \
    --W 800 \
    --H 800 \
    --dataset_type colmap
}

Neural3D_DA=("cut_roasted_beef" "sear_steak" "flame_steak")
Neural3D_DA=( "cook_spinach" "flame_salmon_1" "coffee_martini" )
# run_pipeline
for data_name in "${Neural3D_DA[@]}"; 
do
    echo "Dataset: Neural3D_DA/${data_name}"
    run_pipeline "$GPU_ID" "$data_name"
done
