ModelParams = dict(
    dataset_type='technicolor',

)

OptimizationParams = dict(
    densify_grad_threshold=0.0005, # 0.0002
    scaling_lr=0.001/10, # 0.001
    position_lr_init=1.6e-5/10, # 1.6e-5 #

    feature_lr = 0.0025/5,
    opacity_lr = 0.05/5,
    rotation_lr = 0.001/5,

    # densify_grad_threshold=0.0005,  # 0.0002
    # position_lr_init=1.6e-5 / 5,  # 0.00016, # 1.6e-5 #
    flag=0

)
PipelineParams = dict()
