ModelParams = dict(
    dataset_type=None,  # N3DV

)

OptimizationParams = dict(
#    densify_grad_threshold=0.0002, # 0.0002
#    scaling_lr=0.001, # 0.001
#    position_lr_init=1.6e-5/10, #1.6e-5 # 0.00016/5,

    densify_grad_threshold=0.0002, # 0.0002
    scaling_lr=0.005, # 0.001
    position_lr_init=1.6e-4, #1.6e-5 # 0.00016/5,
    flag=0
)
PipelineParams = dict()
