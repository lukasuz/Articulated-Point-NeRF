from copy import deepcopy

expname = None                    # experiment name
basedir = './logs/'               # where to store ckpts and logs

''' Template of data options
'''
data = dict(
    datadir=None,                 # path to dataset root folder
    dataset_type=None,            
    load2gpu_on_the_fly=True,    # do not load all images into gpu (to save gpu memory)
    testskip=1,                   # subsample testset to preview results
    white_bkgd=False,             # use white background (note that some dataset don't provide alpha and with blended bg color)
    half_res=True,              
    factor=4,                     
    ndc=False,                    # use ndc coordinate (only for forward-facing; not support yet)
    spherify=False,               # inward-facing
    llffhold=8,                   # testsplit
    load_depths=False,            # load depth
    use_bg_points=False,
    add_cam=False,
)

''' Template of training options
'''
train_config = dict(
    bg_col=0,
    N_iters=40000,                # number of optimization steps
    N_rand=4096,                  # batch size (number of random rays per optimization step)
    lrate_feature=8e-2,           # lr of  voxel grid
    lrate_featurenet=8e-4,
    lrate_deformation_net=6e-4,
    lrate_forward_warp=6e-4,
    lrate_densitynet=8e-4,
    lrate_timenet=8e-4,
    lrate_rgbnet=8e-4,           # lr of the mlp  
    lrate_decay=40,               # lr decay by 0.1 after every lrate_decay*1000 steps
    ray_sampler='in_maskcache',        # ray sampling strategies
    weight_main=1.0,              # weight of photometric loss
    weight_entropy_last=0.001,
    weight_rgbper=0.01,            # weight of per-point rgb loss
    tv_every=1,                   # count total variation loss every tv_every step
    tv_after=0,                   # count total variation loss from tv_from step
    tv_before=1e9,                   # count total variation before the given number of iterations
    tv_feature_before=10000,            # count total variation densely before the given number of iterations
    weight_tv_feature=0,
    pg_scale=[2000, 4000, 6000],
    weight_distortion=5e-2,
    weight_mask_loss=5e-2,
    skip_zero_grad_fields=['feature'],
)

''' Template of model and rendering options
'''

model_and_render = dict(
    num_voxels=160**3,          # expected number of voxel
    num_voxels_base=160**3,      # to rescale delta distance
    voxel_dim=12,                 # feature voxel grid dim
    defor_depth=5,               # depth of the deformation MLP 
    net_width=128,             # width of the  MLP
    alpha_init=1e-3,              # set the alpha values everywhere at the begin of training
    fast_color_thres=1e-4,           # threshold of alpha value to skip the fine stage sampled point
    stepsize=0.5,                 # sampling stepsize in volume rendering
    world_bound_scale=1.05,
    no_view_dir=False,
)

N_iters = 160000 * 2#  160000
full_t_iter= N_iters // 2
pcd_train_config = dict(
    bg_col=0,
    pose_one_each=False,
    N_iters=N_iters,
    weight_start_iter=full_t_iter,
    full_t_iter=N_iters // 2,
    lrate_decay=N_iters // 1000,
    # TiNeuVox
    lrate_rgbnet=1e-4,
    lrate_densitynet=1e-4,
    lrate_featurenet=1e-4,
    lrate_canonical_feat=1e-4,

    # Points
    lrate_gammas=1e-3,
    lrate_weights=1e-4,
    lrate_theta_weight=1e-4,
    lrate_forward_warp=1e-4,
    lrate_joints=1e-5,
    lrate_theta=1e-5,
    lrate_feat_net=1e-3,
    skip_zero_grad_fields=[],

    weight_render=2e+2,
    weight_chamfer2D=5e-3,
    
    # Regularizers
    weight_arap=5e-3,
    weight_joint_chamfer=1,
    weight_transformation_reg=1e-1,
    weight_tv=1e+1,
    weight_sparsity=2e-1,

    re_init_feat=False,
    re_init_mlps=False,
    avg_procrustes=False,
    over_parameterized_rot=True,
    use_global_view_dir=False,
    use_direct_loss=False,
    ray_sampler='random',
    embedding='full',
    pose_embedding_dim=64,
    N_rand=4096 * 2
)

pcd_model_and_render = dict(
    stepsize=0.5,
    world_bound_scale=1.05,
    fast_color_thres=1e-4,
    bone_length=10.,
    pcd_density_threshold=0.05,
    skeleton_density_threshold=0.1,
    canonical_pcd_num=1e+4,
)

del deepcopy
