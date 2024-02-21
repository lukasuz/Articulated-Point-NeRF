from builtins import print

import numpy as np
import torch
import math

from .load_dnerf import load_dnerf_data
from .load_robot import load_robot
from .load_zju import load_zju


def load_data(args, cfg, load_test_val=False, bg_col=1):

    K, depths = None, None
    times = None
    embeddings = None

    if args.dataset_type == 'dnerf':
        images, poses, times, render_poses, render_times, hwf, i_split, img_to_cam, masks = load_dnerf_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        near = 2.
        far = 6.
        if images.shape[-1] == 4:
            if bg_col == 1:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'wim':
        images, poses, K, times, render_poses, render_times, render_intrinsics, hw, i_split, img_to_cam, masks = load_robot(args.datadir, skip_images=args.skip_images, test=load_test_val)
        print('Loaded robot', images.shape, render_poses.shape, render_intrinsics.shape, hw, args.datadir)
        i_train, i_val, i_test = i_split

        hwf = [hw[0], hw[1], K[0,0,0]]
        near = 1.
        far = 6.

    elif args.dataset_type == 'zju':
        images, poses, K, times, render_poses, render_times, render_intrinsics, hw, i_split, img_to_cam, masks, embeddings = load_zju(args.datadir, video_len=cfg.data.video_len, step=1, load_test_val=load_test_val, bg_col=bg_col)
        print('Loaded ZJU', images.shape, render_poses.shape, render_intrinsics.shape, hw, args.datadir)
        i_train, i_val, i_test = i_split

        hwf = [hw[0], hw[1], K[0,0,0]]
        near = 1.
        far = 4.

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K
    render_poses = render_poses[...,:4]

    if times is None:
        times = torch.zeros(images.shape[0])
        render_times = torch.zeros(render_poses.shape[0])

    poses = torch.tensor(poses, dtype=torch.float32)
    Ks = torch.tensor(Ks, dtype=torch.float32)
    times = torch.tensor(times, dtype=torch.float32)

    data_dict = dict(
            hwf=hwf, HW=HW, Ks=Ks, near=near, far=far,
            i_train=i_train, i_val=i_val, i_test=i_test,
            poses=poses, render_poses=render_poses,
            images=images, depths=depths,
            irregular_shape=irregular_shape, times=times, 
            render_times=render_times, img_to_cam=img_to_cam,
            masks=masks, embeddings=embeddings,)

    return data_dict


