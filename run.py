import argparse
import copy
import os
import random
import time
import json
from builtins import print
from pathlib import Path
import math
import pickle

import imageio
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from lib import utils, temporalpoints, tineuvox
from lib.load_data import load_data

from torch_efficient_distloss import flatten_eff_distloss
from skeletonizer import create_skeleton, visualise_skeletonizer

import tensorboardX as tbx
import torchvision

from skimage.morphology import remove_small_holes
from skimage import filters
from cc3d import largest_k
import cv2

def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument("--seed", type=int, default=0, help='Random seed')
    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--overwrite_cache", action='store_true')
    parser.add_argument("--use_cache", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--load_test_val", action='store_true')
    parser.add_argument("--joint_placement", action='store_true')
    parser.add_argument("--visualise_weights", action='store_true')
    parser.add_argument("--visualise_canonical", action='store_true')
    parser.add_argument("--repose_pcd", action='store_true')
    parser.add_argument("--first_stage_only", action='store_true')
    parser.add_argument("--second_stage_only", action='store_true')
    parser.add_argument("--debug_bone_merging", action='store_true')
    parser.add_argument("--visualise_warp", action='store_true')
    parser.add_argument("--render_pcd_direct", action='store_true')
    parser.add_argument("--render_pcd", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')
    parser.add_argument("--eval_psnr", action='store_true')
    parser.add_argument("--ablation_tag", type=str)
    parser.add_argument("--degree_threshold", type=float, default=0.)
    parser.add_argument("--skip_load_images", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_save",   type=int, default=5000)
    parser.add_argument("--fre_test", type=int, default=500000,
                        help='frequency of test')
    parser.add_argument("--basedir_append_suffix", type=str, default='',)
    parser.add_argument("--step_to_half", type=int, default=100000,
                        help='The iteration when fp32 becomes fp16')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    return parser

@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, test_times=None, render_factor=0, eval_psnr=False,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,
                      inverse_y=False, flip_x=False, flip_y=False, batch_size = 4096 * 2, verbose=True, 
                      render_pcd_direct=False, render_flow=False,
                      fixed_viewdirs=None):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = torch.clone(Ks)
        HW = HW // render_factor
        Ks[:, :2, :3] = Ks[:, :2, :3] // render_factor
    
    rgbs = []
    depths = []
    weights = []
    flows = []
    psnrs = []
    ssims = []
    joints = {}
    bones = None
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses, disable = not verbose)):

        H, W = HW[i]
        K = Ks[i].to(torch.float32)
        rays_o, rays_d, viewdirs = tineuvox.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        
        if fixed_viewdirs is not None:
            viewdirs = fixed_viewdirs
        
        pixel_coords = torch.stack(torch.meshgrid(torch.arange(0, W), torch.arange(0, H)), dim=-1).reshape(-1, 2).to(torch.float32).to(rays_o.device)
        
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        time_one = test_times[i]*torch.ones_like(rays_o[:,0:1])

        if type(model) is not temporalpoints.TemporalPoints:
            keys = ['rgb_marched', 'depth']
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, ts, **render_kwargs).items() if k in keys}
                for ro, rd, vd, ts in zip(rays_o.split(batch_size, 0), rays_d.split(batch_size, 0), viewdirs.split(batch_size, 0),time_one.split(batch_size, 0))
            ]
        else:
            keys = ['rgb_marched', 'depth', 'weights']
            if render_flow: keys.append('flow')
            render_result_chunks = []

            for ro, rd, vd, ts, px in zip(rays_o.split(batch_size, 0), rays_d.split(batch_size, 0), viewdirs.split(batch_size, 0), time_one.split(batch_size, 0), pixel_coords.split(batch_size, 0)):
                render_kwargs['rays_o'] = ro
                render_kwargs['rays_d'] = rd
                render_kwargs['viewdirs'] = vd
                render_kwargs['pixel_coords'] = px
                cam_per_ray = torch.zeros(len(ro))[:,None]

                if render_flow:
                    i_delta = max(i-1, 0)
                    flow_t_delta = test_times[i_delta] - test_times[i]
                else:
                    flow_t_delta = None
 
                out = model(ts[0], render_depth=True, render_kwargs=render_kwargs, render_weights=True,
                            render_pcd_direct=render_pcd_direct, poses=c2w[None], Ks=Ks[i][None], 
                            cam_per_ray=cam_per_ray, get_skeleton=True)
                
                if out['joints'] is not None:

                    if not render_kwargs['inverse_y']:
                        out['joints'][:,:,0] = (HW[0,0] - 1) - out['joints'][:,:,0]

                    if not i in joints.keys():
                        joints[i] = out['joints'][0].cpu().numpy()
                        bones = out['bones']          

                if render_pcd_direct:
                    out['rgb_marched'] = out['rgb_marched_direct']
                
                chunk = {k: v for k, v in out.items() if k in keys}
                render_result_chunks.append(chunk)

        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()

        try:
            weight = render_result['weights'].cpu().numpy()
            weights.append(weight)
        except:
            pass

        rgbs.append(rgb)
        depths.append(depth)
        
        if gt_imgs is not None and render_factor == 0:
            if eval_psnr:
                p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name = 'alex', device = c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name = 'vgg', device = c2w.device))

    if len(psnrs):
        # create text file and write results into a single file
        if savedir is not None:
            with open(os.path.join(savedir, 'results.txt'), 'w') as f:
                if eval_psnr: f.write('psnr: ' + str(np.mean(psnrs)) + '\n')
                if eval_ssim: f.write('ssim: ' + str(np.mean(ssims)) + '\n')
                if eval_lpips_vgg: f.write('lpips_alex: ' + str(np.mean(lpips_alex)) + '\n')
                if eval_lpips_alex: f.write('lpips_vgg: ' + str(np.mean(lpips_vgg)) + '\n')
        
        if eval_psnr: print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, 'img_{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
        
        for i in trange(len(weights)):
            rgb8 = utils.to8b(weights[i])
            filename = os.path.join(savedir, 'weights_{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    weights = np.array(weights)
    flows = np.array(flows)
    joints = [joints[i] for i in range(len(joints))]
    joints = np.array(joints).astype(np.int32)

    for i in range(len(weights)):
        img = weights[i]        

        for bone in bones:
            img = cv2.line(img, joints[i][bone[0]], joints[i][bone[1]], color=(0, 0, 0), thickness=1)

        for j in range(joints.shape[1]):
            img = cv2.circle(img, joints[i][j], radius=3, color=(0, 0, 0), thickness=-1)

        weights[i] = img
    
    return rgbs, depths, weights, flows

@torch.no_grad()
def render_repose(rot_params, render_poses, HW, Ks, ndc, model, render_kwargs,
                gt_imgs=None, savedir=None, render_factor=0, eval_psnr=False,
                eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,
                inverse_y=False, flip_x=False, flip_y=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    assert type(model) is temporalpoints.TemporalPoints

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    Ks = torch.tensor(Ks)
    rgbs = []
    depths = []
    weights = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    joints = {}
    bones = None

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = tineuvox.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        batch_size = 4096 * 2

        keys = ['rgb_marched', 'depth', 'weights']
        render_result_chunks = []

        for ro, rd, vd in zip(rays_o.split(batch_size, 0), rays_d.split(batch_size, 0), viewdirs.split(batch_size, 0)):
            render_kwargs['rays_o'] = ro
            render_kwargs['rays_d'] = rd
            render_kwargs['viewdirs'] = vd
            out = model(None, render_depth=True, render_kwargs=render_kwargs, render_weights=True, rot_params=rot_params[i], calc_min_max=True, get_skeleton=True, poses=c2w[None], Ks=Ks[i][None])
            chunk = {k: v for k, v in out.items() if k in keys}
            render_result_chunks.append(chunk)

            if out['joints'] is not None:
                
                if not render_kwargs['inverse_y']:
                    out['joints'][:,:,0] = (HW[0,0] - 1) - out['joints'][:,:,0]

                if not i in joints.keys():
                    joints[i] = out['joints'][0].cpu().numpy()
                    bones = out['bones']          


        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        try:
            weight = render_result['weights'].cpu().numpy()
            weights.append(weight)
        except:
            pass

        rgbs.append(rgb)
        depths.append(depth)
        
        if i==0:
            print('Testing', rgb.shape)

    if len(psnrs):
        if eval_psnr: print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, 'img_{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
        
        for i in trange(len(weights)):
            rgb8 = utils.to8b(weights[i])
            filename = os.path.join(savedir, 'weights_{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    weights = np.array(weights)

    joints = [joints[i] for i in range(len(joints))]
    joints = np.array(joints).astype(np.int32)

    if len(joints) > 0:
        for i in range(len(weights)):
            img = weights[i]        

            for bone in bones:
                img = cv2.line(img, joints[i][bone[0]], joints[i][bone[1]], color=(0, 0, 0), thickness=1)

            for j in range(joints.shape[1]):
                img = cv2.circle(img, joints[i][j], radius=3, color=(0, 0, 0), thickness=-1)

            weights[i] = img

    return rgbs, depths, weights

def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def load_everything(args, cfg, use_cache=False, overwrite=False):
    '''Load images / poses / camera settings / data split.
    '''
    cfg.data.skip_images = bool(args.skip_load_images)

    if not os.path.isdir(cfg.data.datadir):
        cache_file_folder = cfg.data.datadir.split('.pickle')[0]
        os.makedirs(cfg.data.datadir.split('.pickle')[0], exist_ok=True)
        cache_file = Path(cache_file_folder) / 'cache.pth'
    else:
        cache_file = Path(cfg.data.datadir) / 'cache.pth'
    if use_cache and not overwrite and cache_file.is_file():
        with cache_file.open("rb") as f:
            data_dict = pickle.load(f)
        return data_dict

    try:
        bg_col = cfg.train_config.bg_col
    except:
        bg_col = None
    data_dict = load_data(cfg.data, cfg, args.load_test_val, bg_col=bg_col)
    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images','times', 'render_times',
            'img_to_cam', 'masks'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    if use_cache:
        with cache_file.open('wb') as f:
            pickle.dump(data_dict, f)

    return data_dict

def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    xyz_min = torch.tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[kwargs['img_to_cam'][i_train]], poses[kwargs['img_to_cam'][i_train]]):
        rays_o, rays_d, viewdirs = tineuvox.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=cfg.data.ndc, flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y, inverse_y=cfg.data.inverse_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max

def train_pcd(args, cfg, cfg_model, cfg_train, read_path, save_path, data_dict, tineuvox_model, canonical_t, tensorboard_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = tbx.SummaryWriter(tensorboard_path)
    os.makedirs(tensorboard_path, exist_ok=True)
    os.chmod(tensorboard_path, 0o755)

    ## SET UP TRAINING RAYS ##
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images, times, render_times, masks = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 
            'render_poses', 'images',
            'times','render_times','masks'
        ]
    ]
    times_i_train = times[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

    # init rendering setup
    render_kwargs = {
        'near': near,
        'far': far,
        'bg': cfg.train_config.bg_col,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y, 
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    # init batch rays sampler
    rgb_tr, index_to_times, rays_o_tr, rays_d_tr, viewdirs_tr, pix_to_ray, masks_tr, index_to_cam = temporalpoints.get_training_rays_in_maskcache_sampling(
        rgb_tr_ori=images,
        masks_tr_ori=masks,
        times=times_i_train,
        train_poses=poses,
        Ks=Ks,
        HW=HW,
        i_train =i_train,
        ndc=cfg.data.ndc, 
        model=tineuvox_model,
        render_kwargs=render_kwargs,
        img_to_cam = data_dict['img_to_cam'], 
        **render_kwargs)

    unique_times = times_i_train.unique()

    model_kwargs = copy.deepcopy(cfg_model)
    canonical_data = torch.load(os.path.join(read_path, 'pcds', f'canonical.tar'))
    canonical_pcd = canonical_data['pcd']
    canonical_feat = canonical_data['feat']
    canonical_alpha = canonical_data['alphas']
    canonical_rgbs = canonical_data['rgbs']
    xyz_min = canonical_data['xyz_min'] * model_kwargs['world_bound_scale']
    xyz_max = canonical_data['xyz_max'] * model_kwargs['world_bound_scale']
    voxel_size = canonical_data['voxel_size']

    ## SET UP MODEL ##
    last_ckpt_path = os.path.join(save_path, 'temporalpoints_last.tar')
    skeleton_data = torch.load(os.path.join(read_path, 'pcds', f'skeleton.tar'))
    skeleton_pcd = torch.tensor(skeleton_data['skeleton_pcd'])
    joints = torch.tensor(skeleton_data['joints'])
    bones = skeleton_data['bones']

    # init model
    if cfg_train.use_global_view_dir:
        frozen_view_dir = viewdirs_tr.median(dim=0)[0]
    else:
        frozen_view_dir = None
    model = temporalpoints.TemporalPoints(
        canonical_pcd=canonical_pcd,
        canonical_feat=canonical_feat,
        canonical_alpha=canonical_alpha,
        canonical_rgbs=canonical_rgbs,
        skeleton_pcd=skeleton_pcd,
        joints=joints,
        bones=bones,
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        voxel_size=voxel_size,
        tineuvox=tineuvox_model,
        embedding=cfg_train.embedding,
        frozen_view_dir=frozen_view_dir,
        over_parameterized_rot=cfg_train.over_parameterized_rot,
        avg_procrustes=cfg_train.avg_procrustes,
        re_init_feat=cfg_train.re_init_feat,
        re_init_mlps=cfg_train.re_init_mlps,
        pose_embedding_dim=cfg_train.pose_embedding_dim,
        **model_kwargs)

    model = model.to(device)

    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0, additional_params=None)

    arap_lst = []
    weight_tv_lst = []
    mse_lst = []
    psnr_lst = []
    trans_reg_loss_lst = []
    joint_chamfer_lst = []
    weight_sparsity_lst = []
    chamfer2D_loss_lst = []

    ## TRAINING ##
    start = 0
    time0 = time.time()
    global_step = -1

    ### Tensorboard prep ###
    tb_num_imgs = 5
    tb_factor = 2

    tb_mask = np.random.randint(0, len(images), tb_num_imgs)
    num_cams = len(np.unique(data_dict['img_to_cam']))
    gt_images = images[tb_mask].permute(0,3,1,2)
    if gt_images.dtype == torch.uint8:
        gt_images = gt_images.float() / 255
    resize = torchvision.transforms.Resize([gt_images.shape[2] // tb_factor, gt_images.shape[2] // tb_factor])
    gt_images = resize(gt_images)

    cam_indx = 3
    cam_indices = torch.where(torch.tensor(data_dict['img_to_cam']) == cam_indx)[0]
    cam_indices = cam_indices[torch.linspace(0, len(cam_indices)-1, 40).round().long()].cpu().numpy()

    gt_images_vid = resize(images[cam_indices].to(device).permute(0,3,1,2))
    if gt_images_vid.dtype == torch.uint8:
        gt_images_vid = gt_images_vid.float() / 255

    # Assume that we always have at least 10 timesteps
    canonical_t_indx = torch.argmin(((unique_times - cfg.data.canonical_t)**2).sqrt()).long()
    def get_range(max_len, num=10):
        t_max = math.ceil(canonical_t_indx + num / 2)
        t_min = math.ceil(canonical_t_indx - num / 2)

        if num >= max_len:
            t_min = 0
            t_max = max_len
        elif t_max > max_len:
            overflow = t_max % max_len
            t_min -= overflow
            t_max = max_len
        elif t_min < 0:
            underflow = abs(t_min)
            t_max += underflow
            t_min = 0

        return t_max, t_min

    sampler = utils.InverseProportionalSampler(len(unique_times))

    try:
        weight_start_iter = cfg_train.weight_start_iter
    except:
        weight_start_iter = 0

    print("")
    print("Weight start iter:", weight_start_iter)
    print("")

    for global_step in trange(1+start, 1+cfg_train.N_iters):
        optimizer.zero_grad(set_to_none = True)

        ## SAMPLE TIME
        num = min(max((len(unique_times) / cfg_train.full_t_iter) * (global_step), 1), len(unique_times))
        t_max, t_min = get_range(len(unique_times), num)

        rnd_i = sampler.sample(t_min, t_max)
        
        time_key = unique_times[rnd_i.item()]
        time_sel = torch.tensor([float(time_key)])

        ## SAMPLE RAYS ##
        b_range = index_to_times[time_sel.item()]

        sel_i = torch.randint(b_range[0], b_range[1], (cfg_train.N_rand,)).long().to(rgb_tr.device)
        img_i = torch.div(sel_i, (images.shape[1] * images.shape[1]), rounding_mode='floor').unsqueeze(-1)
        cam_per_ray = img_i % len(poses)
        target = rgb_tr[sel_i]

        if target.dtype == torch.uint8:
            target = target.float() / 255.

        target_mask = masks_tr[sel_i, 0].float()
        sel_ray = pix_to_ray[sel_i].long()
        rays_o = rays_o_tr[sel_ray]
        rays_d = rays_d_tr[sel_ray]
        viewdirs = viewdirs_tr[sel_ray]

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)
            time_sel = time_sel.to(device)
            target_mask = target_mask.to(device)

        render_kwargs['rays_o'] = rays_o
        render_kwargs['rays_d'] = rays_d
        render_kwargs['viewdirs'] = viewdirs

        res = model(time_sel, False, render_kwargs, render_pcd_direct=False, 
                    poses=poses, Ks=torch.tensor(Ks), cam_per_ray=cam_per_ray)
        t_hat_pcd = res['t_hat_pcd']
        rgb_marched = res['rgb_marched']

        ## LOSSESS ##
        loss = 0
        if cfg_train.weight_render > 0:
            mse_loss = F.mse_loss(rgb_marched, target)
            #alpha_loss = F.mse_loss(alpha_last, target_mask)
            img_loss = mse_loss #+ alpha_loss
            psnr = utils.mse2psnr(mse_loss.clone().detach())
            mse_loss = img_loss * cfg_train.weight_render
            
            mse_lst.append(mse_loss.item())
            psnr_lst.append(psnr.item())
            loss += mse_loss

        if cfg_train.weight_arap > 0:
            arap_loss = cfg_train.weight_arap * model.get_arap_loss(t_hat_pcd)
            arap_lst.append(arap_loss.item())
            loss += arap_loss

        if cfg_train.weight_tv > 0:
                weight_tv_loss = cfg_train.weight_tv * model.get_neighbour_weight_tv_loss()
                weight_tv_lst.append(weight_tv_loss.item())
                loss += weight_tv_loss

        if global_step >= weight_start_iter:
            if (cfg_train.weight_sparsity > 0): # and (global_step > cfg_train.full_t_iter)
                weight_sparsity_loss = cfg_train.weight_sparsity * model.get_weight_sparsity_loss()
                weight_sparsity_lst.append(weight_sparsity_loss.item())
                loss += weight_sparsity_loss
        
        if cfg_train.weight_transformation_reg > 0:
            trans_reg_loss = cfg_train.weight_transformation_reg * model.get_transformation_regularisation_loss()
            trans_reg_loss_lst.append(trans_reg_loss.item())
            loss += trans_reg_loss

        if cfg_train.weight_joint_chamfer > 0:
            joint_chamfer_loss = cfg_train.weight_joint_chamfer * model.get_joint_chamfer_loss()
            joint_chamfer_lst.append(joint_chamfer_loss.item())
            loss += joint_chamfer_loss

        if cfg_train.weight_chamfer2D > 0:
            # Select a random time step
            chamfer_mask = torch.where(times == time_sel)[0]

            num_rnd_cam_i = min(5, len(chamfer_mask))
            rnd_cam_i = torch.randperm(len(chamfer_mask))[:num_rnd_cam_i].long()
            chamfer_mask = chamfer_mask[rnd_cam_i]
            if cfg_train.pose_one_each: # eg. D-Nerf
                poses_temp = data_dict['poses'][chamfer_mask].squeeze(-1).to(device)
                Ks_temp = torch.tensor(data_dict['Ks'][chamfer_mask]).to(device).to(torch.float32)
            else:
                # Select a random pose
                poses_temp = data_dict['poses'].to(device)[rnd_cam_i]
                Ks_temp = torch.tensor(data_dict['Ks']).to(device).to(torch.float32)[rnd_cam_i]
            
            chamfer2D_loss = 0
            iter_pcds = [t_hat_pcd]
            for pcd in iter_pcds:
                projected_points_hat = utils.project_point_to_image_plane(pcd, poses_temp, Ks_temp)
                # # # Mask Compability
                if not render_kwargs['inverse_y']:
                    projected_points_hat[:,:,0] = (images.shape[1] - 1) - projected_points_hat[:,:,0]
                projected_points_hat = projected_points_hat.flip(-1)
                # Mask Compability end

                M = 3000
                N = 3000
                masks_iter = masks[chamfer_mask.cpu()].squeeze(-1).float()
                mask_pcd = [torch.cat([mask.unsqueeze(-1) for mask in torch.where(masks_iter[i] > 0)], dim=-1) for i in range(len(masks_iter))]
                mask_pcd = torch.cat([mask[torch.randint(0, mask.shape[0], (M,)).long().cpu()].unsqueeze(0) for mask in mask_pcd], dim=0).to(projected_points_hat.device).float()

                chamfer2D_loss += model.get_batch_chamfer_loss(projected_points_hat, mask_pcd, N=N, M=None)

            chamfer2D_loss *= cfg_train.weight_chamfer2D
            chamfer2D_loss_lst.append(chamfer2D_loss.item())
            loss += chamfer2D_loss

        # size = images.shape[1]
        # projected_points_hat = projected_points_hat.round().clip(0,size-1)
        # projected_points_hat = projected_points_hat.detach().cpu().numpy().astype(np.int32)
        # mask_pcd = mask_pcd.detach().cpu().numpy().astype(np.int32)

        # import matplotlib.pyplot as plt
        # for i, j in zip(range(len(projected_points_hat)), torch.where(times == time_sel)[0]):

        #     img = np.zeros((size, int(size * 3), 3))
        #     img[projected_points_hat[i, :, 0], projected_points_hat[i, :, 1]] = 1
        #     img[mask_pcd[i, :, 0], size + mask_pcd[i, :, 1]] = 1

        #     # img[:, 512:1024, :] = masks[j].clip(0, 1).numpy().repeat(3, axis=-1)
        #     img[:, size*2:, :] = images[j].float().div(255.).clip(0, 1).numpy()
        #     plt.imsave(f'test_{i}.png', img)

        loss = loss
        loss.backward()

        optimizer.step()

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step%args.i_print == 0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'

            tqdm.write(f'pcd training : iter {global_step:6d} / PSNR: {np.mean(psnr_lst):5.2f} / t_range: {t_min:.2f}-{t_max:.2f} / Eps: {eps_time_str}')

            writer.add_scalar("metrics/PSNR", np.mean(psnr_lst), global_step)
            writer.add_scalar("metrics/IMG_Loss", np.mean(mse_lst), global_step)
            writer.add_scalar("metrics/ARAP", np.mean(arap_lst), global_step)
            writer.add_scalar("metrics/Weight_TV", np.mean(weight_tv_lst), global_step)
            writer.add_scalar("metrics/Trans._Reg.", np.mean(trans_reg_loss_lst), global_step)
            writer.add_scalar("metrics/Joint_Chamfer", np.mean(joint_chamfer_lst), global_step)
            writer.add_scalar("metrics/Weight_Sparsity", np.mean(weight_sparsity_lst), global_step)
            writer.add_scalar("metrics/Chamfer2D", np.mean(chamfer2D_loss_lst), global_step)
            writer.add_scalar("metrics/eps_time", eps_time, global_step)

            mse_lst = []
            psnr_lst = []
            arap_lst = []
            weight_tv_lst = []
            joint_chamfer_lst = []
            trans_reg_loss_lst = []
            weight_sparsity_lst = []
            chamfer2D_loss_lst = []
        
        if (global_step % args.i_save == 0) or (global_step == 1): 
            ## Render training images
            pred_images, _, pred_weights, _ = render_viewpoints(
                model, 
                render_poses=data_dict['poses'][data_dict['img_to_cam'][tb_mask]],
                HW=data_dict['HW'][tb_mask],
                Ks=data_dict['Ks'][data_dict['img_to_cam'][tb_mask]],
                test_times=data_dict['times'][tb_mask],
                ndc=cfg.data.ndc,
                render_kwargs=render_kwargs,
                batch_size = 4096,
                render_factor=tb_factor,
                verbose=False,
                render_pcd_direct=False,
                flip_x = render_kwargs['flip_x'],
                flip_y = render_kwargs['flip_y'],
                inverse_y=render_kwargs['inverse_y'])
                
            pred_images = torch.tensor(pred_images).permute(0,3,1,2)
            pred_weights = torch.tensor(pred_weights).permute(0,3,1,2)
            payload = torch.concat([gt_images.to('cuda'), pred_images, pred_weights], dim=0)
            writer.add_image('payload', torchvision.utils.make_grid(payload, nrow=tb_num_imgs), global_step=global_step)

            # Render static cam comparison video
            # Render full model
            pred_images, _, pred_weights, _ = render_viewpoints(
                model, 
                render_poses=data_dict['poses'][data_dict['img_to_cam'][cam_indices]],
                HW=data_dict['HW'][cam_indices],
                Ks=data_dict['Ks'][data_dict['img_to_cam'][cam_indices]],
                test_times=torch.linspace(0,1, len(cam_indices)), # data_dict['times'][cam_indices],
                ndc=cfg.data.ndc,
                render_kwargs=render_kwargs,
                batch_size = 4096 * 2,
                render_factor=tb_factor,
                verbose=False,
                render_pcd_direct=False,
                flip_x = render_kwargs['flip_x'],
                flip_y = render_kwargs['flip_y'],
                inverse_y=render_kwargs['inverse_y'])
            
            # Render PCD based on frozen rgb and alphas
            pred_images_pcd, _, _, _ = render_viewpoints(
                    model, 
                    render_poses=data_dict['poses'][data_dict['img_to_cam'][cam_indices]],
                    HW=data_dict['HW'][cam_indices],
                    Ks=data_dict['Ks'][data_dict['img_to_cam'][cam_indices]],
                    test_times=torch.linspace(0,1, len(cam_indices)), # data_dict['times'][cam_indices],
                    ndc=cfg.data.ndc,
                    render_kwargs=render_kwargs,
                    batch_size = 4096 * 2,
                    render_factor=tb_factor,
                    verbose=False,
                    render_pcd_direct=True,
                    flip_x = render_kwargs['flip_x'],
                    flip_y = render_kwargs['flip_y'],
                    inverse_y=render_kwargs['inverse_y'])
            pred_images_pcd = torch.tensor(pred_images_pcd).permute(0,3,1,2)
            pred_images = torch.tensor(pred_images).permute(0,3,1,2)
            pred_weights = torch.tensor(pred_weights).permute(0,3,1,2)
            video_temp = torch.concat([gt_images_vid, pred_images_pcd, pred_images, pred_weights], dim=3).unsqueeze(0)

            writer.add_video('video', video_temp, global_step=global_step, fps=4)

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
        }, last_ckpt_path)
        print('pcd training: saved checkpoints at', last_ckpt_path)

def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift

    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images, times, render_times, masks = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 
            'render_poses', 'images',
            'times','render_times', 'masks'
        ]
    ]
    # times = torch.tensor(times)
    times_i_train = times[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
    if os.path.isfile(last_ckpt_path):
        print('fine_last.tar already exists, skipping training first phase of training.')
        return # right now, if there is already a file in the directory, just skip this function

    # init model and optimizer
    start = 0
    # init model
    model_kwargs = copy.deepcopy(cfg_model)

    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale) :
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
    model = tineuvox.TiNeuVox(
        xyz_min=xyz_min, xyz_max=xyz_max,
        num_voxels=num_voxels,
        **model_kwargs)
    model = model.to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

    # init rendering setup
    render_kwargs = {
        'near': near,
        'far': far,
        'bg': cfg.train_config.bg_col,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y, 
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    # NOTE: No mask cache here, only checking whether ray is within feature volume
    # NOTE: We pass all images, because the data loader only loads the training images
    rgb_tr, times_flatten, rays_o_tr, rays_d_tr, viewdirs_tr, pix_to_ray, masks_tr = tineuvox.get_training_rays_in_maskcache_sampling(
        rgb_tr_ori=images,
        masks_tr_ori=masks,
        times=times_i_train,
        train_poses=poses,
        i_train = i_train,
        Ks=Ks,
        HW=HW,
        ndc=cfg.data.ndc, 
        model=model, 
        render_kwargs=render_kwargs, 
        img_to_cam = data_dict['img_to_cam'],
        **render_kwargs)
    index_generator = tineuvox.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
    batch_index_sampler = lambda: next(index_generator)

    torch.cuda.empty_cache()
    psnr_lst = []
    dist_loss_lst = []
    inv_loss_lst = []
    delta_loss_lst = []
    mask_loss_lst = []
    time0 = time.time()
    global_step = -1

    for global_step in trange(1+start, 1+cfg_train.N_iters):
        if global_step == args.step_to_half:
            model.feature.data=model.feature.data.half()
        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, tineuvox.TiNeuVox):
                model.scale_volume_grid(cur_voxels)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            masks_target = masks_tr[sel_i]

            if target.dtype == torch.uint8:
                target = target.float() / 255.

            if masks_target.dtype == torch.uint8:
                masks_target = masks_target.float() / 255.
            else:
                masks_target = masks_target.float()

            target_alpha_inv_last = 1 - masks_target
            sel_ray = pix_to_ray[sel_i].long()
            rays_o = rays_o_tr[sel_ray]
            rays_d = rays_d_tr[sel_ray]
            viewdirs = viewdirs_tr[sel_ray]
            times_sel = times_flatten[sel_i]

        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)
            times_sel = times_sel.to(device)
            target_alpha_inv_last = target_alpha_inv_last.to(device)

        # volume rendering
        render_result = model(rays_o, rays_d, viewdirs, times_sel, global_step=global_step, **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none = True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
  
        psnr = utils.mse2psnr(loss.detach())
        
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss

        if cfg_train.weight_mask_loss > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6).unsqueeze(-1)
            mask_loss = cfg_train.weight_mask_loss * F.binary_cross_entropy(pout, target_alpha_inv_last)
            mask_loss_lst.append(mask_loss.item())
            loss += mask_loss

        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss

        if cfg_train.weight_distortion > 0:
            n_max = render_result['n_max']
            s = render_result['s']
            w = render_result['weights']
            ray_id = render_result['ray_id']
            loss_distortion = cfg_train.weight_distortion * flatten_eff_distloss(w, s, 1/n_max, ray_id)
            dist_loss_lst.append(loss_distortion.item())
            loss +=  loss_distortion

        loss.backward()

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_feature>0:
                model.feature_total_variation_add_grad(
                    cfg_train.weight_tv_feature/len(rays_o), global_step<cfg_train.tv_feature_before)
        optimizer.step()
        psnr_lst.append(psnr.item())
        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step%args.i_print == 0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction : iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Dist: {np.mean(dist_loss_lst):.9f} / Inv: {np.mean(inv_loss_lst):.9f} / '
                       f'Delta: {np.mean(delta_loss_lst):.9f} / Mask Loss: {np.mean(mask_loss_lst):.9f} / Eps: {eps_time_str}')
            psnr_lst = []
            dist_loss_lst = []
            inv_loss_lst = []
            delta_loss_lst = []
            mask_loss_lst = []

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
        }, last_ckpt_path)
        print('scene_rep_reconstruction : saved checkpoints at', last_ckpt_path)

def train(args, cfg, read_path, save_path, data_dict=None, stages=[1,2]):
    # init
    print('train: start')
    tensorboard_path = os.path.join("./logs/tensorboard", save_path)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(save_path, 'config.py'))
    
    # Bouding box search based on camera frustrums
    xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args = args, cfg = cfg, **data_dict)
    
    if 1 in stages:
        # fine detail reconstruction
        eps_time = time.time()
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.model_and_render, cfg_train=cfg.train_config,
                xyz_min=xyz_min, xyz_max=xyz_max,
                data_dict=data_dict)
        eps_loop = time.time() - eps_time
        eps_time_str = f'{eps_loop//3600:02.0f}:{eps_loop//60%60:02.0f}:{eps_loop%60:02.0f}'
        print('train: finish (eps time', eps_time_str, ')')
    
    if 2 in stages:
        # Export point clouds
        ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        model = utils.load_model(tineuvox.TiNeuVox, ckpt_path).to(device)
        stepsize = cfg.model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'inverse_y': cfg.data.inverse_y, 
            'flip_x': cfg.data.flip_x, 
            'flip_y': cfg.data.flip_y,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': cfg.pcd_train_config.bg_col,
                'stepsize': stepsize,
                'render_depth': True,
            },
        }
        bone_length = cfg.pcd_model_and_render.bone_length
        canonical_pcd_num = cfg.pcd_model_and_render.canonical_pcd_num
        pcd_density_threshold = cfg.pcd_model_and_render.pcd_density_threshold
        skeleton_density_threshold = cfg.pcd_model_and_render.skeleton_density_threshold

        # determine actual canonical t
        unique_times = torch.unique(data_dict['times'])
        canonical_t_indx = torch.argmin(((unique_times - cfg.data.canonical_t)**2).sqrt()).long()
        canonical_t = unique_times[canonical_t_indx].item()
        export_point_cloud(model, data_dict, read_path, render_viewpoints_kwargs, canonical_t, pcd_density_threshold, 
            export='both', bone_length=bone_length, canonical_pcd_num=canonical_pcd_num, skeleton_density_threshold=skeleton_density_threshold)

        torch.cuda.empty_cache()
        # train point cloud reconstruction
        eps_time = time.time()
        train_pcd(
            args=args, cfg=cfg, 
            cfg_model=cfg.pcd_model_and_render, cfg_train=cfg.pcd_train_config, 
            read_path=read_path,save_path=save_path, data_dict=data_dict, tineuvox_model=model,
            canonical_t=canonical_t, tensorboard_path=tensorboard_path)
        eps_loop = time.time() - eps_time
        eps_time_str = f'{eps_loop//3600:02.0f}:{eps_loop//60%60:02.0f}:{eps_loop%60:02.0f}'
        print('train: finish (eps time', eps_time_str, ')')

def export_point_cloud(model, data_dict, path, render_viewpoints_kwargs, canonical_t=0., threshold=0.2, export='torch', bone_length=4., canonical_pcd_num=3e+4, skeleton_density_threshold=0.2):
    import open3d as o3d

    folder_path = os.path.join(path, 'pcds')
    os.makedirs(folder_path, exist_ok=True)

    if os.path.isfile(os.path.join(folder_path, 'canonical.tar')) and os.path.isfile(os.path.join(folder_path, 'skeleton.tar')):
        print('PCD and skeleton already exists, skipping export.')
        return 

    def save_pcd(pcd, rgbs, feat, raw_feat, alphas, pcd_path, t, export_torch, xyz_min=None, xyz_max=None, voxel_size=None):
        if export_torch:
            torch.save({
                'pcd': pcd,
                'rgbs': rgbs,
                'feat': feat,
                'raw_feat': raw_feat,
                'alphas': alphas,
                't': t.item(),
                'xyz_min': xyz_min,
                'xyz_max': xyz_max,
                'voxel_size': voxel_size
            }, pcd_path)
        else:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd.cpu().numpy())
            pcd_o3d.colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
            o3d.io.write_point_cloud(pcd_path, pcd_o3d)

    render_poses = data_dict['poses'][data_dict['img_to_cam'][data_dict['i_train']]].float()
    Ks = data_dict['Ks'][data_dict['img_to_cam'][data_dict['i_train']]]
    HW = data_dict['HW'][data_dict['i_train']]
    times = data_dict['times'][data_dict['i_train']].float()

    sorted_indices = torch.argsort(times).cpu()
    render_poses = render_poses[sorted_indices]
    HW = HW[sorted_indices]
    Ks = Ks[sorted_indices]
    times = times[sorted_indices]
    t = torch.tensor([canonical_t])

    path_dict = {}

    xyz_min = torch.tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    freq = 1
    freq_up = None
    freq_low = None

    sigma_pcd = 0
    sigma_skeleton = 0

    def preprocess_volume(alpha_volume, threshold, sigma=1):
        if sigma > 0:
            alpha_volume = filters.gaussian(alpha_volume, sigma=sigma, preserve_range=True)
        binary_volume = alpha_volume > threshold
        binary_volume = remove_small_holes(binary_volume.astype(bool), area_threshold=2**8,)
        binary_volume = largest_k(binary_volume, connectivity=26, k=1).astype(int)
        
        return binary_volume.astype(bool)

    # Get render parameters and get point cloud
    c2w = render_poses[0]
    H, W = HW[0]
    K = Ks[0]
    _, _, viewdirs = tineuvox.get_rays_of_a_view(
            H, W, K, c2w, render_viewpoints_kwargs['ndc'], 
            inverse_y=render_viewpoints_kwargs['inverse_y'], flip_x=render_viewpoints_kwargs['flip_x'], flip_y=render_viewpoints_kwargs['flip_y'])
    viewdir = viewdirs.mean(dim=0).mean(dim=0).reshape((1, 3))
    stepsize = render_viewpoints_kwargs['render_kwargs']['stepsize']

    _, _, _, _, feat, raw_feat, grid_xyz, alpha_volume = model.get_grid_as_point_cloud(
        stepsize=stepsize, time_sel=t, viewdir=viewdir, threshold=threshold, sampling_freq=freq, N_batch=2**21, alpha_xyz_only=True)
    mask = preprocess_volume(alpha_volume.cpu().numpy(), threshold, sigma=sigma_pcd)
    points = grid_xyz[mask]

    if len(points) > canonical_pcd_num:
        freq_up = freq
        op = lambda x: x - 0.1
    elif len(points) < canonical_pcd_num:
        freq_low = freq
        op = lambda x: x + 0.1
    else:
        pass # nice
    
    while freq_up is None or freq_low is None:
        freq = op(freq)
        _, _, _, _, feat, raw_feat, grid_xyz, alpha_volume = model.get_grid_as_point_cloud(
            stepsize=stepsize, time_sel=t, viewdir=viewdir, threshold=threshold, sampling_freq=freq, N_batch=2**21, alpha_xyz_only=True)
        mask = preprocess_volume(alpha_volume.cpu().numpy(), threshold, sigma=sigma_pcd)
        points = grid_xyz[mask]
        
        if len(points) > canonical_pcd_num:
            freq_up = freq
        elif len(points) < canonical_pcd_num:
            freq_low = freq
        else:
            break

    for i in range(0, 10, 1):
        freq = (freq_up + freq_low) / 2
        _, _, _, _, feat, raw_feat, grid_xyz, alpha_volume = model.get_grid_as_point_cloud(stepsize=stepsize, time_sel=t, viewdir=viewdir, threshold=threshold, sampling_freq=freq, N_batch=2**21, alpha_xyz_only=True)
        mask = preprocess_volume(alpha_volume.cpu().numpy(), threshold, sigma=sigma_pcd)
        points = grid_xyz[mask]
        print(f"Canonical sampling freq: {freq}, num points: {len(points)}")
        if len(points) > canonical_pcd_num:
            freq_up = freq
        elif len(points) < canonical_pcd_num:
            freq_low = freq
        else:
            break
        
    points, alphas, rgbs, feat, raw_feat, binary_volume, grid_xyz, _ = model.get_grid_as_point_cloud(
        stepsize=stepsize, time_sel=t, viewdir=viewdir, threshold=threshold, sampling_freq=freq, N_batch=2**21, alpha_xyz_only=False, grid_xyz=grid_xyz[mask,:])
    
    # Save data
    pcd_torch_path  = os.path.join(folder_path, f'canonical.tar')
    xyz_min = torch.minimum(xyz_min, points.min(dim=0)[0])
    xyz_max = torch.maximum(xyz_max, points.max(dim=0)[0])
    
    save_pcd(points, rgbs, feat, raw_feat, alphas, pcd_torch_path, t, export_torch=(export=='torch' or export=='both'), xyz_min=xyz_min, xyz_max=xyz_max, voxel_size=model.voxel_size)
    path_dict[t.item()] = pcd_torch_path
    grid_xyz = grid_xyz.cpu().numpy()

    # Find min & max for bbox
    pcd_path  = os.path.join(folder_path, f'canonical.pcd')
    save_pcd(points, rgbs, feat, raw_feat, alphas, pcd_path, t, export_torch= not (export=='o3d' or export=='both'))

    # create linspace 3d grid based on min and max which are 3 dimensional
    grid_xyz = model.get_grid_xyz(freq)
    grid_xyz = grid_xyz.cpu().numpy()

    binary_volume = preprocess_volume(alpha_volume.cpu().numpy(), skeleton_density_threshold, sigma=sigma_skeleton) # smooth volume for better skeleton extraction
    
    smpl_skeleton = False
    if smpl_skeleton:
        from zju_skeletons import bones as zju_bones
        from zju_skeletons import joints as zju_joints
        zju_num = path.split('_')[-1] # NOTE: Experiment name muss contain _NUM at the end to work
        joints = zju_joints[zju_num]
        bones = zju_bones[1:]

        res = {
            'skeleton_pcd': joints,
            'joints': joints,
            'root': joints[0],
            'bones': bones,
            'pcd': None,
            'weights': None,
            'binary_volume': None
        }
    else:
        res = create_skeleton(binary_volume, grid_xyz, bone_length=bone_length)
    pcd_torch_path = os.path.join(folder_path, f'skeleton.tar')
    torch.save(res, pcd_torch_path)
    print(f"{len(res['bones'])} bones extracted.")

    skel_pcd = torch.tensor(res['skeleton_pcd'])
    skel_path  = os.path.join(folder_path, f'skeleton.pcd')
    save_pcd(skel_pcd, torch.zeros_like(skel_pcd), None, None, None, skel_path, t, export_torch= False)

if __name__=='__main__':
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()
    data_dict = None

    # load images / poses / camera settings / data split
    data_dict = load_everything(args = args, cfg = cfg, use_cache=args.use_cache, overwrite=args.overwrite_cache)
    read_path = os.path.join(cfg.basedir, cfg.expname)
    save_path = read_path

    # train
    if not args.render_only:
        if args.first_stage_only:
            stages = [1]
        elif args.second_stage_only:
            stages = [2]
        else:
            stages = [1,2]
        train(args, cfg, read_path, save_path, data_dict = data_dict, stages=stages)

    # load model for rendring
    if args.render_test or args.render_video or args.repose_pcd or args.visualise_canonical:
        cfg.basedir += args.basedir_append_suffix
        if not args.render_pcd:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
            model_class = tineuvox.TiNeuVox
        else:
            ckpt_path = os.path.join(save_path, 'temporalpoints_last.tar')
            model_class = temporalpoints.TemporalPoints
        
        model = utils.load_model(model_class, ckpt_path).to(device)
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        near=data_dict['near']
        far=data_dict['far']
        stepsize = cfg.model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'inverse_y': cfg.data.inverse_y, 
            'flip_x': cfg.data.flip_x, 
            'flip_y': cfg.data.flip_y,
            'render_kwargs': {
                'near': near,
                'far': far,
                'bg': cfg.train_config.bg_col,
                'stepsize': stepsize,
                'render_depth': True,
                'inverse_y': cfg.data.inverse_y,
            },
        }

        if args.degree_threshold > 0:
            times = data_dict['times'].unique().unsqueeze(-1)
            joints, bones, new_joints, new_bones, prune_bones, _, _, res = model.simplify_skeleton(
                times, 
                deg_threshold=args.degree_threshold, 
                five_percent_heuristic=True,
                visualise_canonical=args.visualise_canonical) # If visualise canonical, we will overwrite the bones
        else:
            prune_bones = torch.tensor([])

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(save_path, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)

        # save threshold and static joints in txt file
        with open(os.path.join(testsavedir, 'threshold.txt'), 'w') as f:
            f.write(f'{args.degree_threshold}\n')
            f.write(f'Static joints: {prune_bones.sum()} / {len(prune_bones)}')

        rgbs, disps, _, _ = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['img_to_cam'][data_dict['i_test']]],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir,
                test_times=data_dict['times'][data_dict['i_test']],
                eval_psnr=args.eval_psnr,eval_ssim = args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)

        imageio.mimwrite(os.path.join(testsavedir, 'test_video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'test_video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)
        
    # render video
    if args.render_video:
        testsavedir = os.path.join(save_path, f'render_video_{ckpt_name}_time')
        os.makedirs(testsavedir, exist_ok=True)

        rgbs, disps, weights, flows = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][0][None,...].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][0][None,...].repeat_interleave(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                test_times=data_dict['render_times'],
                render_pcd_direct=args.render_pcd_direct,
                **render_viewpoints_kwargs)
        
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality =8)
        if len(weights) > 0:
            imageio.mimwrite(os.path.join(testsavedir, 'video.weights.mp4'), utils.to8b(weights), fps=30, quality=8)

    if args.repose_pcd:
        model = render_viewpoints_kwargs['model']
        bones = model.bones
        joints = model.joints.detach().cpu().numpy()
        weights = model.get_weights().detach().cpu().numpy()

        start_scale = 0
        end_scale = 1
        steps = 30
        target_params = torch.randn((len(joints), 4)) * 0.2
        target_params[0] = torch.tensor([0., 0., 0., 0.])

        # NOTE: Here you manually set the bone rotations for animation.
        # You can check the bones via passing --visualise_canonical
        # In this example below we rotate bone num 25 around axis 1,0,0 by -0.3 radians
        # target_params = torch.zeros((len(joints), 4))
        # target_params[25] = torch.tensor([1., 0., 0., -0.3])
        
        target_params = target_params[None] * torch.linspace(start_scale, end_scale, steps)[:,None,None]

        # Create looping animation by inversing the parameters
        target_params = torch.concat([target_params, target_params.flip(0)], dim=0)
        steps = 2 * steps

        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_repose_{args.seed}')
        os.makedirs(testsavedir, exist_ok=True)
        pose = data_dict['poses'][3]

        rgbs, disps, weights = render_repose(
            render_poses=data_dict['poses'][0].repeat(steps, 1, 1),
            HW=data_dict['HW'][0][None,...].repeat(steps, 0),
            Ks=data_dict['Ks'][0][None,...].repeat_interleave(steps, 0),
            render_factor=args.render_video_factor,
            savedir=testsavedir,
            rot_params=target_params,
            eval_psnr=args.eval_psnr, eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
            **render_viewpoints_kwargs)

        imageio.mimwrite(os.path.join(testsavedir, 'train_video.rgb.mp4'), utils.to8b(rgbs), fps = 30, quality = 8)
        imageio.mimwrite(os.path.join(testsavedir, 'train_video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps = 30, quality = 8)
        if len(weights) > 0:
            imageio.mimwrite(os.path.join(testsavedir, 'video.weights.mp4'), utils.to8b(weights), fps=30, quality=8)

    if args.visualise_canonical:
        save_path = os.path.join(cfg.basedir, cfg.expname, 'canonical.pcd')

        threshold = args.degree_threshold

        skeleton_points = model.skeleton_pcd.detach().cpu().numpy()
        canonical_pcd = model.canonical_pcd.detach().cpu().numpy()
        weights = model.get_weights().detach().cpu().numpy()
        root = model.joints[0].detach().cpu().numpy()
        try:
            old_joints = joints
            old_bones = bones
        except:
            new_joints = model.joints.detach().cpu().numpy()
            new_bones = model.bones
            old_joints, old_bones = None, None
        
        visualise_skeletonizer(skeleton_points, root, new_joints, new_bones, canonical_pcd, weights, old_joints=old_joints, old_bones=old_bones)