import json
from typing import List
import torch
import imageio 
import numpy as np
from tqdm import tqdm
import cv2
import blosc
import pickle

coord_change_mat = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def rodrigues_mat_to_rot(R):
  eps =1e-16
  trc = np.trace(R)
  trc2 = (trc - 1.)/ 2.
  #sinacostrc2 = np.sqrt(1 - trc2 * trc2)
  s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
  if (1 - trc2 * trc2) >= eps:
    tHeta = np.arccos(trc2)
    tHetaf = tHeta / (2 * (np.sin(tHeta)))
  else:
    tHeta = np.real(np.arccos(trc2))
    tHetaf = 0.5 / (1 - tHeta / 6)
  omega = tHetaf * s
  return omega

def rodrigues_rot_to_mat(r):
  wx,wy,wz = r
  theta = np.sqrt(wx * wx + wy * wy + wz * wz)
  a = np.cos(theta)
  b = (1 - np.cos(theta)) / (theta*theta)
  c = np.sin(theta) / theta
  R = np.zeros([3,3])
  R[0, 0] = a + b * (wx * wx)
  R[0, 1] = b * wx * wy - c * wz
  R[0, 2] = b * wx * wz + c * wy
  R[1, 0] = b * wx * wy + c * wz
  R[1, 1] = a + b * (wy * wy)
  R[1, 2] = b * wy * wz - c * wx
  R[2, 0] = b * wx * wz - c * wy
  R[2, 1] = b * wz * wy + c * wx
  R[2, 2] = a + b * (wz * wz)
  return R

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(coord_change_mat) @ c2w
    return c2w

# chosen_camera_id = [0, 2, 5, 13, 18]
def load_zju(pickle_path, video_len = 300, size: int = 512, compression=True, bg_col=0, step=1,load_test_val=False, overwrite_views_with_train=None):
    """
    Args:
        video_len:
        data_dir:
    Returns:
    """
    if load_test_val:
        pickle_path =  pickle_path.replace('cache_train', 'cache_test')
    
    file = open(pickle_path,'rb')
    data = pickle.load(file)
    file.close()

    imgs = []
    masks = []
    intrinsics = []
    poses = []
    times = []
    img_to_cam = []
    embeddings = []
    imgs_per_cam = len(np.unique(data['frame_id']))
    id_max = video_len - 1
    video_len = min(len(np.unique(data['frame_id'])) - 1, video_len)
        
    # Load images
    counter = 0
    unique_cams = np.unique(data['camera_id'])
    for id in tqdm(range(0, video_len, step)):
        
        ids = []
        for k, c_id in enumerate(unique_cams):
            f_id = c_id * imgs_per_cam + id
            times.append(data['frame_id'][id] / (id_max-1))
            img = data["img"][f_id]
            fg_mask = data["mask"][f_id]
            if compression:
                img = blosc.unpack_array(img)
                fg_mask = blosc.unpack_array(fg_mask)[None,:,:]

            # img = img / 255.
            img = img * fg_mask + (1 - fg_mask) * bg_col * 255

            img = np.swapaxes(img, 0, -1)
            img = np.swapaxes(img, 0, 1)

            img_scale = 1
            if img.shape[0] != size:
                img_scale = size / img.shape[0]
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

            img = torch.tensor(img, device = 'cpu', dtype=torch.uint8)[None]
            imgs.append(img)
            masks.append(torch.tensor(fg_mask[...,None], device = 'cpu'))
            img_to_cam.append(c_id)

            if len(poses) < len(unique_cams):
                intrinsic = data["camera_intrinsic"][f_id] * img_scale
                intrinsic[2,2] = 1.
                intrinsics.append(intrinsic)

                coordinate_scale = 1.5 # NOTE: From WIM
                if overwrite_views_with_train is not None:
                    rot = train_data['camera_rotation'][overwrite_views_with_train]
                    trans = data["camera_translation"][overwrite_views_with_train] / coordinate_scale
                else:
                    rot = data["camera_rotation"][f_id]
                    trans = data["camera_translation"][f_id] / coordinate_scale
                pose = np.concatenate([np.concatenate([rot, trans], axis=-1), np.array([[0,0,0,1]])], axis=0)
                pose = np.linalg.inv(pose)
                poses.append(pose)
            
            ids.append(counter)
            counter += 1
        
        ids = []

    if overwrite_views_with_train is not None:
        del train_data

    imgs = torch.cat(imgs, 0)
    masks = torch.cat(masks, 0)
    # imgs = torch.tensor(np.array(imgs), device = 'cpu').float()
    poses = np.array(poses)
    intrinsics = np.array(intrinsics)
    times = np.array(times, dtype=np.float32)

    H, W = imgs.shape[1], imgs.shape[2]
    render_poses = torch.tensor(poses[3]).unsqueeze(0).expand(400,-1,-1).float()

    radius = 2.5
    y_offset = np.mean(poses[:,2,3])
    
    render_poses = torch.stack([pose_spherical(angle, -20.0, radius) for angle in np.linspace(180,-180, 40+1)[:-1]], 0)
    render_poses[:,2,3] += y_offset
    # offset = torch.tensor([0,0,0])
    # render_poses[:,:3,-1] += offset

    render_times = torch.linspace(0., 1., render_poses.shape[0])

    render_intrinsics = np.repeat(np.expand_dims(intrinsics[0], 0), render_poses.shape[0], 0)

    if load_test_val:
        i_split = [np.array([]), np.array([]), np.arange(len(imgs))] # No validation in ZJU, NOTE: paths are explicitly changed above
    else:
        i_split = [np.arange(len(imgs)), np.array([]), np.array([])]
    img_to_cam = np.array(img_to_cam, dtype=np.int32)
    # img_to_cam = np.array(list(range(len(imgs))))
    return imgs, poses, intrinsics, times, render_poses, render_times, render_intrinsics, [H, W], i_split, img_to_cam, masks, embeddings
