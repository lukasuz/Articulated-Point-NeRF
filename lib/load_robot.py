import json
from typing import List
import torch
import imageio 
import numpy as np
from tqdm import tqdm
import cv2

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

def data_settings(robot_name: str, test=False):
    if robot_name == "nao":
        coordinate_scale = 0.333
    else:
        coordinate_scale = 1.
    if test:
        chosen_camera_id = [0, 10]
        test_camera_id = [0, 10]
    else:
        chosen_camera_id = list(range(1, 10)) + list(range(11, 20))
        test_camera_id = []

    return chosen_camera_id, test_camera_id, coordinate_scale

# chosen_camera_id = [0, 2, 5, 13, 18]
def load_robot(data_dir, video_len = 300, size: int = 512, test=False, skip_images=False, step=1):
    """
    Args:
        video_len:
        data_dir:
    Returns:
    """

    robot_name = data_dir.split('/')[-1]
    chosen_camera_id, test_camera_id, coordinate_scale = data_settings(robot_name, test)
    
    imgs = None
    masks = None
    times = []
    img_to_cam = []
    c = 0
    i_train = []
    i_test = []
    for f_id in tqdm(range(0, video_len, step)):
        for i, c_id in enumerate(chosen_camera_id):

            times.append(f_id / (video_len-1))
            img_path = f"{data_dir}/frame_{f_id:0>5}_cam_{c_id:0>3}.png"
            config_path = f"{data_dir}/cam_{c_id:0>3}.json"

            if skip_images and f_id > 0:
                pass
            else:
                img = imageio.imread(img_path)
                img_scale = 1

                if img.shape[0] != size:
                    img_scale = size / img.shape[0]
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                
                mask = img[...,-1:]
                mask_float = mask.astype(np.float32) / 255

                img = (img[...,:3].astype(np.float32) * mask_float) + (255. - mask)
  
            if imgs is None:
                imgs = torch.zeros((video_len * len(chosen_camera_id), *img.shape), dtype=torch.uint8, device = 'cpu')
                masks = torch.zeros((video_len * len(chosen_camera_id), img.shape[0], img.shape[1], 1), dtype=torch.uint8, device = 'cpu')
            imgs[c] = torch.tensor(img, dtype=torch.uint8)
            masks[c] = torch.tensor(mask, dtype=torch.uint8)

            if c_id in test_camera_id:
                i_test.append(c)
            else:
                i_train.append(c)
            img_to_cam.append(i)
            c += 1

    intrinsics = []
    poses = []
    for c_id in chosen_camera_id:

            config_path = f"{data_dir}/cam_{c_id:0>3}.json"
            with open(config_path, "r") as f:
                config = json.load(f)

            intrinsic_config = config["camera_data"]["intrinsics"]
            intrinsic = np.zeros((3, 3), dtype="float32")
            intrinsic[0, 0] = intrinsic_config['fx'] * img_scale
            intrinsic[1, 1] = intrinsic_config['fy'] * img_scale
            intrinsic[0, 2] = intrinsic_config['cx'] * img_scale
            intrinsic[1, 2] = intrinsic_config['cy'] * img_scale
            intrinsic[2, 2] = 1
            intrinsics.append(intrinsic)

            extrinsic = np.array(config["camera_data"]["camera_view_matrix"]).T
            extrinsic[:3, -1] = extrinsic[:3, -1] / coordinate_scale
            pose = np.linalg.inv(extrinsic)
            poses.append(pose)

    if skip_images:
        imgs = imgs[0][None].repeat(len(imgs), axis=0)

    poses = np.array(poses)
    intrinsics = np.array(intrinsics)
    times = np.array(times, dtype=np.float32)
    img_to_cam = np.array(img_to_cam)

    H, W = imgs.shape[1], imgs.shape[2]
    radius = np.sqrt((poses[:,:,-1]**2).sum(-1)).mean()
    render_poses = torch.stack([pose_spherical(angle, -20.0, radius) for angle in np.linspace(0, 360, 180+1)[:-1]], 0)
    render_times = torch.linspace(0., 1., render_poses.shape[0])

    render_intrinsics = np.repeat(np.expand_dims(intrinsics[0], 0), render_poses.shape[0], 0)

    i_split = [np.arange(len(i_train)), np.array([]), np.array(i_test)]

    return imgs, poses, intrinsics, times, render_poses, render_times, render_intrinsics, [H, W], i_split, img_to_cam, masks
