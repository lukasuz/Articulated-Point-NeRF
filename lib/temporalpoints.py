import torch
from .tineuvox import poc_fre
from pykeops.torch import LazyTensor
import os
from torch_scatter import segment_coo
from torch.utils.cpp_extension import load
from .tineuvox import Alphas2Weights
import roma
import numpy as np
import roma
from seaborn import color_palette
from .treeprune import merge_joints
from .pointwarper import PointWarper
from .tineuvox import get_rays_of_a_view

from .utils import project_point_to_image_plane

parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

class NoPointsException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class TemporalPoints(torch.nn.Module):
    def __init__(self,
            canonical_pcd,
            canonical_alpha,
            canonical_feat,
            canonical_rgbs,
            skeleton_pcd,
            joints,
            bones,
            xyz_min,
            xyz_max,
            tineuvox,
            neighbours=8,
            timebase_pe=8,
            eps=1e-6,
            stepsize=None,
            voxel_size=None,
            fast_color_thres=0,
            embedding='full',
            frozen_view_dir=None,
            over_parameterized_rot=True,
            re_init_feat=False,
            re_init_mlps=False,
            feat_depth = 4,
            pose_embedding_dim=0,
            **kwargs):
        super(TemporalPoints, self).__init__()
        
        self.canonical_pcd = canonical_pcd
        self.skeleton_pcd = skeleton_pcd
        self.bones = bones
        self.bone_arap_mask = torch.tensor(bones).reshape(-1)
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.eps = torch.tensor(eps)

        self.feat_depth = feat_depth
        self.timebase_pe = timebase_pe
        self.t_dim = 1 + self.timebase_pe * 2
        self.stepsize = stepsize
        self.voxel_size = voxel_size
        self.fast_color_thres = fast_color_thres
        self.embedding = embedding
        self.over_parameterized_rot = over_parameterized_rot
        self.joints_to_keep = None
        self.forward_warp_t_dim = self.t_dim

        self.weights = torch.nn.Parameter(self._weights_from_bones(joints, bones, canonical_pcd, add_noise=True, noise_var=0, soft_weights=True, add_zero_weight=True), requires_grad=True)
        self.forward_warp = PointWarper(canonical_pcd=canonical_pcd, t_dim=self.forward_warp_t_dim, joints=joints, bones=bones, over_parameterized_rot=over_parameterized_rot)
        self.original_joints = torch.nn.Parameter(joints.to(torch.float32), requires_grad=False)
        self.joints = torch.nn.Parameter(joints.to(torch.float32), requires_grad=True)
        self.canonical_feat = torch.nn.Parameter(canonical_feat, requires_grad=True)
        if re_init_feat:
            self.canonical_feat.data = torch.randn_like(self.canonical_feat)
        # self.theta = torch.nn.Parameter(torch.tensor([0.001]), requires_grad=True)
        self.theta_weight = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.merging_dict = None
        self.merging_mat = None

        gammas = torch.ones(len(self.canonical_pcd))
        self.gammas = torch.nn.Parameter(gammas + torch.randn_like(gammas) * 1e-2, requires_grad=True)

        self.pruned_joints = torch.zeros(len(self.joints), dtype=bool)

        self.register_buffer('flat_merging_rules', torch.arange(0, len(self.joints)))
        self.register_buffer('sibling_merging_rules', torch.zeros(len(self.joints), dtype=bool))

        # Only for direct point cloud rendering
        self.canonical_rgbs = torch.nn.Parameter(canonical_rgbs, requires_grad=True)
        self.canonical_alpha = torch.nn.Parameter(canonical_alpha, requires_grad=True)
        self.direct_eps = torch.nn.Parameter(torch.tensor([0.05] * len(self.canonical_alpha)), requires_grad=True)

        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))

        # Define neighbourhood for canonical points and nn distances
        self.neighbours = neighbours
        xyz1 = LazyTensor(self.canonical_pcd[:, None, :])
        xyz2 = LazyTensor(self.canonical_pcd[None, :, :])
        D_ij = ((xyz1 - xyz2) ** 2).sum(-1)
        self.nn_i = D_ij.argKmin(dim=1, K=self.neighbours)
        self.nn_distance = torch.sqrt(((self.canonical_pcd[:,None,:] - self.canonical_pcd[self.nn_i,:])**2).sum(-1) + self.eps) # distance to nearest neighbour per point
        self.mean_min_distance = self.nn_distance[:,1].mean()

        # Define joint neighbourhood distances
        self.og_joint_distance = (self.original_joints[self.bone_arap_mask][0::2,:] 
                                - self.original_joints[self.bone_arap_mask][1::2,:])
        
        # Define feature net (as in PointNerf)
        feat_input_dim = self.canonical_feat.shape[-1] + 3 + 3 * tineuvox.posbase_pe * 2
        feat_input_dim += pose_embedding_dim # pose shape embedding
        feat_ouput_dim = self.canonical_feat.shape[-1]
        feat_width = self.canonical_feat.shape[-1]
        
        self.feat_net = torch.nn.Sequential(
            torch.nn.Linear(feat_input_dim, feat_width), torch.nn.LeakyReLU(inplace=True),
            *[
                torch.nn.Sequential(torch.nn.Linear(feat_width, feat_width), torch.nn.LeakyReLU(inplace=True))
                for _ in range(feat_depth-2)
            ],
            torch.nn.Linear(feat_width, feat_ouput_dim), torch.nn.LeakyReLU(inplace=True)
            )

        # Get rgb net and density from tineuvox
        self.rgbnet = tineuvox.rgbnet
        self.densitynet = tineuvox.densitynet
        self.timenet = tineuvox.timenet

        def weight_reset(m):
            try:
                m.reset_parameters()
            except Exception as e:
                print(e)

        if re_init_mlps:
            self.rgbnet.apply(weight_reset)
            self.densitynet.apply(weight_reset)
            self.timenet.apply(weight_reset)

        self.view_poc = tineuvox.view_poc # N
        self.time_poc = tineuvox.time_poc # N
        self.pos_poc = tineuvox.pos_poc
        self.no_view_dir = tineuvox.no_view_dir # N, viewdir by default
        self.tineuvox = tineuvox
        self.register_buffer('xyz_max_canonical', self.canonical_pcd.max(dim=0)[0])
        self.register_buffer('xyz_min_canonical',  self.canonical_pcd.min(dim=0)[0])
        self.frozen_view_dir = frozen_view_dir

        if frozen_view_dir is not None:
            viewdirs_emb = poc_fre(frozen_view_dir, self.view_poc)
            self.viewdirs_emb = torch.nn.Parameter(viewdirs_emb[None], requires_grad=False)

        self.pose_embedding_dim = pose_embedding_dim
        if pose_embedding_dim > 0:
            input_pose_embedding = len(self.joints) * (3 * len(self.pos_poc) * 2 + 3)
            self.pose_embedding_net = torch.nn.Sequential(
                torch.nn.Linear(input_pose_embedding, input_pose_embedding // 2), torch.nn.LeakyReLU(inplace=True),
                *[
                    torch.nn.Sequential(torch.nn.Linear(input_pose_embedding // 2, input_pose_embedding // 2), torch.nn.LeakyReLU(inplace=True))
                    for _ in range(feat_depth-2)
                ],
                torch.nn.Linear(input_pose_embedding // 2, pose_embedding_dim), torch.nn.LeakyReLU(inplace=True)
                )
        
        self.beta = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.beta_min = torch.nn.Parameter(torch.tensor([0.0001]), requires_grad=False)

    def get_kwargs(self):
        return {
            'canonical_pcd': self.canonical_pcd,
            'skeleton_pcd': self.skeleton_pcd,
            'canonical_alpha': self.canonical_alpha,
            'canonical_feat': self.canonical_feat,
            'canonical_rgbs': self.canonical_rgbs,
            'joints': self.joints,
            'bones': self.bones,
            'neighbours': self.neighbours,
            'timebase_pe': self.timebase_pe,
            'eps': self.eps,
            'stepsize': self.stepsize,
            'weights': self.weights,
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'tineuvox': self.tineuvox,
            'voxel_size': self.voxel_size,
            'fast_color_thres': self.fast_color_thres,
            'embedding': self.embedding,
            'frozen_view_dir': self.frozen_view_dir,
            'over_parameterized_rot':self.over_parameterized_rot,
            'feat_depth': self.feat_depth,
            'pose_embedding_dim': self.pose_embedding_dim,
        }

    def reinitialise_weights(self):
        self.weights.data = self._weights_from_bones(self.joints, self.bones, self.canonical_pcd, add_noise=True, noise_var=0, soft_weights=True, add_zero_weight=True)
        self.theta_weight.data = torch.tensor([0.1])

    def dist_batch(self, p, a, b):
        assert len(a) == len(b), "Same batch size needed for a and b"

        p = p[None, :, :]
        s = b - a
        w = p - a[:, None, :]
        ps = (w * s[:, None, :]).sum(-1)
        res = torch.zeros((a.shape[0], p.shape[1]), dtype=p.dtype)

        # ps <= 0
        ps_smaller_mask = ps <= 0
        lower_mask = torch.where(ps_smaller_mask)
        res[lower_mask] += torch.norm(w[lower_mask], dim=-1)

        l2 = (s * s).sum(-1)
        # ps > 0 and ps >= l2
        ps_mask = ~ps_smaller_mask

        temp_mask_l2 = ps >= l2[:, None]
        upper_mask = torch.where(ps_mask & temp_mask_l2)
        res[upper_mask] += torch.norm(p[0][upper_mask[1]] - b[upper_mask[0]], dim=-1)

        # ps > 0 and ps < l2
        within_mask = torch.where(ps_mask & ~temp_mask_l2)
        res[within_mask] += torch.norm(
            p[0][within_mask[1]] - (a[within_mask[0]] + (ps[within_mask] / l2[within_mask[0]]).unsqueeze(-1) * s[within_mask[0]]), dim=-1)

        return res
    
    def _weights_from_bones(self, joints, bones, pcd, add_noise=False, noise_var=0, val=1, soft_weights=True, add_zero_weight=False):
        bone_distances = self.dist_batch(
            pcd,
            torch.cat([joints[bone[0]].unsqueeze(0) for bone in bones], dim=0),
            torch.cat([joints[bone[1]].unsqueeze(0) for bone in bones], dim=0))
            
        if soft_weights:
            weights = (1 / (0.5 * torch.e ** bone_distances + self.eps)).T.contiguous()
        else:
            bone_argmin = torch.argmin(bone_distances, axis=0)
            weights = torch.zeros((len(bone_argmin), len(bones)))
            weights[torch.arange(len(bone_argmin)), bone_argmin] = val
        
        if add_zero_weight:
            weights = torch.cat([torch.zeros((len(weights), 1)), weights], dim=-1)
        
        if add_noise:
            weights = weights + torch.randn_like(weights) * noise_var

        return weights
    
    def simplify_skeleton(self, times, deg_threshold=10, mass_threshold=0.0, update_skeleton=False, five_percent_heuristic=False, visualise_canonical=False):
        # Calculate time-dependent rotation angles
        times_embed = poc_fre(times, self.time_poc)
        params = self.forward_warp.transform_net(times_embed)

        if self.over_parameterized_rot:
            rot_angles = params[:, :len(self.joints), -1]
        else:
            rot_angles = (params[:, :len(self.joints), :3]**2).sum(-1).sqrt() % (2 * np.pi)

        rotations = torch.zeros((len(times), len(self.joints), 3, 3))
        for ji in range(len(self.joints)):
            if self.over_parameterized_rot:
                temp_rot, _ = self.forward_warp.Rodrigues(params[:, ji, :])
            else:
                temp_rot, _ = self.forward_warp.Rodrigues(params[:, ji, :3])
            rotations[:, ji, :] = temp_rot
        
        rotation_similarity_mat = torch.eye(len(self.joints)).type(torch.bool)
        for i in range(len(self.joints)):
            for j in range(len(self.joints)):
                if j >= i:
                    continue
                res = self._are_rotations_similar(
                    rotations[:,i,...], rotations[:,j,...], 
                    deg_threshold=deg_threshold, five_percent_heuristic=five_percent_heuristic)
                rotation_similarity_mat[i,j] = res
                rotation_similarity_mat[j,i] = res
        
        if five_percent_heuristic:
            # 5% heuristic
            th = int(len(times) * 0.05)
            res = (torch.rad2deg(rot_angles).abs() >= deg_threshold).sum(dim=0)
            zero_motion = res <= th
        else:
            # Avg heuristic
            deg_stds = torch.rad2deg((rot_angles**2).mean(dim=0)) # How much it differs from the canonical pose (zero rad)
            zero_motion = deg_stds <= deg_threshold

        # Create pruning masks based on rotation
        prune_bones = zero_motion
        prune_bones[0] = False # never prune (imaginary) root (bone)

        # Prune joints
        joints = self.joints.detach().cpu().numpy()
        bones = self.bones
        new_joints, new_bones, merging_rules, joints_to_keep, rotations_to_keep, rotation_switch_mask, sibling_transfer_rules = merge_joints(joints, bones, prune_bones, rotation_similarity_mat, convert_merging_rules=False)
        rotations_to_keep = torch.tensor(rotations_to_keep)
        self.merging_rules = merging_rules
        self.joints_to_keep = joints_to_keep
        self.new_bones = new_bones

        # if visualise_canonical:
        #     self.joints.data = torch.tensor(new_joints)
        #     self.bones = new_bones

        self.forward_warp.set_rotation_mask(~prune_bones)
        self.forward_warp.set_sibling_mask(torch.tensor(sibling_transfer_rules))

        # Solve for pruned weights
        # current_weights = self.get_weights()
        # target_weights = torch.zeros_like(current_weights)
        # for from_idx, to_idx in enumerate(merging_rules): 
        #     target_weights[:, to_idx] += current_weights[:, from_idx]
        
        # if update_skeleton:
        #     # Remove pruned weights
        #     target_weights = target_weights[:, rotations_to_keep]
        #     current_weights = self.weights[:, rotations_to_keep]

        # Merging Mat approach
        num_weights = self.weights.shape[-1]
        self.flat_merging_rules = torch.tensor(self.flatten_merging_rules(merging_rules))
        self.sibling_merging_rules = torch.tensor(sibling_transfer_rules)
        self.merging_mat = torch.zeros(num_weights, num_weights, num_weights)
        for i in range(num_weights):
            mask = (self.flat_merging_rules == i)
            self.merging_mat[i] = torch.eye(num_weights) * mask

        # for i in range(len(prune_bones)):
        #     if not prune_bones[i]:
        #         print(f"Joint {i} kept")
        
        print(f"Frozen joints/weights: {prune_bones.sum()} of {len(prune_bones)} ")
        print(f"Joints kept: {[i for i, v in enumerate(~prune_bones) if v]}")
        print(f"Actually pruned joints: {len(joints) - len(new_joints)} of {len(joints)}")

        return joints, bones, new_joints, new_bones, prune_bones, merging_rules, rotations_to_keep, res
    
    def flatten_merging_rules(self, merging_rules):
        endpoints = []
        for i in range(len(merging_rules)):
            new_indx = i
            while True:
                new_indx = merging_rules[new_indx]
                if new_indx == merging_rules[new_indx]:
                    endpoints.append(new_indx)
                    break
        return endpoints

    def _are_rotations_similar(self, rot1, rot2, deg_threshold=20, five_percent_heuristic=False):
        angle = torch.norm(
            roma.rotmat_to_rotvec(
                rot1 @ torch.transpose(rot2, 1, 2))
            , dim=-1)
        
        if not five_percent_heuristic:
            deg_std = torch.rad2deg(torch.sqrt((angle**2).mean(dim=0)))
            return deg_std <= deg_threshold
        else:
            th = int(len(rot1) * 0.05)
            res = torch.rad2deg(angle) >= deg_threshold #
            return res.sum(dim=0) <= th
    
    def repose(self, rot_params):
        return self.forward_warp(self.get_weights(), self.joints, rot_params=rot_params)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, xyz_min=None, xyz_max=None, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        if xyz_min is None:
            xyz_min = self.xyz_min
        if xyz_max is None:
            xyz_max = self.xyz_max
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, _, _, _ = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, xyz_min, xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]

        return ray_pts, ray_id, step_id, mask_inbbox
    
    def get_weights(self):
        theta_weight = torch.max(self.eps, self.theta_weight)
        weights = torch.softmax(self.weights / theta_weight, dim=-1).permute(1,0)
        
        if self.merging_mat == None:
            self.merging_mat = torch.zeros(len(weights), len(weights), len(weights))
            self.flat_merging_rules = torch.arange(0, len(self.joints))
            for i in range(weights.shape[0]):
                mask = (self.flat_merging_rules == i)
                self.merging_mat[i] = torch.eye(len(weights)) * mask

        merged_weights = torch.bmm(self.merging_mat, weights.unsqueeze(0).repeat(len(weights), 1, 1)).sum(1)
        
        return merged_weights.permute(1,0)

    def aggregate_pts(self, t_hat_pcd, rotated_frames, query_radius, cam_per_ray, render_pcd_direct, render_weights, render_kwargs, pose_embedding, calc_min_max=True):
        N = len(render_kwargs['rays_o'])
        render_neighbours = self.neighbours
        render_lbs_weights = None

        with torch.profiler.record_function("sample_ray"):
            assert render_kwargs is not None
            if calc_min_max:
                xyz_min, xyz_max = torch.min(t_hat_pcd, dim=0)[0] - query_radius, torch.max(t_hat_pcd, dim=0)[0] + query_radius
            else:
                xyz_min, xyz_max = None, None
            ray_pts, ray_id, step_id, _ = self.sample_ray(xyz_min=xyz_min, xyz_max=xyz_max, **render_kwargs)

        if len(ray_pts) == 0:
            raise NoPointsException("No points.")
        
        ## Find nearest neighbours
        with torch.profiler.record_function("knn"):
            ray_pts_lazy = LazyTensor(ray_pts[:, None, :].detach())
            shifted_xyz_lazy = LazyTensor(t_hat_pcd[None, :, :].detach())
            D_ij = ((ray_pts_lazy - shifted_xyz_lazy) ** 2).sum(-1)
            to_nn, s_i = D_ij.Kmin_argKmin(dim=1, K=render_neighbours)

        with torch.profiler.record_function("knn-post"):
            nn_mask = torch.where(to_nn[:,-1] <= query_radius)[0]                
            s_i = s_i[nn_mask, :]
            ray_id = ray_id[nn_mask]
            step_id = step_id[nn_mask]
            ray_pts = ray_pts[nn_mask]
            # to_nn = to_nn[nn_mask] # WARNING: Has no gradients!!!
            rel_p = ray_pts[:,None,:] - t_hat_pcd[s_i,:]
            to_nn = (rel_p**2).sum(-1)
        
        if len(s_i) == 0:
            raise NoPointsException("No points.")

        with torch.profiler.record_function("feat_net"):
            features_k = self.canonical_feat[s_i,:]
            rotated_frames_k = rotated_frames[s_i,:,:]
        
            ## Directly render point cloud based on frozen rgb and alpha
            rgbs_direct = None
            alpha_direct = None
            if render_pcd_direct:
                sig = self.mean_min_distance * torch.max(self.direct_eps, torch.tensor(0.))
                w_direct = torch.exp(-(to_nn**2) / (2 * (sig[s_i])**2 + 1e-12))
                w = 1 / (to_nn + self.eps)
                w_direct_density = (torch.tensor(1./render_neighbours) * w_direct).unsqueeze(-1)
                w_direct = w_direct / (w_direct.sum(dim=-1) + 1e-12)[:, None]
                w_direct = w_direct.unsqueeze(-1)
                alpha_k_direct = self.canonical_alpha.clip(0, 1)[s_i].unsqueeze(-1)
                rgbs_k_direct = self.canonical_rgbs.clip(0, 1)[s_i,:]
                
                rgbs_direct = (w_direct * rgbs_k_direct).sum(dim=1)
                alpha_direct = (w_direct_density * alpha_k_direct).sum(dim=1).squeeze(-1)

            ## POINT-NERF APPROACH
            w = 1 / (to_nn + self.eps)
            w = w / w.sum(dim=-1)[:, None]
            w = w.unsqueeze(-1)
            features_k_hat = []

            rotated_frames_flat = rotated_frames_k[...,:3,:3].reshape(-1,3,3)
            rel_p_flat = rel_p.reshape(-1, 3).unsqueeze(-1)
            rel_p_canonical = torch.bmm(rotated_frames_flat, rel_p_flat).squeeze(-1)
            rel_p_emb = poc_fre(rel_p_canonical, self.pos_poc)
            
            feat_input = [
                rel_p_emb,
                features_k.reshape(-1, features_k.shape[-1])
            ]
            if pose_embedding is not None:
                feat_input.append(pose_embedding.expand(len(rel_p_emb), -1))

            feat_input = torch.concat(feat_input, dim=-1)
            out = self.feat_net(feat_input)

            features_k_hat = out.reshape(*rotated_frames_k.shape[:2], out.shape[-1])
            h_feature = (features_k_hat * w).sum(dim=1)

            with torch.profiler.record_function("densitynet"):
                density_result = self.densitynet(h_feature).squeeze(-1)
                interval = render_kwargs['stepsize'] * self.tineuvox.voxel_size_ratio
                alpha = self.tineuvox.activate_density(density_result, interval)
                if alpha.ndim == 0:
                    alpha = alpha.unsqueeze(0) # Activate density can actually spew out scalars with ndim=0

            with torch.profiler.record_function("rgbnet"):
                if self.no_view_dir:
                    rgb_logit = self.rgbnet(h_feature)

                elif self.frozen_view_dir is not None:
                    viewdirs_emb_reshape = self.viewdirs_emb.expand(len(ray_id), -1)

                else:
                    viewdirs_emb = poc_fre(render_kwargs['viewdirs'], self.view_poc)
                    viewdirs_emb_reshape = viewdirs_emb[ray_id]
                
                rgb_logit = self.rgbnet(h_feature, viewdirs_emb_reshape)
                rgbs = torch.sigmoid(rgb_logit)

            if render_weights:
                render_lbs_weights = self._last_weights[s_i,:]
                render_lbs_weights = (render_lbs_weights * w).sum(dim=1)

            return rgbs, alpha, rgbs_direct, alpha_direct, render_lbs_weights, ray_pts, ray_id, step_id, N 

    def sample_thetas(self, tmin, tmax, num=50, reduction='five_percent', deg_threshold=15):
        ts = (tmax - tmin) * torch.rand(num) + tmin
        ts = torch.rand(num)[:, None]
        ts_embed = poc_fre(ts, self.time_poc)
        thetas = self.forward_warp.get_thetas(ts_embed)

        if reduction == 'five_percent':
            th = int(num * 0.05)
            res = torch.rad2deg(thetas) >= deg_threshold 
            thetas = res.sum(dim=0) <= th
        elif reduction == 'mean':
            thetas = thetas.mean(dim=0)
        else:
            raise NotImplementedError()
        
        return thetas

    def forward(self, t, render_depth=False, render_kwargs=None, query_radius=0.01,
                render_weights=False, rot_params=None, render_pcd_direct=False, poses=None, 
                Ks=None, cam_per_ray=None, calc_min_max=True, get_skeleton=False):
        assert (t is None) ^ (rot_params is None)

        # Forward-warp canonical pcd
        with torch.profiler.record_function("poc_fre"):
            if rot_params is None:
                t_embed = poc_fre(t, self.time_poc)
            else:
                t_embed = None

        with torch.profiler.record_function("forward_warp"):
            bones, joints, grid = None, None, None 

            weights = self.get_weights()
            self._last_weights = weights
            
            out = self.forward_warp(weights, 
                                    self.joints, 
                                    t_embed, 
                                    get_frames=True, 
                                    rot_params=rot_params, 
                                    get_skeleton=get_skeleton)
            
            if get_skeleton:
                t_hat_pcd, joints_rel, rotated_frames, joints, bones = out
            else:
                t_hat_pcd, joints_rel, rotated_frames = out
            rotated_frames = torch.inverse(rotated_frames)

            delta_joint = (self.joints - joints_rel).clone().detach()
            
            if self.pose_embedding_dim > 0:
                pose_embedding = self.pose_embedding_net(poc_fre(delta_joint, self.pos_poc).view(1, -1))
            else:
                pose_embedding = None
            
            if get_skeleton:
                joints = project_point_to_image_plane(joints, poses, Ks.to(torch.float32))

                if self.joints_to_keep is not None:
                    joints = joints[:, self.joints_to_keep]
                    bones = self.new_bones

        rgb_marched = None
        alphainv_last = None
        rgb_marched_direct = None
        ret_dict = {}

        try:
            # Kinematic warp and sample
            render_pcd_direct = True
            rgbs, alpha, rgbs_direct, alpha_direct, render_lbs_weights, ray_pts, ray_id, step_id, N  = self.aggregate_pts(
                t_hat_pcd, rotated_frames, query_radius, cam_per_ray,
                render_pcd_direct, render_weights, render_kwargs, pose_embedding, calc_min_max=calc_min_max)
            ray_id_direct = torch.clone(ray_id)

        except NoPointsException:
            return {
                'rgb_marched': torch.ones(len(render_kwargs['rays_o']), 3) * render_kwargs['bg'],
                'rgb_marched_direct': torch.ones(len(render_kwargs['rays_o']), 3) * render_kwargs['bg'],
                'depth': torch.zeros(len(render_kwargs['rays_o'])),
                'weights': torch.ones(len(render_kwargs['rays_o']), 3) * render_kwargs['bg'],
                't_hat_pcd': t_hat_pcd,
                'alphainv_last': None,
                'grid': grid,
                'joints': joints,
                'bones': bones,
            }
                    
        with torch.profiler.record_function("pre-mask"):
            if self.fast_color_thres > 0:
                mask = torch.where(alpha > self.fast_color_thres)[0]
                ray_id = ray_id[mask]
                step_id = step_id[mask]
                alpha = alpha[mask]
                rgbs = rgbs[mask]

                if render_weights:
                    render_lbs_weights = render_lbs_weights[mask]
                
                if alpha_direct is not None:
                    mask_direct = torch.where(alpha_direct > self.fast_color_thres)[0]
                    ray_id_direct = ray_id_direct[mask_direct]
                    alpha_direct = alpha_direct[mask_direct]
                    rgbs_direct = rgbs_direct[mask_direct]

            # compute accumulated transmittance
        with torch.profiler.record_function("Alphas2Weights"):
            weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
            if alpha_direct is not None:
                weights_direct, alphainv_last_direct = Alphas2Weights.apply(alpha_direct, ray_id_direct, N)

        with torch.profiler.record_function("post-mask"):
            if self.fast_color_thres > 0:
                mask = torch.where(weights > self.fast_color_thres)[0]
                weights = weights[mask]
                alpha = alpha[mask]
                ray_id = ray_id[mask]
                step_id = step_id[mask]
                rgbs = rgbs[mask]

                if render_weights:
                    render_lbs_weights = render_lbs_weights[mask]
                
                if alpha_direct is not None:
                    mask_direct = torch.where(weights_direct > self.fast_color_thres)[0]
                    ray_id_direct = ray_id_direct[mask_direct]
                    alpha_direct = alpha_direct[mask_direct]
                    rgbs_direct = rgbs_direct[mask_direct]
                    weights_direct = weights_direct[mask_direct]

        with torch.profiler.record_function("segment_coo"):
            rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgbs),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')

            rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])

            if alpha_direct is not None:
                rgb_marched_direct = segment_coo(
                    src=(weights_direct.unsqueeze(-1) * rgbs_direct),
                    index=ray_id_direct,
                    out=torch.zeros([N, 3]),
                    reduce='sum')

                rgb_marched_direct += (alphainv_last_direct.unsqueeze(-1) * render_kwargs['bg'])

            if render_depth:
                depth = segment_coo(
                        src=(weights * step_id),
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
                ret_dict.update({'depth': depth})

        ret_dict.update({
            't_hat_pcd': t_hat_pcd,
            'rgb_marched': rgb_marched,
            'alphainv_last': alphainv_last,
            'alphainv_last_direct':alphainv_last_direct,
            'grid': grid,
            'rgb_marched_direct': rgb_marched_direct,
            'joints': joints,
            'bones': bones,
        })

        if render_weights:
            weight_mask = self.get_weights().sum(dim=0) > 0
            cols = torch.tensor(color_palette("hls", weight_mask.sum()))
            gen  = torch.Generator(device=render_lbs_weights.device)
            gen.manual_seed(0)
            cols = cols[torch.randperm(cols.shape[0], generator=gen)]

            col_per_weight = 0
            for ci, wi in enumerate(torch.where(weight_mask)[0]):
                col_per_weight += cols[ci, None] * render_lbs_weights[:, wi, None]

            col_per_weight = col_per_weight.type(type(rgbs))

            # soft march
            w_marched = segment_coo(
                src=(weights.unsqueeze(-1) * col_per_weight),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')
            w_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
            ret_dict.update({'weights': w_marched})

        return ret_dict
    
    def get_neighbour_weight_tv_loss(self):
        diff = self._last_weights[:,None,:] - self._last_weights[self.nn_i,:]
        return torch.abs(diff).mean()
    
    def get_weight_sparsity_loss(self):
        return -(
            self._last_weights * torch.log(self._last_weights + self.eps) + 
            (1 - self._last_weights) * torch.log(1 - self._last_weights + self.eps)).mean()

    def get_arap_loss(self, warped_pcd, c=0.03):
        warped_nn_distance = torch.sqrt((warped_pcd[:,None,:] - warped_pcd[self.nn_i,:]).pow(2).sum(-1) + self.eps)
        return (self.nn_distance - warped_nn_distance).abs().sum()

    def get_joint_arap_loss(self):
        joint_distance = (self.joints[self.bone_arap_mask][0::2,:] - self.joints[self.bone_arap_mask][1::2,:])
        return ((self.og_joint_distance - joint_distance)**2).sum()

    def get_joint_chamfer_loss(self):
        _, c2 = self.get_chamfer_loss(self.skeleton_pcd, self.joints, c=None, get_raw=True)
        return c2.sum()

    def _rho(self, x, c):
        return (2 * (x / c)**2) / ((x / c)**2 + 4)

    def get_chamfer_loss(self, pcd1, pcd2, N=None, M=None, c=0.03, get_raw=False):
        if N is not None:
            N_i = torch.randint(0, pcd1.shape[0], (N,)).long().to(pcd1.device)
            pcd1 = pcd1[N_i]

        if M is not None:
            M_i = torch.randint(0, pcd2.shape[0], (M,)).long().to(pcd2.device)
            pcd2 = pcd2[M_i]

        xyz1 = LazyTensor(pcd1[:, None, :])
        xyz2 = LazyTensor(pcd2[None, :, :])
        D_ij = ((xyz1 - xyz2) ** 2).sum(-1)
        nn_i1 = D_ij.argKmin(dim=1, K=1)
        nn_i2 = D_ij.argKmin(dim=0, K=1)
        nn_distance1 = ((pcd1[:, None, :] - pcd2[nn_i1, :])**2).sum(-1)
        nn_distance2 = ((pcd2[:, None, :] - pcd1[nn_i2, :])**2).sum(-1)

        if get_raw:
            return nn_distance1, nn_distance2

        if c is None:
            loss = nn_distance1.mean() + nn_distance2.mean()
        else:
            loss = self._rho(nn_distance1, c).mean() + self._rho(nn_distance2, c).mean()

        return loss
    
    def get_batch_chamfer_loss(self, pcd1, pcd2, N=None, M=None):
        """batch-wise chamfer loss.
        Args:
            pcd1 (torch.Tensor): (B, N, 3) tensor of N 3D points.
            pcd2 (torch.Tensor): (B, M, 3) tensor of M 3D points.
            N (int): Number of points to sample from pcd1.
            M (int): Number of points to sample from pcd2.
        """
        assert len(pcd1) == len(pcd2)

        if N is not None:
            N_i = torch.randint(0, pcd1.shape[1], (N,), device=pcd1.device).long()
            pcd1 = pcd1[:, N_i]

        if M is not None:
            M_i = torch.randint(0, pcd2.shape[1], (M,), device=pcd1.device).long()
            pcd2 = pcd2[:, M_i]

        xyz1 = LazyTensor(pcd1[:, :, None, :])
        xyz2 = LazyTensor(pcd2[:, None, :, :])
        D_ij = ((xyz1 - xyz2) ** 2).sum(-1)
        nn_i1 = D_ij.argKmin(dim=2, K=1)
        nn_i2 = D_ij.argKmin(dim=1, K=1)

        idx = nn_i1.unsqueeze(-1).expand(-1, -1, -1, pcd2.shape[-1])
        nn_distance1 = (pcd1[:, :, None, :] - torch.gather(pcd2[:,:,None,:], 1, idx)).pow(2)

        idx = nn_i2.unsqueeze(-1).expand(-1, -1, -1, pcd1.shape[-1])
        nn_distance2 = (pcd2[:, :, None, :] - torch.gather(pcd1[:,:,None,:], 1, idx)).pow(2)

        return nn_distance1.sum(-1).mean() + nn_distance2.sum(-1).mean()

    def get_transformation_regularisation_loss(self, d=0.0873):
        t = self.forward_warp.prev_global_t.abs()
        thetas = self.forward_warp.prev_thetas.abs()
        return (torch.abs(t).sum() + thetas.sum()) / len(thetas + 1)


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, masks_tr_ori, times, train_poses, HW, Ks, ndc, model, render_kwargs, img_to_cam, i_train, inverse_y=False, flip_x=False, flip_y=False, **kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    # assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    assert len(Ks) == len(train_poses) and len(rgb_tr_ori) == len(HW)

    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori) # N = len(i_train) * rgb_tr_ori.shape[1] * rgb_tr_ori.shape[2]
    rgb_tr = torch.zeros([N,3], device=DEVICE, dtype=rgb_tr_ori.dtype)
    masks_tr = torch.zeros([N, 1], device=DEVICE, dtype=masks_tr_ori.dtype)
    pix_to_ray = torch.zeros([N], device=DEVICE, dtype=torch.int32)
    rays_o_tr = torch.zeros(Ks.shape[0] * rgb_tr_ori.shape[1] * rgb_tr_ori.shape[2], 3).float()
    rays_d_tr = torch.zeros(Ks.shape[0] * rgb_tr_ori.shape[1] * rgb_tr_ori.shape[2], 3).float()
    viewdirs_tr = torch.zeros(Ks.shape[0] * rgb_tr_ori.shape[1] * rgb_tr_ori.shape[2], 3).float()
    index_to_times = {}
    index_to_cam = torch.zeros(Ks.shape[0] * rgb_tr_ori.shape[1] * rgb_tr_ori.shape[2], 1, dtype=torch.int32)
    times = times.unsqueeze(-1)
    masks = torch.zeros(Ks.shape[0], rgb_tr_ori.shape[1], rgb_tr_ori.shape[2], dtype=bool, device=DEVICE)

    H, W = HW[0]
    c = 0
    top = 0
    
    for cam_i, (c2w, K) in enumerate(zip(train_poses, Ks)):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w.float(), ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)

        mask = torch.empty(H, W, device=DEVICE, dtype=torch.bool)
        for i in range(0, H, CHUNK):
            mask[i:i+CHUNK] = model.get_mask(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
        masks[c] = mask
        n = mask.sum()

        index_to_cam[top:top+n].copy_(cam_i)
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        
        top += n
        c += 1

    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    index_to_cam = index_to_cam[:top]

    top = torch.tensor(0)
    for c2w, img, mask_tr_ori, (H, W), K, time_one, PI_iter in zip(train_poses[img_to_cam], rgb_tr_ori, masks_tr_ori, HW, Ks[img_to_cam], times, img_to_cam):
        assert img.shape[:2] == (H, W)
        mask = masks[PI_iter]
        n = mask.sum()

        time_one = time_one.item()
        if time_one in index_to_times.keys():
            index_to_times[time_one] = (index_to_times[time_one][0], (top+n).item())
        else:
            index_to_times[time_one] = (top.item(), (top+n).item())

        rgb_tr[top:top+n].copy_(img[mask])
        masks_tr[top:top+n].copy_(mask_tr_ori[mask])
        pix_to_ray[top:top+n].copy_((PI_iter * H * W + torch.arange(0, H * W).reshape(H, W))[mask])
        
        c += 1
        top += n

    rgb_tr = rgb_tr[:top]
    pix_to_ray = pix_to_ray[:top]
    masks_tr = masks_tr[:top]
    return rgb_tr, index_to_times, rays_o_tr, rays_d_tr, viewdirs_tr, pix_to_ray, masks_tr, index_to_cam