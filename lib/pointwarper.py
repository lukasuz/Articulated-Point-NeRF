import torch
import roma
import numpy as np

class TransformNet(torch.nn.Module):
    def __init__(self, input_dim, num_components, num_params_per_component, num_layers=3, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.num_params_per_component = num_params_per_component
        self.out_dim = num_components * num_params_per_component
        self.register_buffer('rotation_switch_mask', torch.arange(0, num_components).long())

        layers = []
        for i in range(num_layers-1):
            if i == 0:
                layers.append(torch.nn.Linear(input_dim, self.hidden_dim))
                layers.append(torch.nn.ReLU())
                # layers.append(torch.nn.SiLU())

            else:
                layers.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(torch.nn.ReLU())
                # layers.append(torch.nn.SiLU())

        layers.append(torch.nn.Linear(self.hidden_dim, self.out_dim, bias=False))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        b, _ = x.shape
        out = self.net(x)
        if b > 1:
            out = out.reshape(b, self.num_components, self.num_params_per_component)
        else:
            out = out.reshape(self.num_components, self.num_params_per_component)
        
        return out

class PointWarper(torch.nn.Module):
    def __init__(self,
                 t_dim,
                 canonical_pcd,
                 joints,
                 bones,
                 num_layers=5,
                 over_parameterized_rot=True,
                 ):
        super().__init__()
        self.t_dim = t_dim
        self.params_per_compoent = 4
        self.canonical_pcd = canonical_pcd
        self.num_layers = num_layers
        self.over_parameterized_rot = over_parameterized_rot
        self.init_tree(joints, bones, old=False)
            

        self.hom_row = torch.tensor([0,0,0,1], dtype=torch.float32)
        self.transform_net = TransformNet(t_dim, len(joints) + 1, self.params_per_compoent, num_layers=self.num_layers)
        self.register_buffer('rot_mask', torch.zeros(len(joints), dtype=torch.bool))
        self.register_buffer('sibling_mask', torch.arange(0, len(joints)).long())
       
    def kwargs(self):
        return {
            't_dim': self.t_dim,
            'params_per_compoent': self.params_per_compoent,
            'component_num': self.component_num,
            'over_parameterized_rot': self.over_parameterized_rot,
        }

    def init_tree(self, joints, bones, old=True):

        if old:
            self.bones = bones
            self.parent_joint = {b[1]: b[0] for b in bones}
            self.child_joints = {k: [] for k in range(len(joints))}
            for k in self.parent_joint.keys():
                parent_k = self.parent_joint[k]
                self.child_joints[parent_k].append(k)

            # Accelerated tree.
            parent_indices = []
            for i in range(len(self.bones)):
                j = i + 1
                inds = []
                while j >= 0:
                    inds += [j]
                    j = self.parent_joint.get(j, -1)
                parent_indices += [inds[::-1]]
            max_depth = np.max([len(x) for x in parent_indices])
            self.parent_indices = torch.zeros((len(self.bones), max_depth), dtype=torch.long) - 1
            for i,inds in enumerate(parent_indices):
                self.parent_indices[i,:len(inds)] = torch.from_numpy(np.array(inds)).to(self.parent_indices.device, dtype=self.parent_indices.dtype)
            self.parent_joint_ex = torch.from_numpy(np.array([self.parent_joint.get(i, -1) for i in range(len(self.bones)+1)])).to(self.parent_indices.device, dtype=self.parent_indices.dtype)
        else:
            self.bones = bones

            self.parent_joint = {b[1]: b[0] for b in self.bones}
            self.child_joints = {k: [] for k in range(len(joints))}
            for k in self.parent_joint.keys():
                parent_k = self.parent_joint[k]
                self.child_joints[parent_k].append(k)

            # Accelerated tree.
            parent_indices = [[0]]
            for i in range(len(self.bones)):
                j = i + 1
                inds = []
                while j >= 0:
                    inds += [j]
                    j = self.parent_joint.get(j, -1)
                parent_indices += [inds[::-1]]
            max_depth = np.max([len(x) for x in parent_indices])
            self.parent_indices = torch.zeros((len(parent_indices), max_depth), dtype=torch.long) - 1
            for i,inds in enumerate(parent_indices):
                self.parent_indices[i,:len(inds)] = torch.from_numpy(np.array(inds)).to(self.parent_indices.device, dtype=self.parent_indices.dtype)
            self.parent_joint_ex = torch.from_numpy(np.array([self.parent_joint.get(i, 0) for i in range(len(parent_indices))])).to(self.parent_indices.device, dtype=self.parent_indices.dtype)

    def Rodrigues(self, rvec, theta=None):
        # Neural Volumes
        if rvec.shape[-1] == 3:
            theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
            rvec = rvec / theta[:, None]
        elif rvec.shape[-1] == 4:
            theta = rvec[:, -1]
            rvec = rvec[:, :3]
            rvec = rvec / torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))[:, None]
        else:
            raise ValueError()

        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((
            rvec[:, 0] ** 2 + (1. - rvec[:, 0] ** 2) * costh,
            rvec[:, 0] * rvec[:, 1] * (1. - costh) - rvec[:, 2] * sinth,
            rvec[:, 0] * rvec[:, 2] * (1. - costh) + rvec[:, 1] * sinth,

            rvec[:, 0] * rvec[:, 1] * (1. - costh) + rvec[:, 2] * sinth,
            rvec[:, 1] ** 2 + (1. - rvec[:, 1] ** 2) * costh,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) - rvec[:, 0] * sinth,

            rvec[:, 0] * rvec[:, 2] * (1. - costh) - rvec[:, 1] * sinth,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) + rvec[:, 0] * sinth,
            rvec[:, 2] ** 2 + (1. - rvec[:, 2] ** 2) * costh), dim=1).view(-1, 3, 3), theta

    @classmethod
    def matrix_chain_product(cls, matrix_chain: torch.tensor) -> torch.tensor:
        """ Recursive binary tree. """
        chain_len = matrix_chain.shape[1]
        if chain_len == 1:
            return matrix_chain
        sub_a = cls.matrix_chain_product(matrix_chain[:,:chain_len//2])
        sub_b = cls.matrix_chain_product(matrix_chain[:,chain_len//2:])
        return sub_a @ sub_b


    def calc_rec_abs_T_fast(self, R_t: torch.tensor, joints: torch.tensor) -> torch.tensor:
        """
        Recalculates all transforms.
        """

        #### PREVIOUS IMPLEMENTATION ####

        # self.init_tree(joints, self.bones, old=True)

        # Start: Compatibility package.
        # This is unnecesarily complicated and could be removed.
        joints_old = torch.cat((self.hom_row[None, :3], joints), 0)
        # Align joints with their bones.
        joints_old = joints_old[self.parent_joint_ex + 1]
        # End: Compatibility package.

        M_bones_old = torch.cat((torch.cat((R_t, joints_old[...,None] + R_t @ -joints_old[...,None]), -1), self.hom_row[None,None].repeat(R_t.shape[0],1,1)), -2)
        M_bones_old = torch.cat((torch.eye(4)[None], M_bones_old), 0)
        M_paths_old = M_bones_old[self.parent_indices + 1]
        out_old = self.matrix_chain_product(M_paths_old)[:,0]

        #### NEW IMPLEMENTATION ####

        # self.init_tree(joints, self.bones, old=False)
        # # Translation is defined by the parent and hence shared by all siblings (unlike the rotation which is from the child.)
        # joints = joints[self.parent_joint_ex]

        # # Each node transform is a rotation (from the child) around the position (from the parent)
        # # hom(None, t_i) @ hom(R_i) @ hom(None, t_i, inv=True)
        # M_bones = torch.cat((torch.cat((R_t, joints[...,None] + R_t @ -joints[...,None]), -1), self.hom_row[None,None].repeat(R_t.shape[0],1,1)), -2)
        # M_bones[0,:3,-1] = 0.
        # # Handle -1 fake identity nodes by shifting index by 1.
        # M_bones = torch.cat((torch.eye(4)[None], M_bones), 0)
        # M_paths = M_bones[self.parent_indices + 1]
        # # Cummulative product.
        # out = self.matrix_chain_product(M_paths)[:,0]

        return out_old

    def get_thetas(self, ts_embed):
        params = self.transform_net(ts_embed)
        rot_params = params[:, :-1, :3]
        shape = rot_params.shape[:2]
        rot_params = rot_params.reshape(torch.mul(*shape), 3)
        _, thetas = self.Rodrigues(rot_params)
        
        return thetas.reshape(shape, 3)

    def set_rotation_mask(self, rotations_to_keep):
        mask = ~rotations_to_keep
        if self.rot_mask is not None:
            mask = torch.logical_or(mask, self.rot_mask)
        self.rot_mask = mask

    def set_sibling_mask(self, sibling_mask):
        self.sibling_mask = sibling_mask.long()

    def forward(self, weights, joints, t=None, rot_params=None, global_t=None, get_frames=False, avg_procrustes=False, get_skeleton=False):
        assert (t is None) ^ (rot_params is None)

        # Get time-dependent rotation and translation
        with torch.profiler.record_function("transform_net"):
            if rot_params is None:
                params = self.transform_net(t.unsqueeze(0))
                self.prev_params = params
                global_t = params[-1, :3]
                rot_params = params[:len(joints), :]
                R_t, self.prev_thetas = self.Rodrigues(rot_params)

                self.prev_global_t = global_t

            else:
                R_t, self.prev_thetas = self.Rodrigues(rot_params)

        with torch.profiler.record_function("calc_rec_abs_T"):

            R_t = R_t[self.sibling_mask]
            if self.rot_mask is not None:
                R_t[self.rot_mask] = torch.eye(3)

            self.prev_thetas = self.prev_thetas # [self.sibling_mask]

            # Do recusrive bone transformations
            bone_Ts = self.calc_rec_abs_T_fast(R_t, joints)

        with torch.profiler.record_function("weighted_G_tw"):
            # Apply weights to transformation matrix
            weighted_G_tw = (bone_Ts * weights[:, :, None, None]).sum(dim=1)

            if avg_procrustes:
                weighted_T = weighted_G_tw[:,:3,-1,None]
                weighted_R = roma.special_procrustes(weighted_G_tw[:,:3,:3])

                weighted_G_tw = torch.cat((weighted_R, weighted_T), -1)
                weighted_G_tw = torch.cat((weighted_G_tw, self.hom_row[None,None].repeat(weighted_G_tw.shape[0], 1, 1)), -2)
            
            # Transform points
            xyz = self.canonical_pcd
            xyzh = torch.concat([xyz, torch.ones((len(xyz), 1))], axis=-1)
            xyzh = torch.bmm(weighted_G_tw, xyzh.unsqueeze(-1)).squeeze(-1) 
            xyz = xyzh[:,:3]

            jointsh = torch.concat([joints, torch.ones((len(joints), 1))], axis=-1)
            jointsh = torch.bmm(bone_Ts, jointsh.unsqueeze(-1)).squeeze(-1) 
            joints_warped_rel = jointsh[:,:3]

            if global_t is None:
                global_t = torch.zeros(3, dtype=torch.float32, device=xyz.device)
                
            if global_t is not None:
                xyz = xyz + global_t

                if get_skeleton:
                    joints_warped = joints_warped_rel + global_t

        out = [xyz.contiguous(), joints_warped_rel]
        if get_frames:
            out.append(weighted_G_tw)

        if get_skeleton:
            out.append(joints_warped)
            out.append(self.bones)

        return out