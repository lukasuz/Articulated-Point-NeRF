"""
Code from "Template-free Articulated Neural Point Clouds for Reposable View Synthesis"
Github: https://github.com/lukasuz/Articulated-Point-NeRF
Project Page: https://lukas.uzolas.com/Articulated-Point-NeRF/
"""

import torch
import numpy as np
from skimage.morphology import skeletonize_3d as skeletonize
from skimage.morphology import remove_small_holes
from scipy.sparse.csgraph import shortest_path
from scipy.special import softmax
from skimage import filters
from cc3d import largest_k

from seaborn import color_palette
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

def adjacency_to_graph(distances):
    """ Turns a adjacency matrix into a graph representation.
    Arguments:
        distances: NxN np.array, adjacency matrix containing distances between nodes
    
    Returns:
        dict, graph representation of the adjacency matrix
    """
    graph = {}
    for i, node in enumerate(distances):
        adj = []
        adj_distances = []
        for j, connected in enumerate(node):
            if connected and i != j:
                adj.append(j)
                adj_distances.append(distances[i, j])

        adj = np.array(adj)
        adj_distances = np.array(adj_distances)
        sort_indicies = np.argsort(adj_distances)
        adj = tuple(adj[sort_indicies])
        adj_distances = tuple(adj_distances[sort_indicies])

        graph[i] = {
            'neighbours': adj,
            'n_distances': adj_distances
        }

    return graph

class DistQueue():
    """ Queue that sorts elements by distance.
    """
    def __init__(self) -> None:
        self._elements = np.array([], dtype=int)
        self._distances = np.array([], dtype=float)
        self._prev_joints = np.array([], dtype=int)
        self._distances_prev_joint = np.array([], dtype=float)
    
    def enqueue(self, element, distance, prev_joint, distance_prev_joint) -> None:
        # Find closest larger value
        if len(self._distances) == 0:
            indx = 0
        else:
            mask = self._distances > distance
            if not mask.any(): # no bigger elements, insert at the end
                indx = len(self._distances)
            else:  # Insert right before larger value
                indx = np.argmin(self._distances < distance)
       
        self._elements = np.insert(self._elements, indx, element)
        self._distances = np.insert(self._distances, indx, distance)
        self._prev_joints  = np.insert(self._prev_joints, indx, prev_joint)
        self._distances_prev_joint = np.insert(self._distances_prev_joint, indx, distance_prev_joint)

    def pop(self) -> tuple:
        element, self._elements = self._elements[0], self._elements[1:]
        distance, self._distances = self._distances[0], self._distances[1:]
        prev_joint, self._prev_joints = self._prev_joints[0], self._prev_joints[1:]
        distance_prev_joint, self._distances_prev_joint = self._distances_prev_joint[0], self._distances_prev_joint[1:]
        return element, distance, prev_joint, distance_prev_joint
    
    def not_empty(self) -> bool:
        return len(self._distances) > 0

def bfs(graph, start_node_indx, bone_length):
    """ Breadth-first search to find the joints and bones of the skeleton.
    Arguments:
        graph: dict, graph representation of the adjacency matrix
        start_node_indx: int, index of the starting node
        bone_length: num, how long each bone should approximately be (volume coordinates)
    
    Returns:
        (list, list) indices of the joints, indices of the bones respectively
    """
    visited = []
    joints = [start_node_indx]
    bones = []
    visited.append(start_node_indx)
    queue = DistQueue()
    queue.enqueue(start_node_indx, 0., start_node_indx, 0.)

    while queue.not_empty():        
        indx, cm_distance, prev_joint, distance_prev_joint = queue.pop()
        node = graph[indx]

        neighbours_to_visit = [n for n in node['neighbours'] if n not in visited]
        add_bone = (distance_prev_joint >= bone_length) or len(neighbours_to_visit) == 0

        if add_bone:
            bones.append([prev_joint, indx])
            joints.append(indx)
            prev_joint = indx
            distance_prev_joint = 0

        for i, neighbour in enumerate(neighbours_to_visit):

            if neighbour not in visited:
                visited.append(neighbour)
                nn_cm_distance = cm_distance + node['n_distances'][i]
                nn_distance_prev_joint = distance_prev_joint + node['n_distances'][i]
                queue.enqueue(neighbour, nn_cm_distance, prev_joint, nn_distance_prev_joint)
    
    return joints, bones

def dist_batch(p, a, b):
    """ Vectorized point-to-line distance

    Arguments:
        p: Nx3 torch.tensor, points
        a: Mx3 torch.tensor, start of lines
        b: Mx3 torch.tensor, end of lines

    Returns:
        MxN torch.tensor, distance from each point to each line
    """
    assert len(a) == len(b), "Same batch size needed for a and b"

    p = p[None, :, :]
    s = b - a
    w = p - a[:, None, :]
    ps = (w * s[:, None, :]).sum(-1)
    res = torch.zeros((a.shape[0], p.shape[1]), dtype=p.dtype)

    # ps <= 0
    ps_smaller_mask = ps <= 0
    lower_mask = torch.where(ps_smaller_mask)
    res[lower_mask] = torch.norm(w[lower_mask], dim=-1)

    # ps > 0 and ps >= l2
    l2 = (s * s).sum(-1)
    ps_mask = ~ps_smaller_mask

    temp_mask_l2 = ps >= l2[:, None]
    upper_mask = torch.where(ps_mask & temp_mask_l2)
    res[upper_mask] = torch.norm(p[0][upper_mask[1]] - b[upper_mask[0]], dim=-1)

    # ps > 0 and ps < l2
    within_mask = torch.where(ps_mask & ~temp_mask_l2)
    res[within_mask] = torch.norm(
        p[0][within_mask[1]] - (a[within_mask[0]] + (ps[within_mask] / l2[within_mask[0]]).unsqueeze(-1) * s[within_mask[0]]), dim=-1)

    return res

def weight_from_bones(joints, bones, pcd, theta=0.05):
    """ Calculates the skinning weights for each point.

    Arguments:
        joints: Nx3 np.array, the joint coordinates
        bones: list of len N-1, where each entry contains parent and child bone
        pcd: Mx3 np.array, the point cloud
        theta: num, theta for the softmax
    
    Returns:
        weights: MxN np.array, the weights for each point
    """
    bone_distances = np.zeros((len(bones), len(pcd)))

    # Needs torch tensors as input
    bone_distances = dist_batch(
        torch.tensor(pcd),
        torch.tensor(np.array([joints[bone[0]] for bone in bones])).float(),
        torch.tensor(np.array([joints[bone[1]] for bone in bones])).float(),
        ).cpu().numpy()

    weights = (1 / (0.5 * np.e ** bone_distances + 1e-6)).T
    weights = softmax(weights / theta, axis=1)

    return weights

def preprocess_volume(alpha_volume, threshold, sigma=1):
    """
    Arguments:
        alpha_volume: LxMxN np.array, alpha volume before thresholding
        threshold: num, threshold for the alpha volume
        sigma: num, sigma for the gaussian filtering of the alpha volume
    
    Returns:
        LxMxN np.array, binary volume after thresholding
    """
    if sigma > 0:
        alpha_volume = filters.gaussian(alpha_volume, sigma=sigma, preserve_range=True)
    binary_volume = alpha_volume > threshold
    binary_volume = remove_small_holes(binary_volume.astype(bool), area_threshold=2**8,)
    binary_volume = largest_k(binary_volume, connectivity=26, k=1).astype(int)
    
    return binary_volume.astype(bool)

def create_skeleton(alpha_volume, grid_xyz, bone_length=10., threshold=0.05, sigma=0, weight_theta=0.1, bone_heursitic=True):
    """
    Arguments:
        alpha_volume: LxMxN np.array, alpha volume before thresholding
        grid_xyz: LxMxN np.array, the coordinate grid for the binary volume
        bone_length: num, how long each bone should approximately be (not in world coordinates but in volume coordinates)
        threshold: num, threshold for the alpha volume
        sigma: num, sigma for the gaussian filtering of the alpha volume. NOTE: Only used for the skeleton.
        weight_theta: num, theta for the softmax scaling
        bone_heursitic: bool, whether to use the bone heuristic or not.
    
    Returns:
        dict containing the point cloud and all kinematic components
    """

    ## Preprocessing, assume that we have one blob
    binary_volume = preprocess_volume(alpha_volume, threshold=threshold, sigma=0)
    if sigma > 0:
        binary_volume_smooth = preprocess_volume(alpha_volume, threshold=threshold, sigma=sigma)
    else:
        binary_volume_smooth = binary_volume
    
    # Create integer volume grid, easier to work with
    xv, yv, zv = np.meshgrid(
        np.arange(0, grid_xyz.shape[0]),
        np.arange(0, grid_xyz.shape[1]),
        np.arange(0, grid_xyz.shape[2]),
        indexing='ij'
    )
    grid = np.concatenate([
        np.expand_dims(xv, axis=-1),
        np.expand_dims(yv, axis=-1),
        np.expand_dims(zv, axis=-1)
    ], axis=-1)

    skeleton = skeletonize(binary_volume_smooth) == 255
    points = grid[skeleton].reshape(-1, 3)

    ## Graphify
    # Neighbours are points within a NxNxN grid, N=3
    offset = np.abs(points[:,None,:] - points[None,:,:])
    NN = np.logical_and.reduce(offset <= 1, axis=-1)
    distances = np.sqrt(np.sum((points[:,None,:] - points[None,:,:])**2, axis=-1))

    distance_graph = NN * distances

    D = shortest_path(distance_graph, directed=True, method='FW')
    root_indx = D.sum(1).argmin()

    graph = adjacency_to_graph(distance_graph)

    joints, bones = bfs(graph, root_indx, bone_length)
    starts = np.array([bone[0] for bone in bones])
    tails = np.array([bone[1] for bone in bones])

    bone_has_child = []
    for i in range(len(bones)):
        bone_has_child.append(tails[i] in starts)

    # Clean bones heuristic
    if bone_heursitic:
        bone_has_child = np.array(bone_has_child)
        del_indices = []
        for u_start in np.unique(starts):
            indx = np.where(u_start == starts)[0]
            if bone_has_child[indx].any():
                for i in indx:
                    if not bone_has_child[i]:
                        del_indices.append(i)
            else:
                # Keep longest
                distances_temp = []
                for i in indx:
                    bone = bones[i]
                    distances_temp.append(np.sqrt(np.sum(points[int(bone[0])] - points[bone[1]])**2))
                
                longest_indx = np.argmax(distances_temp)
                for i, ii in enumerate(indx):
                    if i != longest_indx:
                        del_indices.append(ii)

        del_indices.sort()
        del_indices.reverse()
        for i in del_indices:
            del bones[i]
        
        new_joints = list(np.unique(bones).astype(int)) # Remove unnecessary joints
        joints = [joint for joint in joints if joint in new_joints] # Keep order of previous joints

    ## Turn absolute bone coordinates into indices
    rel_bones = []
    for bone in bones:
        b1, b2 = bone
        b1 = int(np.where(np.array(joints) == b1)[0])
        b2 = int(np.where(np.array(joints) == b2)[0])
        rel_bones.append([b1, b2])
    bones = rel_bones

    ## Transform from grid space into real-world coordinates
    xyz_max = grid_xyz.max(axis=0).max(axis=0).max(axis=0)
    xyz_min = grid_xyz.min(axis=0).min(axis=0).min(axis=0)
    vol_max = np.array(binary_volume.shape)
    points = (points / vol_max[None,:]) * (xyz_max - xyz_min) + xyz_min
    points = points.astype(np.float32)

    ## Calculate weights
    pcd = grid_xyz[binary_volume > 0]
    weights = weight_from_bones(points[joints], bones, pcd, theta=weight_theta)

    res = {
        'skeleton_pcd': points,
        'root': points[root_indx],
        'joints': points[joints],
        'bones': bones,
        'pcd': pcd,
        'weights': weights,
    }

    return res

def visualise_skeletonizer(skeleton_points, root, joints, bones, pcd, weights, old_joints=None, old_bones=None):
    cs = {
        'root': np.array([[1., 0., 0.]]),
        'joint': np.array([[0., 0., 1.]]),
        'bone': np.array([[0.1, 0.1, 0.8]]),
        'point': np.array([[0., 0., 0.]])
    }

    # Add joints and root
    # joint_points = root.reshape(1, 3)
    # cols = cs['root']
    # joint_points = np.concatenate([joint_points, joints], axis=0)
    joint_points = joints
    cols = np.concatenate([cs['joint'], cs['joint'].repeat(len(joints) - 1, axis=0)], axis=0)

    # Add bones
    col_bones = cs['bone'].repeat(len(bones), axis=0)
    
    # Add weights
    col_palette = np.array(color_palette("husl", weights.shape[1]))
    col_palette = np.random.rand(*col_palette.shape)
    
    # Add weights
    cols_weights = (np.expand_dims(weights, axis=-1) * col_palette).sum(axis=1)

    # Weight Visualisation
    weight_pcd = o3d.geometry.PointCloud()
    weight_pcd.points = o3d.utility.Vector3dVector(pcd)
    weight_pcd.colors = o3d.utility.Vector3dVector(cols_weights)

    # Skeleton Visualisation
    joint_pcd = o3d.geometry.PointCloud()
    joint_pcd.points = o3d.utility.Vector3dVector(joint_points)
    joint_pcd.colors = o3d.utility.Vector3dVector(cols)

    skeleton_pcd = o3d.geometry.PointCloud()
    skeleton_pcd.points = o3d.utility.Vector3dVector(skeleton_points)
    skeleton_pcd.colors = o3d.utility.Vector3dVector(cs['point'].repeat(len(skeleton_points), 0))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(joints)
    line_set.lines = o3d.utility.Vector2iVector(bones)
    line_set.colors = o3d.utility.Vector3dVector(col_bones)

    gui.Application.instance.initialize()

    window = gui.Application.instance.create_window("Skeleton-Viewer", 1024, 750)

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)

    window.add_child(scene)

    bp_material = rendering.MaterialRecord()
    bp_material.point_size = 10

    mp_material = rendering.MaterialRecord()
    mp_material.point_size = 5

    scene.scene.add_geometry("Weights", weight_pcd, bp_material)
    scene.scene.add_geometry("Skeleton", skeleton_pcd, mp_material)
    scene.scene.add_geometry("Joints", joint_pcd, mp_material)
    scene.scene.add_geometry("Bones", line_set, rendering.MaterialRecord())

    bounds = skeleton_pcd.get_axis_aligned_bounding_box()
    scene.setup_camera(60, bounds, bounds.get_center())

    labels = [
        (root, 'root (j0)')
    ]

    joint_to_idx = {}
    for i in range(1, len(joints)): # skip root, joint == 0
        if old_joints is not None:
            x = np.where(np.all(joints[i] == old_joints, axis=1))[0][0]
        else:
            x = i
        joint_to_idx[i] = x
        labels.append((joints[i], f'j{x}'))

    for i in range(len(bones)):
        bs, be = bones[i]
        pos = (joints[bs] + joints[be]) / 2

        if old_bones is not None:
            x =  joint_to_idx[be]
        else:
            x = be

        labels.append((pos, f'b{x}'))        

    for item in labels:
        scene.add_3d_label(item[0], item[1])

    gui.Application.instance.run()

if __name__ == "__main__":
    alpha_volume = np.load('./data/alpha_volume_f16.npy')
    with open("./data/grid.txt", 'r') as f:  
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = np.array(lines[i].replace('\n', '').split(','), dtype=float)
    
    min = lines[0]
    max = lines[1]
    shape = lines[2].astype(int)

    xv, yv, zv = np.meshgrid(
        np.linspace(min[0], max[0], shape[0]),
        np.linspace(min[1], max[1], shape[1]),
        np.linspace(min[2], max[2], shape[2]),
        indexing='ij')

    grid_xyz = np.concatenate([
        np.expand_dims(xv, axis=-1),
        np.expand_dims(yv, axis=-1),
        np.expand_dims(zv, axis=-1)
    ], axis=-1)

    res = create_skeleton(alpha_volume, grid_xyz, bone_length=10., sigma=1, weight_theta=0.03)
    visualise_skeletonizer(*res.values())
