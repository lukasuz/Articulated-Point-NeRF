import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def cluster_children(children, rotation_similarity_matrix):
    # Find all combinations between siblings that have similar motion,
    # Combinations are sorted such that the merging is order-independent
    combs = list(combinations(children, 2))
    similar_combs = []
    for comb in combs:
        are_similar = rotation_similarity_matrix[comb[0], comb[1]]
        if are_similar:
            similar_combs.append(comb)
    combs = similar_combs

    # Create clusters of siblings that similar motion
    # NOTE: It is transitive, i.e. if A,B and B,C are similar, we deem A,C also similar
    clusters = []
    for i, comb in enumerate(combs):
        c1, c2 = comb
        cluster_found = False
        for cluster in clusters:
            if c1 in cluster or c2 in cluster:
                cluster.add(c1)
                cluster.add(c2)
                cluster_found = True
        if not cluster_found:
            clusters.append(set(comb))

    silbing_merging_dict = {}
    # For each cluster with similar motion, merge the weights, or if zero motion, merge with parent
    for cluster in clusters:
        cluster_indices = np.array(list(cluster))
        # If we do not have zero motion, choose one weight randomly
        to_indx = cluster_indices[0]
        from_indices = cluster_indices[1:]
        silbing_merging_dict[to_indx] = from_indices
    
    return silbing_merging_dict

def merge_joints(joints, bones, prune_bones, rotation_similarity_matrix, root_idx=0, convert_merging_rules=True):
    assert len(joints) == len(prune_bones)

    # Determine parent and children joint for each joint
    parent_joint = {b[1]: b[0] for b in bones}
    child_joints = {k: [] for k in range(len(joints))}
    for k in parent_joint.keys():
        parent_k = parent_joint[k]
        child_joints[parent_k].append(k)
    
    joint_has_multiple_children = np.array([len(child_joints[joint]) > 1 for joint in range(len(joints))])

    # Find all kinematic paths from leaves to root and skip pruned bones
    paths = []
    paths_og = []
    are_leaves = np.array([len(child_joints[joint]) == 0 for joint in range(len(joints))])
    for joint_indx, is_leaf in enumerate(are_leaves):
        if not is_leaf:
            continue
        
        # Create path from leaf to root, only add joints that are not pruned
        path = []
        path_og = []
        while joint_indx != root_idx:

            if not prune_bones[joint_indx] or joint_has_multiple_children[parent_joint[joint_indx]]:
                if len(path) == 0 and not joint_has_multiple_children[parent_joint[joint_indx]]:
                    path.append(joint_indx)
                path.append(parent_joint[joint_indx])
            else:
                pass
            path_og.append(joint_indx)
            joint_indx = parent_joint[joint_indx]

        # Make sure that root is added at the end of each path
        if len(path) == 0:
            path.append(root_idx)
        elif path[-1] != root_idx:
            path.append(root_idx)

        path.reverse()
        paths.append(path)
        path_og.append(root_idx)
        path_og.reverse()
        paths_og.append(path_og)

    # Determine new bones based on paths
    new_bones = set()
    for path in paths:
        for i in range(len(path) - 1):
            new_bones.add((path[i], path[i+1]))
    
    # Choose new joints based on bones that are left
    new_bones = np.array([[b[0], b[1]] for b in new_bones])
    new_joints_indices = np.unique(new_bones)
    new_joints_indices.sort()
    new_joints = joints[new_joints_indices]

    # Determine transformations to keep based on tail of new bones (ingoing indices)
    rotations_to_keep_indices = []
    for bone in new_bones:
        start, tail = bone
        children = child_joints[start]

        if len(children) > 1:
            valid = False
            for child in children:
                for path in paths_og:
                    valid = (child in path) and (tail in path)
                    if valid: break
                if valid: break
        elif children == 0:
            raise Exception("Bone has no children, this should not happen.")
        else: # children == 1   
            child = children[0]
        rotations_to_keep_indices.append(child)

    # Create mask for rotations that are kept
    rotations_to_keep = np.zeros(len(joints)).astype(bool)
    rotations_to_keep[rotations_to_keep_indices] = True
    rotations_to_keep[root_idx] = True # Always keep root

    # Account for adapted order between bones
    sort_mask = np.argsort(new_bones[:,1], axis=0)
    rotations_to_keep_indices = np.array(rotations_to_keep_indices)[sort_mask]
    rotation_switch_mask = np.copy(rotations_to_keep_indices)
    c = 0
    for old_idx in np.unique(rotations_to_keep_indices):
        mask = np.where(rotations_to_keep_indices == old_idx)
        rotation_switch_mask[mask] = c
        c = c + 1

    # Account for root 
    rotation_switch_mask += 1
    rotation_switch_mask = np.concatenate([np.array([0]), rotation_switch_mask])

    # Mask for joints that are kept (not all joints that have zero motion are automatically pruned)
    joints_to_keep = np.zeros(len(joints)).astype(bool)
    joints_to_keep[new_joints_indices] = True

    # Fix bone indices after choosing new joints
    new_bones_temp = np.copy(new_bones)
    for new_idx, old_idx in enumerate(new_joints_indices):
        mask = np.where(new_bones == old_idx)
        new_bones_temp[mask] = new_idx
    new_bones = new_bones_temp
    
    # Sort bones based on new joint indices (ingoing indices)
    sort_mask = np.argsort(new_bones[:,1], axis=0)
    new_bones = new_bones[sort_mask]

    # We further have to switch positons!

    # Determine weight merging rules, i.e. if a joints was pruned we 
    # apply its weight to its parent (transitively if multiple pruned).
    # We use index numbering before pruning
    merging_rules = np.arange(len(are_leaves), dtype=np.int16)
    for joint_indx, is_leaf in enumerate(are_leaves):
        if not is_leaf:
            continue
        
        pending = []
        # prev_indx = joint_indx
        while True:            
            # print(joint_indx)
            # For all pending joints, find its parent that is not pruned and save in merging rules
            if prune_bones[joint_indx]:
                pending.append(joint_indx)
            else:
                for pending_elm in pending:
                    merging_rules[pending_elm] = joint_indx
                pending = []
            # prev_indx = joint_indx
            joint_indx = parent_joint[joint_indx]

            if joint_indx == root_idx:
                # Add remaining joint merging rules, note that we never point towards the root
                for pending_elm in pending:
                    merging_rules[pending_elm] = joint_indx
                break

    # Merge siblings based on same motion. Two things are important here: 1. We only consider children
    # for merging if they have not been merged with a parent. 2. Not only the merging rules are adapted
    # but we also save which sibling is merged to which so that we can copy to correct rotation during
    # forward kinematics. If we do not do so, kinematic path is incorrect. Note that this has not to be
    # considered for the parent-children merging use-case, only when merging siblings.
    sibling_transfer_rules = np.arange(len(are_leaves), dtype=np.int16)
    for children in child_joints.values():
        non_merged_children = []

        for child in children:
            if merging_rules[child] == child: # if check whether children have not be merged with parent
                non_merged_children.append(child)

        if len(non_merged_children) > 1:
            res = cluster_children(non_merged_children, rotation_similarity_matrix)
            for k, v in res.items():
                merging_rules[v] = k
                sibling_transfer_rules[v] = k

    # Now convert merging rules from old tree to new tree
    if convert_merging_rules:
        # We basically check which previous bone tails merged into which new bone tails
        # which we need to know for the merging rules, becuase those are determined on the un-pruned tree
        merging_rule_translation = {i: None for i in range(len(joints))}
        for path, path_og in zip(paths, paths_og):
            pending = []
            for joint_idx in path_og:
                if joint_idx not in path:
                    pending.append(joint_idx)
                else:
                    for pending_elm in pending:
                        merging_rule_translation[pending_elm] = joint_idx
                    merging_rule_translation[joint_idx] = joint_idx
                    pending = []

        merging_rules_temp = np.copy(merging_rules)
        for old_idx, new_idx in merging_rule_translation.items():
            if new_idx is not None:
                mask = np.where(merging_rules == old_idx)
                merging_rules_temp[mask] = new_idx
            else:
                # These will be leaf nodes which have been removed completely
                pass
        merging_rules = merging_rules_temp

    # rotation_switch_mask = np.arange(len(rotation_switch_mask))
    return new_joints, new_bones, merging_rules, joints_to_keep, rotations_to_keep, rotation_switch_mask, sibling_transfer_rules

def visualise_merging(joints, bones, new_joints, new_bones, prune, merging_rules, save=False):
    plt.figure(1)
    ax = plt.axes(projection='3d')
    # Plot and number bones
    for i in range(len(bones)):
        col = 'r' if prune[i+1] else 'b'
        # Plot bone
        ax.plot3D(
            [joints[bones[i][0]][0], joints[bones[i][1]][0]], 
            [joints[bones[i][0]][1], joints[bones[i][1]][1]],
            [joints[bones[i][0]][2], joints[bones[i][1]][2]], 'o-', color=col, linewidth=3)
        
        # Number bone
        ax.text(
            (joints[bones[i][0]][0] + joints[bones[i][1]][0]) / 2, 
            (joints[bones[i][0]][1] + joints[bones[i][1]][1]) / 2,
            (joints[bones[i][0]][2] + joints[bones[i][1]][2]) / 2, f'b: {str(i+1)}')
    
    
    for i in range(len(joints)):
        # Number joints
        ax.text(
            joints[i, 0], 
            joints[i, 1],
            joints[i, 2], f'j: {str(i)}')
        
        # Plot merging rule
        if merging_rules[i] != i:
            ax.plot3D(
                [joints[i][0], joints[merging_rules[i]][0]], 
                [joints[i][1], joints[merging_rules[i]][1]],
                [joints[i][2], joints[merging_rules[i]][2]], '-', color='k', linewidth=1)
            
    ax.scatter3D(joints[0,0], joints[0,1], joints[0,2], color='0.5')
    if save:
        plt.savefig('pre_prune.png')

    joints = new_joints
    bones = new_bones
    plt.figure(2)
    ax = plt.axes(projection='3d')
    for i in range(len(bones)):
        # Plot bone
        ax.plot3D(
            [joints[bones[i][0]][0], joints[bones[i][1]][0]], 
            [joints[bones[i][0]][1], joints[bones[i][1]][1]],
            [joints[bones[i][0]][2], joints[bones[i][1]][2]], 'o-', color='b', linewidth=3)
    
        # Number bone
        ax.text(
            (joints[bones[i][0]][0] + joints[bones[i][1]][0]) / 2, 
            (joints[bones[i][0]][1] + joints[bones[i][1]][1]) / 2,
            (joints[bones[i][0]][2] + joints[bones[i][1]][2]) / 2, f'b: {str(i+1)}')

    # Number joints
    for i in range(len(joints)):
        ax.text(
            joints[i, 0], 
            joints[i, 1],
            joints[i, 2], f'j: {str(i)}')

    ax.scatter3D(joints[0,0], joints[0,1], joints[0,2], color='0.5')
    
    if save:
        plt.savefig('pruned.png')
    else:
        plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    prune = np.array([False,  True,  True,  True,  True, False,  True,  True, False, False,
        False, False, False, False, False,  True,  True, False, False, False,
         True, False, False,  True,  True, False, False,  True, False])
    
    bones = np.array([[ 0,  1],
       [ 0,  2],
       [ 0,  3],
       [ 1,  4],
       [ 2,  5],
       [ 3,  6],
       [ 4,  7],
       [ 4,  8],
       [ 5,  9],
       [ 6, 10],
       [ 7, 11],
       [ 8, 12],
       [ 9, 13],
       [10, 14],
       [11, 15],
       [12, 16],
       [13, 17],
       [14, 18],
       [16, 19],
       [15, 20],
       [17, 21],
       [18, 22],
       [22, 23],
       [21, 24],
       [19, 25],
       [20, 26],
       [25, 27],
       [26, 28]])
    joints = np.array([[-1.0992033e-01,  1.3773570e-02, -1.0279535e-02],
       [ 7.6934747e-02,  6.5083411e-03, -7.3386007e-03],
       [-2.7021080e-01, -3.2428697e-02,  4.4113278e-04],
       [-2.9672319e-01,  4.6315782e-02, -6.7577744e-03],
       [ 2.5141305e-01,  3.1155343e-03,  4.3142084e-03],
       [-3.1287333e-01, -1.7965129e-01, -2.1890838e-02],
       [-2.9323018e-01,  1.6384502e-01, -2.6754361e-02],
       [ 3.1417277e-01, -1.1342228e-01,  4.9913065e-03],
       [ 3.1049362e-01,  1.2055474e-01,  6.3988729e-04],
       [-3.4124950e-01, -2.1221733e-01, -2.2969779e-01],
       [-3.3008698e-01,  1.5247130e-01, -2.1007670e-01],
       [ 3.0097824e-01, -1.8389797e-01, -1.1228296e-01],
       [ 3.0050650e-01,  1.8477358e-01, -1.5063888e-01],
       [-3.3436468e-01, -1.8365714e-01, -3.3947092e-01],
       [-3.0856955e-01,  1.3106114e-01, -3.5737619e-01],
       [ 2.8975996e-01, -1.7703798e-01, -2.6137495e-01],
       [ 2.8911844e-01,  1.8766299e-01, -2.9598197e-01],
       [-2.5058663e-01, -1.9066030e-01, -4.7641444e-01],
       [-2.4169801e-01,  1.6212095e-01, -5.3202862e-01],
       [ 3.4470978e-01,  1.5022933e-01, -4.2264348e-01],
       [ 3.3241081e-01, -1.7573217e-01, -3.7991861e-01],
       [-2.7808723e-01, -1.8404952e-01, -6.3611424e-01],
       [-1.5684691e-01,  2.0259875e-01, -5.9941179e-01],
       [-1.7258058e-01,  1.8936220e-01, -6.1647087e-01],
       [-2.7333865e-01, -1.1606482e-01, -6.5879959e-01],
       [ 3.2273161e-01,  1.5365937e-01, -5.9671843e-01],
       [ 3.3882979e-01, -1.8429358e-01, -5.7582635e-01],
       [ 3.2410374e-01,  1.5688671e-01, -6.4709193e-01],
       [ 2.9201394e-01, -2.0426621e-01, -6.6062433e-01]], dtype=np.float32)
    
    rotation_similarity_matrix = np.array([[ True,  True, False, False,  True, False, False, False, False,
        False, False, False, False, False, False,  True,  True, False,
        False, False, False, False, False, False,  True, False, False,
         True, False],
       [ True,  True,  True,  True,  True, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False,  True, False, False,
        False, False],
       [False,  True,  True,  True,  True, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False,  True, False, False,
        False, False],
       [False,  True,  True,  True,  True, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False,  True, False, False,
        False, False],
       [ True,  True,  True,  True,  True, False, False, False, False,
        False, False, False, False, False, False,  True,  True, False,
        False, False, False, False, False,  True,  True, False, False,
         True, False],
       [False, False, False, False, False,  True, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False],
       [False, False, False, False, False, False,  True, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False],
       [False, False, False, False, False, False, False,  True, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False],
       [False, False, False, False, False, False, False, False,  True,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False],
       [False, False, False, False, False, False, False, False, False,
         True, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False],
       [False, False, False, False, False, False, False, False, False,
        False,  True, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False],
       [False, False, False, False, False, False, False, False, False,
        False, False,  True, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False],
       [False, False, False, False, False, False, False, False, False,
        False, False, False,  True, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False],
       [False, False, False, False, False, False, False, False, False,
        False, False, False, False,  True, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False],
       [False, False, False, False, False, False, False, False, False,
        False, False, False, False, False,  True, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False],
       [ True, False, False, False,  True, False, False, False, False,
        False, False, False, False, False, False,  True, False, False,
        False, False, False, False, False, False,  True, False, False,
        False, False],
       [ True, False, False, False,  True, False, False, False, False,
        False, False, False, False, False, False, False,  True, False,
        False, False,  True, False, False, False, False, False, False,
         True, False],
       [False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,  True,
        False, False, False, False, False, False, False, False, False,
        False, False],
       [False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
         True, False, False, False, False, False, False, False, False,
        False, False],
       [False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False,  True, False, False, False, False, False, False, False,
        False, False],
       [False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False,  True, False,
        False, False,  True, False, False, False, False, False, False,
        False, False],
       [False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False,  True, False,  True, False, False, False,
        False, False],
       [False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False,  True, False, False, False, False,
        False, False],
       [False, False, False, False,  True, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False,  True, False,  True, False, False, False,
        False, False],
       [ True,  True,  True,  True,  True, False, False, False, False,
        False, False, False, False, False, False,  True, False, False,
        False, False, False, False, False, False,  True, False, False,
        False, False],
       [False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False,  True, False,
        False, False],
       [False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,  True,
        False, False],
       [ True, False, False, False,  True, False, False, False, False,
        False, False, False, False, False, False, False,  True, False,
        False, False, False, False, False, False, False, False, False,
         True, False],
       [False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False,  True]])

    new_joints, new_bones, merging_rules, joints_to_keep, rotations_to_keep, _, _ = merge_joints(joints, bones, prune, rotation_similarity_matrix, convert_merging_rules=False, save=False)
    visualise_merging(joints, bones, new_joints, new_bones, prune, merging_rules)