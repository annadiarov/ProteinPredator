"""
Author: Shengyu Huang
Last modified: 30.11.2020
"""

import os,sys,glob,torch
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import open3d as o3d
from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences_4protein


class ProteinPatchDataset(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """
    def __init__(self,infos,config,data_augmentation=True):
        super(ProteinPatchDataset, self).__init__()
        self.infos = infos
        self.base_dir = config.root
        self.overlap_radius = config.overlap_radius
        self.data_augmentation = data_augmentation
        self.config = config
        self.rot_factor = 1.
        self.augment_noise = config.augment_noise
        self.max_points = 30000

        # To just load files once
        self.all_src_coords = np.load(os.path.join(self.base_dir, self.infos['src'][0]), allow_pickle=True)
        self.all_tgt_coords = np.load(os.path.join(self.base_dir, self.infos['tgt'][0]), allow_pickle=True)

    def __len__(self):
        return len(self.infos['rot'])

    def __getitem__(self,item):
        # get transformation
        rot=self.infos['rot'][item]
        trans=self.infos['trans'][item]

        # Very slow implementation that reads npy file each time
        # get pointclouds
        #src_path=os.path.join(self.base_dir,self.infos['src'][item])
        #tgt_path=os.path.join(self.base_dir,self.infos['tgt'][item])
        # load coords
        #src_pcd = np.load(src_path, allow_pickle=True)[self.infos['src_pdb_idx'][item]][self.infos['src_patch_idx'][item]]
        #tgt_pcd = np.load(tgt_path, allow_pickle=True)[self.infos['tgt_pdb_idx'][item]][self.infos['tgt_patch_idx'][item]]

        # Faster implementation
        src_pcd = self.all_src_coords[self.infos['pdb_idx'][item]][self.infos['patch_idx'][item]]
        tgt_pcd = self.all_tgt_coords[self.infos['pdb_idx'][item]][self.infos['patch_idx'][item]]

        total_pcd = np.concatenate([src_pcd,tgt_pcd],axis=0)
        total_pcd = total_pcd[:,:3] - np.mean(total_pcd[:,:3], axis=0, keepdims=True)
        total_pcd = self._scale_coords(total_pcd)
        src_pcd = total_pcd[:len(src_pcd)]
        tgt_pcd = total_pcd[len(src_pcd):]

        # if we get too many points, we do some downsampling
        if(src_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
            src_pcd = src_pcd[idx]
        if(tgt_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[idx]

        # add gaussian noise
        if self.data_augmentation:            
            # rotate the point cloud
            euler_ab=np.random.rand(3)*np.pi*2/self.rot_factor # anglez, angley, anglex
            rot_ab= Rotation.from_euler('zyx', euler_ab).as_matrix()
            if(np.random.rand(1)[0]>0.5):
                src_pcd=np.matmul(rot_ab,src_pcd.T).T
                rot=np.matmul(rot,rot_ab.T)
            else:
                tgt_pcd=np.matmul(rot_ab,tgt_pcd.T).T
                rot=np.matmul(rot_ab,rot)
                trans=np.matmul(rot_ab,trans)

            src_pcd += (np.random.rand(src_pcd.shape[0],3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0],3) - 0.5) * self.augment_noise
        
        # if(trans.ndim==1):
        #     trans=trans[:,None]

        # get correspondence at fine level
        tsfm = to_tsfm(rot, trans)
        correspondences = get_correspondences_4protein(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd),self.overlap_radius)
            
        src_feats = np.ones_like(src_pcd[:,:1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:,:1]).astype(np.float32)

        rotation = Rotation.from_euler('zyx', [rot[0], rot[1], rot[2]])
        src_pcd = rotation.apply(src_pcd) + np.expand_dims(trans, axis=0)

        rot = rotation.inv().as_matrix()
        rot = rot.astype(np.float32)
        trans = trans.reshape(3, 1)

        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)
        
        return src_pcd,tgt_pcd,src_feats,tgt_feats,rot,trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)

    @staticmethod
    def _scale_coords(coords):
        # Find the maximum absolute value in each dimension
        max_x = np.max(np.abs(coords[:, 0]))
        max_y = np.max(np.abs(coords[:, 1]))
        max_z = np.max(np.abs(coords[:, 2]))
        min_x = np.min(np.abs(coords[:, 0]))
        min_y = np.min(np.abs(coords[:, 1]))
        min_z = np.min(np.abs(coords[:, 2]))

        original_width = max_x - min_x
        original_height = max_y - min_y
        original_depth = max_z - min_z

        if original_width == 0: original_width = 1
        if original_height == 0: original_height = 1
        if original_depth == 0: original_depth = 1

        desired_width = 1.5  # set the desired width
        desired_height = (desired_width / original_width) * original_height
        desired_depth = (desired_width / original_width) * original_depth
        scale_x = desired_width / original_width
        scale_y = desired_height / original_height
        scale_z = desired_depth / original_depth

        # Scale each dimension accordingly
        scaled_x = coords[:, 0] * scale_x
        scaled_y = coords[:, 1] * scale_y
        scaled_z = coords[:, 2] * scale_z

        scaled_coordinates = np.column_stack((scaled_x, scaled_y, scaled_z))
        return scaled_coordinates