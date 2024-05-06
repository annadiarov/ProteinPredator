import os
import re
import glob
import pickle
import argparse
import numpy as np


def arg_parse():
    descr = ('Generate pickle file from files in data_dir with the names:'
             '`{data_dir}/{patch_basename}{patch_id[0]}{file_format}`')
    help_ddir = 'Path to directory containing protein PLY files'
    help_ofile = 'Path to save the output pickle file'
    help_pbase = 'Basename of the patch files'
    help_fbase = 'Basename of the feature files'
    help_pids = 'Comma separated patch IDs'
    help_fformat = 'File format of the patches (ie. `.npy`)'

    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('-ddir', '--data-dir', type=str, required=True,
                        help=help_ddir)
    parser.add_argument('-ofile', '--output-file', type=str, required=True,
                        help=help_ofile)
    parser.add_argument('-pids', '--patch-ids', type=str, required=False,
                        help=help_pids, default='1,2')
    parser.add_argument('-fformat', '--file-format', type=str, required=False,
                        help=help_fformat, default='')
    parser.add_argument('-pbase', '--patch-basename', type=str, required=False,
                        help=help_pbase, default='all_sample_p')
    parser.add_argument('-fbase', '--feat-basename', type=str, required=False,
                        help=help_fbase, default='all_sample_f')

    return parser.parse_args()

def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main(data_dir, patch_basename, feat_basename, patch_ids, file_format,
         output_file):
    patch_ids_split = patch_ids.split(',')
    assert len(
        patch_ids_split) == 2, f"Got {len(patch_ids_split)} IDs for patches, expected 2."
    p1_coords_path = os.path.join(data_dir,
                                  f"{patch_basename}{patch_ids_split[0]}{file_format}")
    p2_coords_path = os.path.join(data_dir,
                                  f"{patch_basename}{patch_ids_split[1]}{file_format}")

    p1_feat_path = os.path.join(data_dir,
                                f"{feat_basename}{patch_ids_split[0]}{file_format}")
    p2_feat_path = os.path.join(data_dir,
                                f"{feat_basename}{patch_ids_split[1]}{file_format}")

    assert os.path.exists(p1_coords_path), f"File {p1_coords_path} does not exist."
    assert os.path.exists(p2_coords_path), f"File {p2_coords_path} does not exist."
    assert os.path.exists(p1_feat_path), f"File {p1_feat_path} does not exist."
    assert os.path.exists(p2_feat_path), f"File {p2_feat_path} does not exist."

    # Get train/test folder according to data_dir
    if data_dir.endswith('/'): data_dir = data_dir[:-1]
    train_test_folder = os.path.basename(data_dir)
    p1_coords_rel_path = os.path.join('.', train_test_folder, os.path.basename(p1_coords_path))
    p2_coords_rel_path = os.path.join('.', train_test_folder, os.path.basename(p2_coords_path))
    p1_feat_rel_path = os.path.join('.', train_test_folder, os.path.basename(p1_feat_path))
    p2_feat_rel_path = os.path.join('.', train_test_folder, os.path.basename(p2_feat_path))

    p1_coords = np.load(p1_coords_path, allow_pickle=True)

    src, src_feat = [], []
    tgt, tgt_feat = [], []
    # Only one list because ligand and receptor will have the same idx
    pdb_idx_list, patch_idx_list = [], []
    rot, trans = [], []

    for pdb_idx in range(len(p1_coords)):
        p1 = p1_coords[pdb_idx]
        if p1 == []:
            continue
        else:
            n_patches = len(p1)
            src.extend([p1_coords_rel_path for _ in range(n_patches)])
            src_feat.extend([p1_feat_rel_path for _ in range(n_patches)])
            tgt.extend([p2_coords_rel_path for _ in range(n_patches)])
            tgt_feat.extend([p2_feat_rel_path for _ in range(n_patches)])

            pdb_idx_list.extend([pdb_idx for _ in range(n_patches)])
            patch_idx_list.extend([n_patch for n_patch in range(n_patches)])
            rot.extend([np.random.rand(3) * 10 for _ in range(n_patches)])
            trans.extend([np.random.rand(3) for _ in range(n_patches)])

    # Currently we are not using these values, we set them to 0.2 by default
    overlap = [0.2 for _ in range(len(rot))]

    info_pkl = {
        'src': src,  # Path to patch coordinates
        'src_feat': src_feat,  # Path to patch features
        'tgt': tgt,  # Path to patch coordinates
        'tgt_feat': tgt_feat,  # Path to patch features
        'pdb_idx': pdb_idx_list,  # Index that maps patch to PDB
        'patch_idx': patch_idx_list,
        # Index to identify different patches in PDB
        'rot': rot,
        'trans': trans,
        'overlap': overlap,
    }

    save_pickle(info_pkl, output_file)
    print(f'File successfully saved at {output_file}')


if __name__ == '__main__':
    args = arg_parse()
    data_dir, patch_basename, feat_basename, patch_ids, file_format, output_file = \
        args.data_dir, args.patch_basename, args.feat_basename, \
        args.patch_ids, args.file_format, args.output_file
    main(data_dir, patch_basename, feat_basename, patch_ids, file_format,
         output_file)
