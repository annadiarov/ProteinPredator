import os
import re
import glob
import pickle
import argparse
import numpy as np


def arg_parse():
    descr = 'Generate pickle file from data dir for train/test/benchmark_info'
    help_ddir = 'Path to directory containing protein PLY files'
    help_ofile = 'Path to save the output pickle file'

    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('-ddir', '--data-dir', type=str, required=True,
                        help=help_ddir)
    parser.add_argument('-ofile', '--output-file', type=str, required=True,
                        help=help_ofile)
    return parser.parse_args()


def natural_key(string_):
    """ Sort strings in human order """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def is_valid_ply_filename(ply_filename):
    """ PLY filenames must follow this pattern: PDBID_R.ply or PDBID_L.ply """
    try:
        ply_filename = ply_filename.split('.')[0]
        pdb_id, lig_rec = ply_filename.split('_')
        if len(pdb_id) == 4 and len(lig_rec) == 1 and lig_rec in 'RL':
            return True
    except ValueError:
        return False


def main(args):
    # Sort PLY files so we have PDBID_L, PDBID_R consecutively
    ply_files = sorted(glob.glob(os.path.join(args.data_dir, '*.ply')), key=natural_key)
    # Get train/test folder according to data_dir
    train_test_folder = [f.split('/')[-2] for f in ply_files]

    ply_files = [os.path.basename(f) for f in ply_files]

    # Check PLY files are correct
    for f in ply_files:
        assert is_valid_ply_filename(f), \
            (f'Got invalid PLY filename {f}. Make sure it follows the pattern '
             f'PDBID_L.ply, PDBID_R.ply')

    n_files = len(ply_files)
    src = [os.path.join('.', train_test_folder[i], ply_files[i]) for i in range(0, n_files, 2)]
    tgt = [os.path.join('.', train_test_folder[i], ply_files[i]) for i in range(1, n_files, 2)]
    rot = [np.random.rand(3) * 10 for _ in range(1, n_files, 2)]
    trans = [np.random.rand(3) for _ in range(1, n_files, 2)]
    # Currently we are not using these values, we set them to 0.2 by default
    overlap = [0.2 for _ in range(1, n_files, 2)]

    info_pkl = {
        'src': src,
        'tgt': tgt,
        'rot': rot,
        'trans': trans,
        'overlap': overlap,
    }

    save_pickle(info_pkl, args.output_file)
    print(f'File successfully saved at {args.output_file}')


if __name__ == '__main__':
    args = arg_parse()
    main(args)
