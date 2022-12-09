#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys

DISKANN_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PROG_FVECS_TO_FBIN = f"{DISKANN_DIR}/build/tests/utils/fvecs_to_bin"
PROG_BUILD_DISK_INDEX = f"{DISKANN_DIR}/build/tests/build_disk_index"
PROG_DISK_INDEX_TO_TENSORS = f"{DISKANN_DIR}/build/tests/utils/disk_index_to_tensors"
PROG_SEARCH_DISK_INDEX = f"{DISKANN_DIR}/build/tests/search_disk_index"

def check_file_exists(path):
    if not os.path.isfile(path):
        print(f"file {path} does not exist", file=sys.stderr)
        exit(1)

def check_dir_exists(path):
    if not os.path.isdir(path):
        print(f"directory {path} does not exist", file=sys.stderr)
        exit(1)

def run_program(program, options):
    check_file_exists(program)
    cmd = [program] + options
    print("\n=== Running ===", ' '.join(cmd))
    subprocess.run(cmd, check=True)

def handle_to_fbin(sift_base, dataset, max_npts):
    learn_fvecs_path = f"{sift_base}_learn.fvecs"
    learn_fbin_path = f"{dataset}_learn.fbin"
    query_fvecs_path = f"{sift_base}_query.fvecs"
    query_fbin_path = f"{dataset}_query.fbin"
    check_file_exists(learn_fvecs_path)
    check_file_exists(query_fvecs_path)

    options = [learn_fvecs_path, learn_fbin_path]
    if max_npts > 0:
        options.append(str(max_npts))
    run_program(PROG_FVECS_TO_FBIN, options)

    options = [query_fvecs_path, query_fbin_path]
    run_program(PROG_FVECS_TO_FBIN, options)

def handle_build(dataset):
    learn_fbin_path = f"{dataset}_learn.fbin"
    index_path_prefix = f"{dataset}_R32_L50_A1.2"
    check_file_exists(learn_fbin_path)

    options = ['--data_type', 'float', '--dist_fn', 'l2', '--data_path', learn_fbin_path,
               '--index_path_prefix', index_path_prefix, '-R', '32', '-L', '50', '-B', '0.003', '-M', '1']
    run_program(PROG_BUILD_DISK_INDEX, options)

def handle_convert(dataset):
    disk_index_path = f"{dataset}_R32_L50_A1.2_disk.index"
    tensors_prefix = f"{dataset}_R32_L50_A1.2_tensor"
    check_file_exists(disk_index_path)

    options = ['float', disk_index_path, tensors_prefix]
    run_program(PROG_DISK_INDEX_TO_TENSORS, options)

def handle_query(dataset, k_depth, npts_to_cache, use_ts, ts_async, list_sizes):
    index_path_prefix = f"{dataset}_R32_L50_A1.2"
    disk_index_path = f"{dataset}_R32_L50_A1.2_disk.index"
    tensors_prefix = f"{dataset}_R32_L50_A1.2_tensor"
    query_fbin_path = f"{dataset}_query.fbin"
    gt_file_path = f"{dataset}_query_gt100"
    res_path_prefix = f"{dataset}_query_res"
    check_file_exists(query_fbin_path)
    check_file_exists(disk_index_path)
    check_dir_exists(f"{tensors_prefix}_embedding.zarr")
    check_dir_exists(f"{tensors_prefix}_num_nbrs.zarr")
    check_dir_exists(f"{tensors_prefix}_nbrhood.zarr")

    options = ['--data_type', 'float', '--dist_fn', 'l2', '--index_path_prefix', index_path_prefix,
               '--query_file', query_fbin_path, '--gt_file',  gt_file_path, '-K', str(k_depth),
               '--result_path', res_path_prefix, '--num_nodes_to_cache', str(npts_to_cache), '-L']
    for l in list_sizes:
        options.append(str(l))
    if use_ts:
        options += ['--index_tensors_prefix', tensors_prefix]
        if ts_async:
            options.append('--use_tensors_async')
    run_program(PROG_SEARCH_DISK_INDEX, options)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver of DiskANN tests")
    subparsers = parser.add_subparsers(help="Supported commands", dest='subparser', required=True)

    parser_to_fbin = subparsers.add_parser('to_fbin', help="generate input binary from sift fvecs")
    parser_to_fbin.add_argument('--sift_base', help="input prefix, should be the prefix <this>_learn.fvecs", required=True)
    parser_to_fbin.add_argument('--dataset', help="dataset name, will be the prefix <this>_learn.fbin", required=True)
    parser_to_fbin.add_argument('--max_npts', help="#points to extract (if want smaller dataset)", type=int, default=0)
    
    parser_build = subparsers.add_parser('build', help="build on-disk index from fbin file")
    parser_build.add_argument('--dataset', help="dataset name, should be the prefix <this>_learn.fbin", required=True)

    parser_convert = subparsers.add_parser('convert', help="convert disk index to zarr format tensors")
    parser_convert.add_argument('--dataset', help="dataset name, should be the prefix <this>_learn.fbin", required=True)

    parser_query = subparsers.add_parser('query', help="run query (search) on index in various modes")
    parser_query.add_argument('--dataset', help="dataset name, should be the prefix <this>_learn.fbin", required=True)
    parser_query.add_argument('--k_depth', help="how many nearest neighbors to query", type=int, default=10)
    parser_query.add_argument('--npts_to_cache', help="#points to cache ahead of time", type=int, default=10000)
    parser_query.add_argument('--use_ts', action='store_true', help="use tensorstore backend")
    parser_query.add_argument('--ts_async', action='store_true', help="use tensorstore in async mode")
    parser_query.add_argument('--list_sizes', help="list of search list sizes", metavar='L', type=int, nargs='+',
                              default=('10', '50', '100'))

    args = parser.parse_args()
    if args.subparser == "to_fbin":
        handle_to_fbin(args.sift_base, args.dataset, args.max_npts)
    elif args.subparser == "build":
        handle_build(args.dataset)
    elif args.subparser == "convert":
        handle_convert(args.dataset)
    elif args.subparser == "query":
        handle_query(args.dataset, args.k_depth, args.npts_to_cache, args.use_ts, args.ts_async,
                     args.list_sizes)
