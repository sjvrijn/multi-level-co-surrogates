#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import xarray as xr


def merge_files_in_dir(directory: Path, recursive: bool=False):
    """Merge all TMP files in the given directory"""
    if recursive:
        for subdir in filter_directories(directory.iterdir()):
            merge_files_in_dir(subdir, recursive)
    
    for file in filter_nc_files(directory.iterdir()):
        try:
            merge_tmp_files(file)
        except Exception as e:
            print(f'Error occurred when merging {file}:')
            print(e)


def save_reduced_files(directory: Path, new_directory: Path=None, recursive: bool=False):
    """Store reduced versions of all files from the given `directory` by removing
    the 'values' variable and 'idx' dimension from the '.nc' files, and saving
    them in the given `new_directory`
    """
    if new_directory:
        new_directory.mkdir(exist_ok=True, parents=True)

    if recursive:
        for subdir in filter_directories(directory.iterdir()):
            if new_directory:
                new_subdir = new_directory / subdir.name
            save_reduced_files(subdir, new_subdir, recursive)
    
    for file in filter_nc_files(directory.iterdir()):
        try:
            drop_values_from_file(file, new_directory)
        except Exception as e:
            print(f'Error occurred when reducing {file}:')
            print(e)


def merge_tmp_files(base_file: Path):
    """Merge all <base_file>TMP* files into <base_file> and remove TMP files"""
    with xr.open_mfdataset(f'{base_file}*') as ds:
        merged = ds.load()
    merged.to_netcdf(base_file)

    for file in base_file.parent.glob(f'{base_file.name}TMP*'):
        file.unlink()


def drop_values_from_file(file: Path, new_directory: Path=None):
    """Drop the 'values' variable and corresponding 'idx' dimension from the
    given file and store in the given folder `new_directory`.
    """
    with xr.open_dataset(file) as full_ds:
        smaller_ds = full_ds.drop_vars(['values', 'idx'], errors='ignore')
            
    if new_directory:
        file = new_directory / file.name
    with smaller_ds.load() as ds:
        ds.to_netcdf(file)

        
def filter_nc_files(path_list):
    """Given a list of paths, return only those that end in '.nc'"""
    return [p for p in path_list if p.suffix == '.nc']

def filter_directories(path_list):
    """Given a list of paths, return only those that are directories"""
    return [p for p in path_list if p.is_dir()]


if __name__ == '__main__':
    parser = ArgumentParser('')
    parser.add_argument("operation", choices=['merge', 'reduce'], help='Action to perform')
    parser.add_argument("directory", help='In which directory to search for .nc files')
    parser.add_argument("-r", "--recursive", help='Whether the given directory should be traversed recursively', action='store_true')
    parser.add_argument("-n", "--new-dir", help='Directory to store reduced files in. Will be created if it does not yet exist')
    args = parser.parse_args()

    directory = Path(args.directory)
    new_dir = Path(args.new_dir) if args.new_dir else None

    if args.operation == 'merge':
        merge_files_in_dir(directory, recursive=args.recursive)
    elif args.operation == 'reduce':
        save_reduced_files(directory, new_dir, recursive=args.recursive)
