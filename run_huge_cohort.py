#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import os
from pathlib import Path

# Define the start path for the search
start_path = '/mnt/SATELLITE 04/FOXTROT-CRC-DX-BLOCK'

# Search for all .jpg files and extract their parent directories
dir_list = set()
for dirpath, dirnames, filenames in os.walk(start_path):
    for filename in filenames:
        if filename.endswith('.jpg'):
            dir_list.add(os.path.dirname(os.path.join(dirpath, filename)))

# Sort and de-duplicate the list of directories
dir_list = sorted(dir_list)

# Replace 'FOXTROT-CRC-DX-BLOCKS' with 'FOXTROT-CRC-DX-BLOCKS-NORM'
output_list = [d.replace('FOXTROT-CRC-DX-BLOCKS', 'FOXTROT-CRC-DX-BLOCKS-NORM') for d in dir_list]

# Create the output directories if they don't exist
for d in output_list:
    if not os.path.exists(d):
        os.makedirs(d)

# replace the elements in the list with Path objects and extract the parent directories
output_list = [Path(d).parent for d in output_list]
dir_list = [Path(d).parent for d in dir_list]

# Print the resulting list
print(output_list[:10])
print(dir_list[:10])
'''
import subprocess

# Define the paths to the Normalize.py script and the Python interpreter
script_path = '/path/to/Normalize.py'
python_path = '/path/to/python'

# Loop over the input and output directories, and call the Normalize.py script on each pair
for input_dir, output_dir in zip(dir_list, output_list):
    input_path = str(input_dir)
    output_path = str(output_dir)
    subprocess.run(['python', 'Normalize.py', '-ip', input_path, '-op', output_path])
'''
