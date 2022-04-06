"""
Contains various utility functions.

Added by @dansah
"""

import os

def get_log_dir(rel_output_dir):
    return os.path.join('.', 'out', '%sdata' % rel_output_dir)