"""
Contains various utility functions.

Added by @dansah
"""

import pickle
import os

def get_log_dir(rel_output_dir):
    return os.path.join('.', 'out', '%sdata' % rel_output_dir)

def save_buffer_args(buffer_args, log_dir):
    try:
        os.makedirs(log_dir)
    except:
        print("NOTE: Potentially overriding previous buffer args in %s" % log_dir)
    # Save the buffer args (based on ddpg.py in baselines)
    with open(os.path.join(log_dir, 'buffer_args.pkl'), 'wb') as f:
        pickle.dump(buffer_args, f)

def load_buffer_args(log_dir):
    with open(os.path.join(log_dir, 'buffer_args.pkl'), 'rb') as f:
        loaded_buffer_args = pickle.load(f)
    return loaded_buffer_args