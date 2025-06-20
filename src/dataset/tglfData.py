import torch
import h5py
import numpy as np
import json
import os
from .base import BaseDataPipe

class TGLFData(BaseDataPipe):
    """
    Class to handle TGLF data with mode.
    """

    def __init__(self, cfg, num_workers, base_seed, mode):
        super().__init__(cfg, num_workers, base_seed, mode)

    
        # Load parameters from JSON file
        json_path = os.path.join(cfg.dataset_root, "qlnn_classification_data.json")
        
        # Open the file with the keys
        try:
            with open(json_path, 'r') as f:
                params = json.load(f)
            
            # Extract parameters from the JSON
            self.input_keys = params["input_keys"]              # List of all input keys
            self.target_keys = params["target_keys"]            # List of all output keys
            self.log_ops_keys = params["log_ops_keys"]
            self.target_log10_min = params["target_log10_min"]
            self.target_log10_max = params["target_log10_max"]
            self.mode_key = params["mode_key"]                  # The key for the mode data
    
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load parameters from {json_path}: {e}")

        # self.cfg = cfg
        
        # self.input_keys = cfg.input_keys    # List of all input keys
        # self.target_keys = cfg.target_keys  # List of all output keys
        # self.mode_key = cfg.mode_key        # The key for the mode data
        
        # self.log_ops_keys = cfg.log_ops_keys # List of all log ops keys

        # self.target_log10_min = cfg.target_log10_min
        # self.target_log10_max = cfg.target_log10_max

    def _read_path(self, file_path):
        try: 
            with h5py.File(file_path, 'r') as f:
                # Read the input and target data based on keys from the JSON
                input_data = np.array([f[key][:] for key in self.input_keys]).T  # Transpose for shape consistency
                target_data = np.array([f[key][:] for key in self.target_keys]).T  # Transpose for shape consistency

                mode_data = f[self.mode_key][:]  # This is your 1-hot matrix
        except OSError as e:
            raise RuntimeError(f"Failed to read file {file_path}: {e}")

        # Optionally apply log scaling to target data
        # This will scale each target feature (log10) between the min and max range
        for i, key in enumerate(self.target_keys):
            if key in self.log_ops_keys:
                target_data[:, i] = np.log10(target_data[:, i])
                target_data[:, i] = np.clip(target_data[:, i], self.target_log10_min[i], self.target_log10_max[i])


        inputs = torch.tensor(input_data, dtype=torch.float32)  # convert from  np array to tensor
        targets = torch.tensor(target_data, dtype=torch.float32)  # convert from np array to tensor
        mode = torch.tensor(mode_data, dtype=torch.float32)  # convert mode from np array to tensor

        # asinh the targets as they are too large
        targets = torch.asinh(targets)
        # print(input_data.shape[0]) #46418 data points
        return (inputs, (targets, mode)), input_data.shape[0]

    def _get_slice(self, data, index):
        """
        Get a slice of the data for a specific batch and time step.
        Handles both single tensor and tuple data formats, including nested tuples.

        Args:
            data: Single tensor or tuple of tensors (potentially nested)
            index (int): Index of the slice to retrieve

        Returns:
            Single tensor slice or tuple of slices (potentially nested)
        """
        if isinstance(data, tuple):
            result = []
            for d in data:
                if isinstance(d, tuple):
                    # Handle nested tuple
                    nested_result = tuple(nd[index] for nd in d)
                    result.append(nested_result)
                else:
                    result.append(d[index])
            return tuple(result)
        else:
            return data[index]