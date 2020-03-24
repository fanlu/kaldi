#!/usr/bin/env python3

# Copyright 2020 Xiaomi Corporation, Beijing, China (author: Haowen Qiu)
# Apache 2.0

import logging
import os

import numpy as np
import torch


def allocate_gpu_devices(num: int = 1):
    """ Allocate GPU devices

    Args:
        num: the number of devices we want to allocate

    Returns:
        device_list: list of pair (allocated_device_id, tensor)

    """
    device_list = []
    if not torch.cuda.is_available():
        return dev_list
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        if len(device_list) >= num:
            break
        try:
            # If cuda cards are in 'Exclusive_Process' mode, we can 
            # try to get one device by putting a tensor on it. 
            device = torch.device('cuda', i)
            tensor = torch.zeros(1)
            tensor = tensor.to(device)
            device_list.append((i, tensor))
        except RuntimeError:  # CUDA error: all CUDA-capable devices are busy or unavailable
            logging.info(
                'Failed to select GPU {} out of {} (starting from 0), skip it'
                .format(i, device_count - 1))
            pass
    return device_list


def get_init_method(local_rank, dir):
    if local_rank == 0:
        init_method = 'tcp://{}:7275'.format(os.environ["HOSTNAME"])
        with open('{}/SGE_MASTER'.format(dir), 'w') as f:
            f.write(init_method)
    else:
        import time
        while not os.path.exists('{}/SGE_MASTER'.format(dir)):
            time.sleep(5)
        with open('{}/SGE_MASTER'.format(dir), 'r') as f:
            init_method = f.readlines()[0]
    logging.info('local_rank: {}, {}'.format(local_rank, init_method))
    return init_method
