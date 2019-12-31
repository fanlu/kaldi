#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from distutils.util import strtobool
import json
import os
import torch
import glob
import numpy as np
import pandas as pd

def main():
  all_ckpt_info = []
  for info in glob.glob("exp/chain/train/epoch-*-info"):
    cur_dict = {}
    for line in open(info, "r").readlines():
      k, v = line.strip().split(":")
      cur_dict[k] = v.strip()
    all_ckpt_info.append(cur_dict)
  df = pd.DataFrame(all_ckpt_info)
  df = df.astype({'objf': 'float'})
  print(df.nlargest(args.num, 'objf'))
  last = df.nlargest(args.num, 'objf')['model_path']
  print("average over", last)
  avg = None
  # sum
  for path in last:
    states = torch.load(path, map_location=torch.device("cpu"))["state_dict"]
    if avg is None:
      avg = states
    else:
      for k in avg.keys():
        avg[k] += states[k]

  # average
  for k in avg.keys():
    if avg[k] is not None:
      avg[k] /= args.num
  checkpoint = {
      'state_dict': avg,
      'epoch': "avg",
      'learning_rate': "avg",
      'objf': 1
  }
  torch.save(checkpoint, args.out)


def get_parser():
  parser = argparse.ArgumentParser(description='average models from snapshot')
  parser.add_argument("--snapshots", required=False, type=str, nargs="+")
  parser.add_argument("--out", required=True, type=str)
  parser.add_argument("--num", default=5, type=int)
  parser.add_argument("--backend", default='chainer', type=str)
  parser.add_argument("--log", default=None, type=str, nargs="?")
  parser.add_argument('--avg-iters', type=strtobool, default=False,
                      help='average iters snapshot.')
  return parser


if __name__ == '__main__':
  args = get_parser().parse_args()
  main()
