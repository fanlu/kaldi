# Copyright 2017    Johns Hopkins University (Dan Povey)
#           2017    Hossein Hadian
#           2019    Kyu Han (JD.com)
# Apache 2.0.

""" This module has the implementation of attention layers.
"""

from __future__ import print_function
from __future__ import division
import math
import re
import sys
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase

"""This class is for parsing lines like
    'blocksum-layer name=blocksum1 dim=1024 input=Append(-3,0,3)'
    which will contain two components, the first component is the 'NaturalGradientPerElementScaleComponent',
    the second component is 'SumBlockComponent'. 
    This layer is developed for FSMN like architectures [need ref here].
    Parameters of the class, and their defaults:
    input='[-1]'             [Descriptor giving the input of the layer.]
    dim=-1                   [Dimension of the output]
    The following (shown with their effective defaults) are just passed through
    to the component's config line.
    l2-regularize=0.0
    glorot-init=true
"""

class XconfigBlockSumLayer(XconfigLayerBase):

  def __init__(self, first_token, key_to_value, prev_names=None):
    XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

  def set_default_configs(self):
    self.config = {'input': '[-1]',
                   'dim': -1,
                   'max-change': 0.75,
                   'glorot-init': False,  # no use for blocksum with scale factor
                   'l2-regularize': ''}

  def check_configs(self):
    if self.config['dim'] <= 0:
      raise RuntimeError("'dim' must be specified and > 0.")

  def output_name(self, auxiliary_output=None):
    assert auxiliary_output is None
    return self.name

  def output_dim(self, auxiliary_output=None):
    assert auxiliary_output is None
    assert self.config['dim'] > 0
    return self.config['dim']

  def get_full_config(self):
    ans = []
    config_lines = self._generate_config()

    for line in config_lines:
      for config_name in ['ref', 'final']:
        # we do not support user specified matrices in this layer
        # so 'ref' and 'final' configs are the same.
        ans.append((config_name, line))
    return ans

  def _generate_config(self):
    # by 'descriptor_final_string' we mean a string that can appear in
    # config-files, i.e. it contains the 'final' names of nodes.
    input_desc = self.descriptors['input']['final-string']
    input_dim = self.descriptors['input']['dim']
    output_dim = self.config['dim']

    opts = ''
    for opt_name in ['l2-regularize']:
      value = self.config[opt_name]
      if value != '':
        opts += ' {0}={1}'.format(opt_name, value)

    ng_per_element_scale_options = ""
    ng_per_element_scale_options += " max-change={0}".format(self.config['max-change'])

    if self.config['glorot-init'] is True:
      param_mean = 1.0 / (input_dim / output_dim)
      param_stddev = 1.0 / math.sqrt(input_dim / output_dim)
      ng_per_element_scale_options += " param-mean={0} param-stddev={1}".format(
          param_mean, param_stddev)
    else:
      ng_per_element_scale_options += " param-mean=0 param-stddev=1"

    pes_str = ng_per_element_scale_options
    blocksum_scale = output_dim * 1.0 / input_dim

    configs = []
    line = ('component name={0}.element_wise_scale type=NaturalGradientPerElementScaleComponent dim={1} {2} '
            '{3}'.format(self.name, input_dim, opts, pes_str))
    configs.append(line)
    line = ('component-node name={0}.element_wise_scale component={0}.element_wise_scale input={1}'.format(
        self.name, input_desc))
    configs.append(line)
    cur_node = "{0}.element_wise_scale".format(self.name)

    line = ('component name={0} type=SumBlockComponent input-dim={1} output-dim={2} '
            'scale={3}').format(self.name, input_dim, output_dim, blocksum_scale)
    configs.append(line)
    line = ('component-node name={0} component={0} input={1}').format(self.name, cur_node)
    configs.append(line)

    return configs
