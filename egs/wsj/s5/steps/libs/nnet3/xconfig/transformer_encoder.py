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

# This class is for parsing lines like
#  'attention-renorm-layer num-heads=10 value-dim=50 key-dim=50 time-stride=3 num-left-inputs=5 num-right-inputs=2.'
#
# Parameters of the class, and their defaults:
#   input='[-1]'               [Descriptor giving the input of the layer.]
#   self-repair-scale=1.0e-05  [Affects relu, sigmoid and tanh layers.]
#   learning-rate-factor=1.0   [This can be used to make the affine component
#                               train faster or slower].
#   Documentation for the rest of the parameters (related to the
#   attention component) can be found in nnet-attention-component.h


class XconfigTransformerEncoderLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        # Here we just list some likely combinations.. you can just add any
        # combinations you want to use, to this list.
        assert first_token in ['attention-addrenorm-feedforward-addrenorm-layer',
                               'attention-layer',
                               'addrenorm-feedforward-addrenorm-layer',
                               'attention-addbatchnorm-feedforward-addbatchnorm-layer']
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = { 'input':'[-1]',
                        'dim': -1,
                        'max-change' : 0.75,
                        'self-repair-scale' : 1.0e-05,
                        'target-rms' : 1.0,
                        'learning-rate-factor' : 1.0,
                        'ng-affine-options' : '',
                        'l2-regularize': 0.0,
                        'num-left-inputs-required': -1,
                        'num-right-inputs-required': -1,
                        'output-context': True,
                        'output-projection': True,
                        'time-stride': 1,
                        'num-heads': 1,
                        'key-dim': -1,
                        'key-scale': 0.0,
                        'value-dim': -1,
                        'model-dim': -1,
                        'ff-dim': -1,
                        'num-left-inputs': -1,
                        'num-right-inputs': -1,
                        'dropout-proportion': 0.5}  # dropout-proportion only
                                                    # affects layers with
                                                    # 'dropout' in the name.

    def check_configs(self):
        if self.config['self-repair-scale'] < 0.0 or self.config['self-repair-scale'] > 1.0:
            raise RuntimeError("self-repair-scale has invalid value {0}"
                               .format(self.config['self-repair-scale']))
        if self.config['target-rms'] < 0.0:
            raise RuntimeError("target-rms has invalid value {0}"
                               .format(self.config['target-rms']))
        if self.config['learning-rate-factor'] <= 0.0:
            raise RuntimeError("learning-rate-factor has invalid value {0}"
                               .format(self.config['learning-rate-factor']))
        for conf in ['value-dim', 'key-dim', 'model-dim', 'ff-dim',
                     'num-left-inputs', 'num-right-inputs']:
            if self.config[conf] < 0:
                raise RuntimeError("{0} has invalid value {1}"
                                   .format(conf, self.config[conf]))
        if self.config['key-scale'] == 0.0:
            self.config['key-scale'] = 1.0 / math.sqrt(self.config['key-dim'])

    def output_name(self, auxiliary_output=None):
        # at a later stage we might want to expose even the pre-nonlinearity
        # vectors
        assert auxiliary_output == None
        
        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        last_nonlinearity = split_layer_name[-2]
        #TODO: return something like: attention1.renorm2 (assuming that there are always 2 norm layers in each encoder block)
        return '{0}.{1}2'.format(self.name, last_nonlinearity)

    def attention_input_dim(self):
        context_dim = (self.config['num-left-inputs'] +
                       self.config['num-right-inputs'] + 1)
        num_heads = self.config['num-heads']
        key_dim = self.config['key-dim']
        value_dim = self.config['value-dim']
        query_dim = key_dim + context_dim;
        return num_heads * (key_dim + value_dim + query_dim)

    def attention_output_dim(self):
        context_dim = (self.config['num-left-inputs'] +
                       self.config['num-right-inputs'] + 1)
        num_heads = self.config['num-heads']
        value_dim = self.config['value-dim']
        return (num_heads *
                (value_dim +
                 (context_dim if self.config['output-context'] else 0)))

    def output_dim(self, auxiliary_output = None):
      return self.config['model-dim']

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
        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        nonlinearities = split_layer_name[:-1]

        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        model_dim = self.config['model-dim']
        ff_dim = self.config['ff-dim']
        wanna_projection = self.config['output-projection']

        # the child classes e.g. tdnn might want to process the input
        # before adding the other components

        return self._add_components(input_desc, input_dim, model_dim, ff_dim, nonlinearities, wanna_projection)

    def _add_components(self, input_desc, input_dim, model_dim, ff_dim, nonlinearities, wanna_projection = True):
        dim = self.attention_input_dim()
        self_repair_scale = self.config['self-repair-scale']
        target_rms = self.config['target-rms']
        max_change = self.config['max-change']
        ng_affine_options = self.config['ng-affine-options']
        l2_regularize = self.config['l2-regularize']
        learning_rate_factor=self.config['learning-rate-factor']
        learning_rate_option=('learning-rate-factor={0}'.format(learning_rate_factor)
                              if learning_rate_factor != 1.0 else '')
        l2_regularize_option = ('l2-regularize={0} '.format(l2_regularize)
                                if l2_regularize != 0.0 else '')
        configs = []

        cnt_affine_layers = 1
        cnt_norm_layers = 1
        cnt_relu_layers = 1
        print(input_desc, input_dim, model_dim)
        if not input_dim == model_dim:
            line = ('component name={0}.embedding'
                    ' type=NaturalGradientAffineComponent'
                    ' input-dim={1}'
                    ' output-dim={2}'
                    ' max-change={3}'
                    ' {4} {5} {6}'
                    ''.format(self.name, input_dim, model_dim,
                          max_change, ng_affine_options,
                          learning_rate_option, l2_regularize_option))
            configs.append(line)
            line = ('component-node name={0}.embedding'
                    ' component={0}.embedding input={1}'
                    ''.format(self.name, input_desc))
            configs.append(line)

            res_node = '{0}.embedding'.format(self.name)
        else:
            res_node = input_desc

        line = ('component name={0}.affine{1}'
                ' type=NaturalGradientAffineComponent'
                ' input-dim={2}'
                ' output-dim={3}'
                ' max-change={4}'
                ' {5} {6} {7}'
                ''.format(self.name, cnt_affine_layers, model_dim, dim,
                      max_change, ng_affine_options,
                      learning_rate_option, l2_regularize_option))
        configs.append(line)
        line = ('component-node name={0}.affine{1}'
                ' component={0}.affine{1} input={2}'
                ''.format(self.name, cnt_affine_layers, res_node))
        configs.append(line)
        
        cur_node = '{0}.affine{1}'.format(self.name, cnt_affine_layers)
        cnt_affine_layers += 1

        for nonlinearity in nonlinearities:
            if nonlinearity == 'relu':
                line = ('component name={0}.{1}{2}'
                        ' type=RectifiedLinearComponent dim={3}'
                        ' self-repair-scale={4}'
                        ''.format(self.name, nonlinearity, cnt_relu_layers, dim,
                            self_repair_scale))
                configs.append(line)
                line = ('component-node name={0}.{1}{2}'
                        ' component={0}.{1}{2} input={3}'
                        ''.format(self.name, nonlinearity, cnt_relu_layers, cur_node))
                configs.append(line)
  
                cur_node = '{0}.{1}{2}'.format(self.name, nonlinearity, cnt_relu_layers)
                cnt_relu_layers += 1

            elif nonlinearity == 'attention':
                line = ('component name={0}.{1}'
                        ' type=RestrictedAttentionComponent'
                        ' value-dim={2}'
                        ' key-dim={3}'
                        ' num-left-inputs={4}'
                        ' num-right-inputs={5}'
                        ' num-left-inputs-required={6}'
                        ' num-right-inputs-required={7}'
                        ' output-context={8}'
                        ' time-stride={9}'
                        ' num-heads={10}'
                        ' key-scale={11}'
                        ''.format(self.name, nonlinearity,
                                  self.config['value-dim'],
                                  self.config['key-dim'],
                                  self.config['num-left-inputs'],
                                  self.config['num-right-inputs'],
                                  self.config['num-left-inputs-required'],
                                  self.config['num-right-inputs-required'],
                                  self.config['output-context'],
                                  self.config['time-stride'],
                                  self.config['num-heads'],
                                  self.config['key-scale']))
                configs.append(line)
                line = ('component-node name={0}.{1}'
                        ' component={0}.{1} input={2}'
                        ''.format(self.name, nonlinearity, cur_node))
                configs.append(line)

                cur_node = '{0}.{1}'.format(self.name, nonlinearity)
           
                if wanna_projection is True:
                    dim = self.attention_output_dim()

                    line = ('component name={0}.affine{1}'
                            ' type=NaturalGradientAffineComponent'
                            ' input-dim={2}'
                            ' output-dim={3}'
                            ' max-change={4}'
                            ' {5} {6} {7}'
                            ''.format(self.name, cnt_affine_layers, dim, model_dim,
                                      max_change, ng_affine_options,
                                      learning_rate_option, l2_regularize_option))
                    configs.append(line)
                    line = ('component-node name={0}.affine{1}'
                            ' component={0}.affine{1} input={2}'
                            ''.format(self.name, cnt_affine_layers, cur_node))
                    configs.append(line)

                    cur_node = '{0}.affine{1}'.format(self.name, cnt_affine_layers)
                    cnt_affine_layers += 1

            elif nonlinearity == "feedforward":
                line = ('component name={0}.affine{1}'
                        ' type=NaturalGradientAffineComponent'
                        ' input-dim={2}'
                        ' output-dim={3}'
                        ' max-change={4}'
                        ' {5} {6} {7}'
                        ''.format(self.name, cnt_affine_layers, model_dim, ff_dim,
                                  max_change, ng_affine_options,
                                  learning_rate_option, l2_regularize_option))
                configs.append(line)
                line = ('component-node name={0}.affine{1}'
                        ' component={0}.affine{1} input={2}'
                        ''.format(self.name, cnt_affine_layers, cur_node))
                configs.append(line)

                cur_node = '{0}.affine{1}'.format(self.name, cnt_affine_layers)
                cnt_affine_layers += 1

                line = ('component name={0}.relu{1}'
                        ' type=RectifiedLinearComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, cnt_relu_layers, ff_dim, self_repair_scale))
                configs.append(line)
                line = ('component-node name={0}.relu{1}'
                        ' component={0}.relu{1} input={2}'
                        ''.format(self.name, cnt_relu_layers, cur_node))
                configs.append(line)

                cur_node = '{0}.relu{1}'.format(self.name, cnt_relu_layers)
                cnt_relu_layers += 1

                line = ('component name={0}.affine{1}'
                        ' type=NaturalGradientAffineComponent'
                        ' input-dim={2}'
                        ' output-dim={3}'
                        ' max-change={4}'
                        ' {5} {6} {7}'
                        ''.format(self.name, cnt_affine_layers, ff_dim, model_dim,
                                  max_change, ng_affine_options,
                                  learning_rate_option, l2_regularize_option))
                configs.append(line)
                line = ('component-node name={0}.affine{1}'
                        ' component={0}.affine{1} input={2}'
                        ''.format(self.name, cnt_affine_layers, cur_node))
                configs.append(line)
                
                cur_node = '{0}.affine{1}'.format(self.name, cnt_affine_layers)

            elif nonlinearity == 'sigmoid':
                line = ('component name={0}.{1}'
                        ' type=SigmoidComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, dim,
                            self_repair_scale))
                configs.append(line)
                line = ('component-node name={0}.{1}'
                        ' component={0}.{1} input={2}'
                        ''.format(self.name, nonlinearity, cur_node))
                configs.append(line)

                cur_node = '{0}.{1}'.format(self.name, nonlinearity)

            elif nonlinearity == 'tanh':
                line = ('component name={0}.{1}'
                        ' type=TanhComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, dim,
                            self_repair_scale))
                configs.append(line)
                line = ('component-node name={0}.{1}'
                        ' component={0}.{1} input={2}'
                        ''.format(self.name, nonlinearity, cur_node))
                configs.append(line)

                cur_node = '{0}.{1}'.format(self.name, nonlinearity)
             
            elif nonlinearity == 'renorm':
                line = ('component name={0}.{1}'
                        ' type=NormalizeComponent dim={2}'
                        ' target-rms={3}'
                        ''.format(self.name, nonlinearity, dim,
                            target_rms))
                configs.append(line)
                line = ('component-node name={0}.{1}'
                        ' component={0}.{1} input={2}'
                        ''.format(self.name, nonlinearity, cur_node))
                configs.append(line)
                cur_node = '{0}.{1}'.format(self.name, nonlinearity)

            elif nonlinearity == 'addrenorm':
                line = ('component name={0}.{1}{2}'
                        ' type=NormalizeComponent dim={3}'
                        ' target-rms={4}'
                        ' add-log-stddev=false'
                        ''.format(self.name, nonlinearity, cnt_norm_layers, model_dim,
                            target_rms))
                configs.append(line)
                line = ('component-node name={0}.{1}{2}'
                        ' component={0}.{1}{2} input=Sum({3}, {4})'
                        ''.format(self.name, nonlinearity, cnt_norm_layers, res_node, cur_node))
                configs.append(line)

                res_node = '{0}.{1}{2}'.format(self.name, nonlinearity, cnt_norm_layers)
                cur_node = res_node
                cnt_norm_layers += 1

            elif nonlinearity == 'addbatchnorm':
                line = ('component name={0}.{1}{2}'
                        ' type=BatchNormComponent dim={3}'
                        ' target-rms={4}'
                        ''.format(self.name, nonlinearity, cnt_norm_layers, model_dim,
                            target_rms))
                configs.append(line)
                line = ('component-node name={0}.{1}{2}'
                        ' component={0}.{1}{2} input=Sum({3}, {4})'
                        ''.format(self.name, nonlinearity, cnt_norm_layers, res_node, cur_node))
                configs.append(line)

                res_node = '{0}.{1}{2}'.format(self.name, nonlinearity, cnt_norm_layers)
                cur_node = res_node
                cnt_norm_layers += 1

            elif nonlinearity == 'dropout':
                line = ('component name={0}.{1} type=DropoutComponent '
                           'dim={2} dropout-proportion={3}'.format(
                               self.name, nonlinearity, dim,
                               self.config['dropout-proportion']))
                configs.append(line)
                line = ('component-node name={0}.{1}'
                        ' component={0}.{1} input={2}'
                        ''.format(self.name, nonlinearity, cur_node))
                configs.append(line)

                cur_node = '{0}.{1}'.format(self.name, nonlinearity)

            else:
                raise RuntimeError("Unknown nonlinearity type: {0}"
                                   .format(nonlinearity))

        return configs
