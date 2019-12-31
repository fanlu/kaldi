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
#  'mssa-layer name=attention multi-stride=1,3,5 num-heads=4 value-dim=60 key-dim=40 model-dim=256 ff-dim=1024 num-left-inputs=5 num-right-inputs=5 output-context=false'
#
# Parameters of the class, and their defaults:
#   input='[-1]'               [Descriptor giving the input of the layer.]
#   self-repair-scale=1.0e-05  [Affects relu, sigmoid and tanh layers.]
#   learning-rate-factor=1.0   [This can be used to make the affine component
#                               train faster or slower].
#   Documentation for the rest of the parameters (related to the
#   attention component) can be found in nnet-attention-component.h


class XconfigMultiStrideSelfAttentionLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        # Here we just list some likely combinations.. you can just add any
        # combinations you want to use, to this list.
        assert first_token in ['mssa-layer']
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
                        'multi-stride': '1,3,5',
                        'time-stride': -1,
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
        if self.config['time-stride'] < 0:
            return '{0}.dropout'.format(self.name)
        else:
            return '{0}.addrenorm2'.format(self.name)

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

        # the child classes e.g. tdnn might want to process the input
        # before adding the other components

        return self._add_components(input_desc, input_dim, model_dim, ff_dim, nonlinearities)

    def _add_components(self, input_desc, input_dim, model_dim, ff_dim, nonlinearities):
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

        # Check whether self-attention has multiple strides
        if self.config['time-stride'] < 0: 
            cnt = 0
            strides = self.config['multi-stride'].split(',')
            for stride in strides:
                cnt += 1
                # Embedding layer
                line = ('component name={0}-{1}.embedding'
                        ' type=NaturalGradientAffineComponent'
                        ' input-dim={2}'
                        ' output-dim={3}'
                        ' max-change={4}'
                        ' {5} {6} {7}'
                        ''.format(self.name, cnt, input_dim, model_dim, 
                                  max_change, ng_affine_options, learning_rate_option, l2_regularize_option))
                configs.append(line)
                line = ('component-node name={0}-{1}.embedding'
                        ' component={0}-{1}.embedding input={2}'
                        ''.format(self.name, cnt, input_desc))
                configs.append(line)
                res_node = '{0}-{1}.embedding'.format(self.name, cnt)
                # Affine layer before self-attention
                dim = self.attention_input_dim()
                line = ('component name={0}-{1}.affine1'
                        ' type=NaturalGradientAffineComponent'
                        ' input-dim={2}'
                        ' output-dim={3}'
                        ' max-change={4}'
                        ' {5} {6} {7}'
                        ''.format(self.name, cnt, model_dim, dim,
                                  max_change, ng_affine_options, learning_rate_option, l2_regularize_option))
                configs.append(line)
                line = ('component-node name={0}-{1}.affine1'
                        ' component={0}-{1}.affine1 input={2}'
                        ''.format(self.name, cnt, res_node))
                configs.append(line)
                cur_node = '{0}-{1}.affine1'.format(self.name, cnt)
                # Self-attention
                line = ('component name={0}-{1}.attention'
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
                        ''.format(self.name, cnt, 
                                  self.config['value-dim'],
                                  self.config['key-dim'],
                                  self.config['num-left-inputs'],
                                  self.config['num-right-inputs'],
                                  self.config['num-left-inputs-required'],
                                  self.config['num-right-inputs-required'],
                                  self.config['output-context'],
                                  stride,
                                  self.config['num-heads'],
                                  self.config['key-scale']))
                configs.append(line)
                line = ('component-node name={0}-{1}.attention'
                        ' component={0}-{1}.attention input={2}'
                        ''.format(self.name, cnt, cur_node))
                configs.append(line)
                cur_node = '{0}-{1}.attention'.format(self.name, cnt)
                # Affine layer after self-attention
                dim = self.attention_output_dim()
                line = ('component name={0}-{1}.affine2'
                        ' type=NaturalGradientAffineComponent'
                        ' input-dim={2}'
                        ' output-dim={3}'
                        ' max-change={4}'
                        ' {5} {6} {7}'
                        ''.format(self.name, cnt, dim, model_dim,
                                  max_change, ng_affine_options, learning_rate_option, l2_regularize_option))
                configs.append(line)
                line = ('component-node name={0}-{1}.affine2'
                        ' component={0}-{1}.affine2 input={2}'
                        ''.format(self.name, cnt, cur_node))
                configs.append(line)
                cur_node = '{0}-{1}.affine2'.format(self.name, cnt)
                # Add-norm layer 1 
                line = ('component name={0}-{1}.addrenorm1'
                        ' type=NormalizeComponent dim={2}'
                        ' target-rms={3}'
                        ' add-log-stddev=false'
                        ''.format(self.name, cnt, model_dim, target_rms))
                configs.append(line)
                line = ('component-node name={0}-{1}.addrenorm1'
                        ' component={0}-{1}.addrenorm1 input=Sum({2}, {3})'
                        ''.format(self.name, cnt, res_node, cur_node))
                configs.append(line)
                res_node = '{0}-{1}.addrenorm1'.format(self.name, cnt)
                cur_node = res_node
                # Affine layer 1 in FF before ReLu
                line = ('component name={0}-{1}.affine3'
                        ' type=NaturalGradientAffineComponent'
                        ' input-dim={2}'
                        ' output-dim={3}'
                        ' max-change={4}'
                        ' {5} {6} {7}'
                        ''.format(self.name, cnt, model_dim, ff_dim,
                                  max_change, ng_affine_options, learning_rate_option, l2_regularize_option))
                configs.append(line)
                line = ('component-node name={0}-{1}.affine3'
                        ' component={0}-{1}.affine3 input={2}'
                        ''.format(self.name, cnt, cur_node))
                configs.append(line)
                cur_node = '{0}-{1}.affine3'.format(self.name, cnt)
                # ReLu in FF
                line = ('component name={0}-{1}.relu1'
                        ' type=RectifiedLinearComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, cnt, ff_dim, self_repair_scale))
                configs.append(line)
                line = ('component-node name={0}-{1}.relu1'
                        ' component={0}-{1}.relu1 input={2}'
                        ''.format(self.name, cnt, cur_node))
                configs.append(line)
                cur_node = '{0}-{1}.relu1'.format(self.name, cnt)
                # Affine layer 2 in FF after ReLu
                line = ('component name={0}-{1}.affine4'
                        ' type=NaturalGradientAffineComponent'
                        ' input-dim={2}'
                        ' output-dim={3}'
                        ' max-change={4}'
                        ' {5} {6} {7}'
                        ''.format(self.name, cnt, ff_dim, model_dim,
                                  max_change, ng_affine_options, learning_rate_option, l2_regularize_option))
                configs.append(line)
                line = ('component-node name={0}-{1}.affine4'
                        ' component={0}-{1}.affine4 input={2}'
                        ''.format(self.name, cnt, cur_node))
                configs.append(line)
                cur_node = '{0}-{1}.affine4'.format(self.name, cnt)
                 # Add-norm layer 2
                line = ('component name={0}-{1}.addrenorm2'
                        ' type=NormalizeComponent dim={2}'
                        ' target-rms={3}'
                        ' add-log-stddev=false'
                        ''.format(self.name, cnt, model_dim, target_rms))
                configs.append(line)
                line = ('component-node name={0}-{1}.addrenorm2'
                        ' component={0}-{1}.addrenorm2 input=Sum({2}, {3})'
                        ''.format(self.name, cnt, res_node, cur_node))
                configs.append(line)
                cur_node = '{0}-{1}.addrenorm2'.format(self.name, cnt)
            # Final projection layer
            dim = len(strides) * int(model_dim)
            line = ('component name={0}.affine'
                    ' type=NaturalGradientAffineComponent'
                    ' input-dim={1}'
                    ' output-dim={2}'
                    ' max-change={3}'
                    ' {4} {5} {6}'
                    ''.format(self.name, dim, model_dim,
                              max_change, ng_affine_options, learning_rate_option, l2_regularize_option))
            configs.append(line)
            line = ('component-node name={0}.affine'
                    ' component={0}.affine input=Append'
                    ''.format(self.name))
            line += '('
            cnt = 0
            for stride in strides:
                cnt += 1
                line += '{0}-{1}.addrenorm2'.format(self.name, cnt)
                if cnt is not len(strides):
                    line += ', '
            line += ')' 
            configs.append(line)
            cur_node = '{0}.affine'.format(self.name)
            # Final ReLu layer 
            line = ('component name={0}.relu'
                    ' type=RectifiedLinearComponent dim={1}'
                    ' self-repair-scale={2}'
                    ''.format(self.name, model_dim, self_repair_scale))
            configs.append(line)
            line = ('component-node name={0}.relu'
                    ' component={0}.relu input={1}'
                    ''.format(self.name, cur_node))
            configs.append(line)
            cur_node = '{0}.relu'.format(self.name)
            # Final batchnorm layer
            line = ('component name={0}.batchnorm'
                    ' type=BatchNormComponent dim={1}'
                    ' target-rms={2}'
                    ''.format(self.name, model_dim,
                              target_rms))
            configs.append(line)
            line = ('component-node name={0}.batchnorm'
                    ' component={0}.batchnorm input={1}'
                    ''.format(self.name, cur_node))
            configs.append(line)
            cur_node = '{0}.batchnorm'.format(self.name)
            # Final dropout layer 
            line = ('component name={0}.dropout'
                    ' type=GeneralDropoutComponent dim={1}'
                    ' dropout-proportion={2} continuous=true'
                    ''.format(self.name, model_dim, self.config['dropout-proportion']))
            configs.append(line)
            line = ('component-node name={0}.dropout'
                    ' component={0}.dropout input={1}'
                    ''.format(self.name, cur_node))
            configs.append(line)
            cur_node = '{0}.dropout'.format(self.name)
        else:
            if not input_dim == model_dim:
                # Embedding layer
                line = ('component name={0}.embedding'
                        ' type=NaturalGradientAffineComponent'
                        ' input-dim={1}'
                        ' output-dim={2}'
                        ' max-change={3}'
                        ' {4} {5} {6}'
                        ''.format(self.name, input_dim, model_dim,
                                  max_change, ng_affine_options, learning_rate_option, l2_regularize_option))
                configs.append(line)
                line = ('component-node name={0}.embedding'
                        ' component={0}.embedding input={1}'
                        ''.format(self.name, input_desc))
                configs.append(line) 
                res_node = '{0}.embedding'.format(self.name)
            else:
                res_node = input_desc            
            # Affine layer before self-attention
            dim = self.attention_input_dim()
            line = ('component name={0}.affine1'
                    ' type=NaturalGradientAffineComponent'
                    ' input-dim={1}'
                    ' output-dim={2}'
                    ' max-change={3}'
                    ' {4} {5} {6}'
                    ''.format(self.name, model_dim, dim,
                              max_change, ng_affine_options, learning_rate_option, l2_regularize_option))
            configs.append(line)
            line = ('component-node name={0}.affine1'
                    ' component={0}.affine1 input={1}'
                    ''.format(self.name, res_node))
            configs.append(line)
            cur_node = '{0}.affine1'.format(self.name)
            # Self-attention
            line = ('component name={0}.attention'
                    ' type=RestrictedAttentionComponent'
                    ' value-dim={1}'
                    ' key-dim={2}'
                    ' num-left-inputs={3}'
                    ' num-right-inputs={4}'
                    ' num-left-inputs-required={5}'
                    ' num-right-inputs-required={6}'
                    ' output-context={7}'
                    ' time-stride={8}'
                    ' num-heads={9}'
                    ' key-scale={10}'
                    ''.format(self.name,
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
            line = ('component-node name={0}.attention'
                    ' component={0}.attention input={1}'
                    ''.format(self.name, cur_node))
            configs.append(line)
            cur_node = '{0}.attention'.format(self.name)
            # Affine layer after self-attention
            dim = self.attention_output_dim()
            line = ('component name={0}.affine2'
                    ' type=NaturalGradientAffineComponent'
                    ' input-dim={1}'
                    ' output-dim={2}'
                    ' max-change={3}'
                    ' {4} {5} {6}'
                    ''.format(self.name, dim, model_dim,
                              max_change, ng_affine_options, learning_rate_option, l2_regularize_option))
            configs.append(line)
            line = ('component-node name={0}.affine2'
                    ' component={0}.affine2 input={1}'
                    ''.format(self.name, cur_node))
            configs.append(line)
            cur_node = '{0}.affine2'.format(self.name)
            # Add-norm layer 1
            line = ('component name={0}.addrenorm1'
                    ' type=NormalizeComponent dim={1}'
                    ' target-rms={2}'
                    ' add-log-stddev=false'
                    ''.format(self.name, model_dim, target_rms))
            configs.append(line)
            line = ('component-node name={0}.addrenorm1'
                    ' component={0}.addrenorm1 input=Sum({1}, {2})'
                    ''.format(self.name, res_node, cur_node))
            configs.append(line)
            res_node = '{0}.addrenorm1'.format(self.name)
            cur_node = res_node
            # Affine layer 1 in FF before ReLu
            line = ('component name={0}.affine3'
                    ' type=NaturalGradientAffineComponent'
                    ' input-dim={1}'
                    ' output-dim={2}'
                    ' max-change={3}'
                    ' {4} {5} {6}'
                    ''.format(self.name, model_dim, ff_dim,
                              max_change, ng_affine_options, learning_rate_option, l2_regularize_option))
            configs.append(line)
            line = ('component-node name={0}.affine3'
                    ' component={0}.affine3 input={1}'
                    ''.format(self.name, cur_node))
            configs.append(line)
            cur_node = '{0}.affine3'.format(self.name)
            # ReLu in FF
            line = ('component name={0}.relu1'
                    ' type=RectifiedLinearComponent dim={1}'
                    ' self-repair-scale={2}'
                    ''.format(self.name, ff_dim, self_repair_scale))
            configs.append(line)
            line = ('component-node name={0}.relu1'
                    ' component={0}.relu1 input={1}'
                    ''.format(self.name, cur_node))
            configs.append(line)
            cur_node = '{0}.relu1'.format(self.name)
            # Affine layer 2 in FF after ReLu
            line = ('component name={0}.affine4'
                    ' type=NaturalGradientAffineComponent'
                    ' input-dim={1}'
                    ' output-dim={2}'
                    ' max-change={3}'
                    ' {4} {5} {6}'
                    ''.format(self.name, ff_dim, model_dim,
                              max_change, ng_affine_options, learning_rate_option, l2_regularize_option))
            configs.append(line)
            line = ('component-node name={0}.affine4'
                    ' component={0}.affine4 input={1}'
                    ''.format(self.name, cur_node))
            configs.append(line)
            cur_node = '{0}.affine4'.format(self.name)
            # Add-norm layer 2
            line = ('component name={0}.addrenorm2'
                    ' type=NormalizeComponent dim={1}'
                    ' target-rms={2}'
                    ' add-log-stddev=false'
                    ''.format(self.name, model_dim, target_rms))
            configs.append(line)
            line = ('component-node name={0}.addrenorm2'
                    ' component={0}.addrenorm2 input=Sum({1}, {2})'
                    ''.format(self.name, res_node, cur_node))
            configs.append(line)
            cur_node = '{0}.addrenorm2'.format(self.name)

        return configs
