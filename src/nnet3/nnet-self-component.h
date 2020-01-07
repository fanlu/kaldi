// nnet3/nnet-self-component.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_NNET3_NNET_SELF_COMPONENT_H_
#define KALDI_NNET3_NNET_SELF_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {
/*
  Class SelfAttentionComponent is used together with
  StatisticsExtractionComponent to extract weighted mean and
  standard-deviation statistics.

  SelfAttentionComponent is a part of the self-attention mechanism, which does 
  a linear combination of a series of input frames

 # In SelfAttentionComponent, the first n columns of the input matrix are interpreted
 # as the weight vectors in multi-head attentions. If only a single-head attention is used,
 # then the first column of the input matrix is the weight vector. The n + 1 th column of
 # the input matrix is the count from the extraction component, it will not be used in 
 # this component.
 # The tricky here is that a softmax computation is involved in this component because
 # the weights are arrange as column vector of the matrix, so we can't use the original SoftMax
 # component and have to do the softmax copmutation here.
 # We need to transpose the weights to row vector, then call ApplySoftMaxPerRow(),
 # then transpose the weights back to column vector.

 configs and their defaults:  input-dim=-1, input-period=1, left-context=-1, right-context=-1,
    num-heads=1, num-log-count-features=0, output-stddevs=true

 */

class SelfAttentionComponent: public Component {
 public:
  // Initializes to defaults which would not pass Check(); use InitFromConfig()
  // or Read() or copy constructor to really initialize.
  SelfAttentionComponent();
  // copy constructor, used in Copy()
  SelfAttentionComponent(const SelfAttentionComponent &other);

  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const {
    return num_log_count_features_ + (num_heads_ ? (num_heads_ ) * (input_dim_ - 1 - num_heads_) : (input_dim_ - 1));
  }
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "SelfAttentionComponent"; }
  //TODO
  virtual int32 Properties() const {
    return kReordersIndexes|kBackpropAdds|
        (output_stddevs_ || num_log_count_features_ > 0 ?
         kBackpropNeedsOutput : 0) |
        (num_log_count_features_ == 0 || num_heads_ > 0 ? kBackpropNeedsInput : 0) |
	(num_heads_ > 0 ? kUsesMemo : 0);
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo_in,
                        Component *, // to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const {
    return new SelfAttentionComponent(*this);
  }

  virtual void DeleteMemo(void *memo) const { delete static_cast<Memo*>(memo); }
  // Some functions that are only to be reimplemented for GeneralComponents.
  virtual void GetInputIndexes(const MiscComputationInfo &misc_info,
                               const Index &output_index,
                               std::vector<Index> *desired_indexes) const;

  // returns true if at least one of its inputs is computable.
  virtual bool IsComputable(const MiscComputationInfo &misc_info,
                            const Index &output_index,
                            const IndexSet &input_index_set,
                            std::vector<Index> *used_inputs) const;

  // This function reorders the input and output indexes so that they
  // are sorted first on n and then x and then t.
  virtual void ReorderIndexes(std::vector<Index> *input_indexes,
                              std::vector<Index> *output_indexes) const;

  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const;

  struct Memo {
	CuMatrix<BaseFloat> softmax;
  };

 private:
  // Checks that the parameters are valid.
  void Check() const;

  // Disallow assignment operator.
  SelfAttentionComponent &operator =(
      const SelfAttentionComponent &other);

  int32 input_dim_;
  int32 input_period_;
  int32 left_context_;
  int32 right_context_;
  int32 num_heads_;
  int32 num_log_count_features_;
  bool output_stddevs_;
  BaseFloat variance_floor_;
};

class SelfAttentionComponentPrecomputedIndexes:
      public ComponentPrecomputedIndexes {
 public:

  // in the first stage of creating the output we sum over row ranges of
  // the input.  forward_indexes.Dim() equals the number of rows of the
  // output, and each element is a (start, end) range of inputs, that is
  // summed over.
  CuArray<Int32Pair> forward_indexes;

  // backward_indexes contains the same information as forward_indexes, but in a
  // different format.  backward_indexes.Dim() is the same as the number of rows
  // of input, and each element contains the (start,end) of the range of outputs
  // for which this input index appears as an element of the sum for that
  // output.  This is possible because of the way the inputs and outputs are
  // ordered and because of how we select the elments to appear in the sum using
  // a window.  This quantity is used in backprop.
  CuArray<Int32Pair> backward_indexes;

  virtual ~SelfAttentionComponentPrecomputedIndexes() { }

  ComponentPrecomputedIndexes *Copy() const {
    return new SelfAttentionComponentPrecomputedIndexes(*this);
  }

  virtual void Write(std::ostream &os, bool binary) const;

  virtual void Read(std::istream &is, bool binary);

  virtual std::string Type() const { return "SelfAttentionComponentPrecomputedIndexes"; }
};



} // namespace nnet3
} // namespace kaldi


#endif