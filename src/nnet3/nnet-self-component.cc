// nnet3/nnet-self-component.cc

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

#include <iterator>
#include <sstream>
#include <iomanip>
#include "nnet3/nnet-general-component.h"
#include "nnet3/nnet-self-component.h"
#include "nnet3/nnet-computation-graph.h"
#include "nnet3/nnet-parse.h"

namespace kaldi {
namespace nnet3 {

// used in I/O
static void CopyPairVector(const CuArray<Int32Pair> &in,
                        std::vector<std::pair<int32, int32> > *out) {
  in.CopyToVec(reinterpret_cast<std::vector<Int32Pair>*>(out));
}
// used in I/O
static void CopyPairVector(const std::vector<std::pair<int32, int32> > &in,
                        CuArray<Int32Pair> *out) {
  const std::vector<Int32Pair> *in_cast =
      reinterpret_cast<const std::vector<Int32Pair>*>(&in);
  out->CopyFromVec(*in_cast);
}

void SelfAttentionComponentPrecomputedIndexes::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SelfAttentionComponentPrecomputedIndexes>");
  WriteToken(os, binary, "<ForwardIndexes>");
  std::vector<std::pair<int32, int32> > indexes_cpu;
  CopyPairVector(forward_indexes, &indexes_cpu);
  WriteIntegerPairVector(os, binary, indexes_cpu);
  WriteToken(os, binary, "<BackwardIndexes>");
  CopyPairVector(backward_indexes, &indexes_cpu);
  WriteIntegerPairVector(os, binary, indexes_cpu);
  WriteToken(os, binary, "</SelfAttentionComponentPrecomputedIndexes>");
}

void SelfAttentionComponentPrecomputedIndexes::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary,
                       "<SelfAttentionComponentPrecomputedIndexes>",
                       "<ForwardIndexes>");
  std::vector<std::pair<int32, int32> > indexes_cpu;
  ReadIntegerPairVector(is, binary, &indexes_cpu);
  CopyPairVector(indexes_cpu, &forward_indexes);
  ExpectToken(is, binary, "<BackwardIndexes>");
  ReadIntegerPairVector(is, binary, &indexes_cpu);
  CopyPairVector(indexes_cpu, &backward_indexes);
  ExpectToken(is, binary, "</SelfAttentionComponentPrecomputedIndexes>");
}

void SelfAttentionComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = cfl->GetValue("input-dim", &input_dim_);
  cfl->GetValue("input-period", &input_period_);
  cfl->GetValue("left-context", &left_context_);
  cfl->GetValue("right-context", &right_context_);
  cfl->GetValue("num-heads", &num_heads_);
  cfl->GetValue("num-log-count-features", &num_log_count_features_);
  cfl->GetValue("output-stddevs", &output_stddevs_);
  cfl->GetValue("variance-floor", &variance_floor_);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  // do some basic checks here but Check() will check more completely.
  if (!ok || input_dim_ <= 0 || left_context_ + right_context_ <= 0 ||
      num_log_count_features_ < 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Check();
}

SelfAttentionComponent::SelfAttentionComponent():
    input_dim_(-1), input_period_(1), left_context_(-1), right_context_(-1),
    num_heads_(0),
    num_log_count_features_(0), output_stddevs_(false),
    variance_floor_(1.0e-10) { }


SelfAttentionComponent::SelfAttentionComponent(
    const SelfAttentionComponent &other):
    input_dim_(other.input_dim_), input_period_(other.input_period_),
    left_context_(other.left_context_), right_context_(other.right_context_),
    num_heads_(other.num_heads_),
    num_log_count_features_(other.num_log_count_features_),
    output_stddevs_(other.output_stddevs_),
    variance_floor_(1.0e-10) {
  Check();
}

void SelfAttentionComponent::Check() const {
  KALDI_ASSERT(input_dim_ > 0);
  KALDI_ASSERT(input_period_ > 0);
  KALDI_ASSERT(left_context_ >= 0 && right_context_ >= 0 &&
               left_context_ + right_context_ > 0);
  KALDI_ASSERT(left_context_ % input_period_ == 0 &&
               right_context_ % input_period_ == 0);
  KALDI_ASSERT(variance_floor_ > 0.0 && variance_floor_ < 1.0);
//  KALDI_ASSERT(!output_stddevs_ || (input_dim_ - 1) % 2 == 0);
}

void SelfAttentionComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SelfAttentionComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<InputPeriod>");
  ReadBasicType(is, binary, &input_period_);
  ExpectToken(is, binary, "<LeftContext>");
  ReadBasicType(is, binary, &left_context_);
  ExpectToken(is, binary, "<RightContext>");
  ReadBasicType(is, binary, &right_context_);
  ExpectToken(is, binary, "<NumHeads>");
  ReadBasicType(is, binary, &num_heads_);
  ExpectToken(is, binary, "<NumLogCountFeatures>");
  ReadBasicType(is, binary, &num_log_count_features_);
  ExpectToken(is, binary, "<OutputStddevs>");
  ReadBasicType(is, binary, &output_stddevs_);
  ExpectToken(is, binary, "<VarianceFloor>");
  ReadBasicType(is, binary, &variance_floor_);
  ExpectToken(is, binary, "</SelfAttentionComponent>");
  Check();
}

void SelfAttentionComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SelfAttentionComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<InputPeriod>");
  WriteBasicType(os, binary, input_period_);
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context_);
  WriteToken(os, binary, "<RightContext>");
  WriteBasicType(os, binary, right_context_);
  WriteToken(os, binary, "<NumHeads>");
  WriteBasicType(os, binary, num_heads_);
  WriteToken(os, binary, "<NumLogCountFeatures>");
  WriteBasicType(os, binary, num_log_count_features_);
  WriteToken(os, binary, "<OutputStddevs>");
  WriteBasicType(os, binary, output_stddevs_);
  WriteToken(os, binary, "<VarianceFloor>");
  WriteBasicType(os, binary, variance_floor_);
  WriteToken(os, binary, "</SelfAttentionComponent>");
}

void SelfAttentionComponent::ReorderIndexes(
    std::vector<Index> *input_indexes,
    std::vector<Index> *output_indexes) const {
    std::sort(input_indexes->begin(), input_indexes->end(),
              IndexLessNxt());
    std::sort(output_indexes->begin(), output_indexes->end(),
              IndexLessNxt());
}

void SelfAttentionComponent::GetInputIndexes(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    std::vector<Index> *desired_indexes) const {
  desired_indexes->clear();
  Index input_index(output_index);
  int32 middle_t = output_index.t,
      t_start = middle_t - left_context_,
      t_last = middle_t + right_context_;
  KALDI_ASSERT(middle_t % input_period_ == 0);
  for (int32 t = t_start; t <= t_last; t += input_period_) {
    input_index.t = t;
    desired_indexes->push_back(input_index);
  }
}

bool SelfAttentionComponent::IsComputable(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    const IndexSet &input_index_set,
    std::vector<Index> *used_inputs) const {
  if (used_inputs)
    used_inputs->clear();
  // you are not supposed to access the output of this component other than at
  // multiples of the input period.  We could make this an error but decided to
  // just have it return false.
  if (output_index.t % input_period_ != 0)
    return false;

  Index input_index(output_index);
  int32 output_t = output_index.t,
      t_start = output_t - left_context_,
      t_last = output_t + right_context_;
  if (!used_inputs) {
    for (int32 t = t_start; t <= t_last; t += input_period_) {
      input_index.t = t;
      if (input_index_set(input_index))
        return true;
    }
    return false;
  } else {
    bool ans = false;
    for (int32 t = t_start; t <= t_last; t += input_period_) {
      input_index.t = t;
      if (input_index_set(input_index)) {
        ans = true;
        used_inputs->push_back(input_index);
      }
    }
    return ans;
  }
}

ComponentPrecomputedIndexes*
SelfAttentionComponent::PrecomputeIndexes(
    const MiscComputationInfo &misc_info,
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    bool need_backprop) const {
  int32 num_input_indexes = input_indexes.size(),
      num_output_indexes = output_indexes.size();
  SelfAttentionComponentPrecomputedIndexes *ans = new
      SelfAttentionComponentPrecomputedIndexes();

  Int32Pair invalid_pair;
  invalid_pair.first = -1;
  invalid_pair.second = -1;
  // forward_indexes_cpu[i] will be the (begin, end) of input indexes
  // included in the sum for the i'th output index.
  std::vector<Int32Pair> forward_indexes_cpu(num_output_indexes,
                                             invalid_pair);
  // backward_indexes_cpu[i] will be the (begin, end) of output indexes
  // for which the i'th input index participates in the sum.
  // because of the way the indexes are sorted (and the fact that only
  // required indexes are present at the input), it naturally has this
  // structure [i.e. no gaps in the sets of indexes].
  std::vector<Int32Pair> backward_indexes_cpu(num_input_indexes,
                                              invalid_pair);

  // this map maps from Index to the position in 'input_indexes'.
  unordered_map<Index, int32, IndexHasher> index_to_input_pos;
  for (int32 i = 0; i < num_input_indexes; i++)
    index_to_input_pos[input_indexes[i]] = i;

  for (int32 i = 0; i < num_output_indexes; i++) {
    Index input_index(output_indexes[i]);
    int32 middle_t = input_index.t,
        t_start = middle_t - left_context_,
        t_last = middle_t + right_context_;
    for (int32 t = t_start; t <= t_last; t += input_period_) {
      input_index.t = t;
      unordered_map<Index, int32, IndexHasher>::iterator iter =
          index_to_input_pos.find(input_index);
      if (iter != index_to_input_pos.end()) {
        int32 input_pos = iter->second;
        if (forward_indexes_cpu[i].first == -1) {
          forward_indexes_cpu[i].first = input_pos;
          forward_indexes_cpu[i].second = input_pos + 1;
        } else {
          KALDI_ASSERT(forward_indexes_cpu[i].second == input_pos);
          forward_indexes_cpu[i].second++;
        }
        if (backward_indexes_cpu[input_pos].first == -1) {
          backward_indexes_cpu[input_pos].first = i;
          backward_indexes_cpu[input_pos].second = i + 1;
        } else {
          KALDI_ASSERT(backward_indexes_cpu[input_pos].second == i);
          backward_indexes_cpu[input_pos].second++;
        }
      }
    }
    KALDI_ASSERT(forward_indexes_cpu[i].first != -1);
  }
  for (int32 i = 0; i < num_input_indexes; i++) {
    KALDI_ASSERT(backward_indexes_cpu[i].first != -1);
  }

  ans->forward_indexes = forward_indexes_cpu;
  if (need_backprop)
    ans->backward_indexes = backward_indexes_cpu;
  return ans;
}

void* SelfAttentionComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  out->SetZero();
  KALDI_ASSERT(indexes_in != NULL);
  const SelfAttentionComponentPrecomputedIndexes *indexes =
      dynamic_cast<const SelfAttentionComponentPrecomputedIndexes*>(indexes_in);
  int32 num_rows_out = out->NumRows();
  KALDI_ASSERT(indexes != NULL &&
               indexes->forward_indexes.Dim() == num_rows_out &&
               in.NumCols() == input_dim_ &&
               out->NumCols() == OutputDim());

  int32 num_minibatches = out->NumRows();
  int32 context_len = in.NumRows() / num_minibatches;

  // Here we transpose the matrix, repack it and do the softmax
  CuMatrix<BaseFloat> softmax_inmat(num_heads_ * num_minibatches, context_len);
  CuMatrix<BaseFloat> softmax_outmat(num_heads_ * num_minibatches, context_len);
  for (int i = 0; i < num_minibatches ; i++) {
    // Now the weights are appended after extraction component, so we start from the first column
    softmax_inmat.RowRange(i * num_heads_, num_heads_).CopyFromMat(in.Range(i * context_len, context_len, 0, num_heads_), kTrans);
  }
  // softmax_outmat.ApplySoftMaxPerRow(softmax_inmat);
  softmax_outmat.SoftMaxPerRow(softmax_inmat);
  // This floor on the output helps us deal with almost-zeros in a way that doesn't lead to overflow.
  softmax_outmat.ApplyFloor(1.0e-20);

  Memo *memo = new Memo();
  memo->softmax.Resize(num_heads_ * num_minibatches, context_len);
  memo->softmax.CopyFromMat(softmax_outmat);

  // feature_dim may include the second order x^2 information from the extraction component
  int32 feature_dim = input_dim_ - 1 - num_heads_;
  CuMatrix<BaseFloat> inmat_forscale(context_len * num_minibatches, num_heads_ * feature_dim);
  for (int h = 0; h < num_heads_ ; h++) {
    CuVector<BaseFloat> weights(context_len * num_minibatches);
    for (int i = 0; i < num_minibatches ; i++)
      weights.Range(i * context_len, context_len).CopyRowsFromMat(softmax_outmat.RowRange(i * num_heads_ + h, 1));
      
    inmat_forscale.ColRange(h * feature_dim, feature_dim).CopyFromMat(in.ColRange(num_heads_ + 1, feature_dim));
    inmat_forscale.ColRange(h * feature_dim, feature_dim).MulRowsVec(weights);
  }
  //this line compute the mean by summing for all minibatches and all heads at a time
  out->AddRowRanges(inmat_forscale, indexes->forward_indexes);

  if (output_stddevs_) {
    KALDI_ASSERT((input_dim_ - 1 - num_heads_) % 2 == 0);
    int32 encoder_dim = (input_dim_ - 1 - num_heads_) / 2;
    for (int h = 0; h < num_heads_ ; h++) {
      CuSubMatrix<BaseFloat> mean(*out, 0, num_rows_out, h * encoder_dim * 2, encoder_dim);
      CuSubMatrix<BaseFloat> variance(*out, 0, num_rows_out, h * encoder_dim * 2 + encoder_dim, encoder_dim);
      variance.AddMatMatElements(-1.0, mean, mean, 1.0);
      variance.ApplyFloor(variance_floor_);
      // compute the standard deviation via square root.
      variance.ApplyPow(0.5);
    }
  } 
  return static_cast<void*>(memo);
}

void SelfAttentionComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv_in,
    void *memo_in,
    Component *, // to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(indexes_in != NULL);
  const SelfAttentionComponentPrecomputedIndexes *indexes =
      dynamic_cast<const SelfAttentionComponentPrecomputedIndexes*>(
          indexes_in);
  int32 num_rows_out = out_deriv_in.NumRows();
  CuMatrix<BaseFloat> out_deriv(out_deriv_in);

  Memo *memo = static_cast<Memo*> (memo_in);
  KALDI_ASSERT(memo != NULL);
  const CuMatrix<BaseFloat> &softmax_mat = memo->softmax; 
  int32 num_minibatches = memo->softmax.NumRows() / num_heads_;
  int32 context_len = in_value.NumRows() / num_minibatches;
  // feature_dim may include the second order x^2 information from the extraction component
  int32 feature_dim = input_dim_ - 1 - num_heads_;

  if (output_stddevs_) {
    int32 encoder_dim = (input_dim_ - 1 - num_heads_) / 2;
    for (int h = 0; h < num_heads_; h++) {
      CuSubMatrix<BaseFloat> mean_deriv(out_deriv, 0, num_rows_out,
                                    h * encoder_dim * 2, encoder_dim),
      variance_deriv(out_deriv, 0, num_rows_out,
                     h * encoder_dim * 2 + encoder_dim, encoder_dim),
      mean_value(out_value, 0, num_rows_out,
                 h * encoder_dim * 2, encoder_dim),
      stddev_value(out_value, 0, num_rows_out,
                   h * encoder_dim * 2 + encoder_dim, encoder_dim);

      variance_deriv.DivElements(stddev_value);
      variance_deriv.Scale(0.5);

      mean_deriv.AddMatMatElements(-2.0, mean_value, variance_deriv, 1.0);
    }
  }

  //compute the derivative w.r.t value before softmax
  CuMatrix<BaseFloat> weight_deriv_after_softmax(num_heads_ * num_minibatches, context_len);
  CuMatrix<BaseFloat> out_deriv_per_minibat(num_heads_, feature_dim);
  CuMatrix<BaseFloat> in_value_per_minibat(context_len, feature_dim);
  for (int i = 0; i < num_minibatches; i++) {
    in_value_per_minibat.CopyFromMat(in_value.Range(i * context_len, context_len, 1 + num_heads_, feature_dim));
    for (int h = 0; h < num_heads_; h++)
      out_deriv_per_minibat.RowRange(h, 1).CopyFromMat(out_deriv.Range(i, 1, h * feature_dim, feature_dim));

    weight_deriv_after_softmax.RowRange(i * num_heads_, num_heads_).AddMatMat(1.0, out_deriv_per_minibat, kNoTrans, in_value_per_minibat, kTrans, 0);
  }
    
  CuSubMatrix<BaseFloat> weight_deriv(*in_deriv, 0, in_value.NumRows(),
                                    0, num_heads_);
  CuMatrix<BaseFloat> weight_deriv_before_softmax(num_heads_ * num_minibatches, context_len);
  weight_deriv_before_softmax.DiffSoftmaxPerRow(softmax_mat, weight_deriv_after_softmax);
  for (int i=0; i < num_minibatches; i++)
    weight_deriv.RowRange(i * context_len, context_len).CopyFromMat(weight_deriv_before_softmax.RowRange(i * num_heads_, num_heads_), kTrans);

  //compute the derivative w.r.t encoder output
  CuSubMatrix<BaseFloat> encoder_deriv(*in_deriv, 0, in_value.NumRows(),
                                    1 + num_heads_, feature_dim);
  encoder_deriv.SetZero();
  CuMatrix<BaseFloat> softmax_value(in_value.NumRows(), num_heads_);
  for(int i=0; i < num_minibatches; i++)
    softmax_value.RowRange(i * context_len, context_len).CopyFromMat(softmax_mat.RowRange(i * num_heads_, num_heads_), kTrans);
    
  for(int h = 0; h < num_heads_; h++) {
    CuMatrix<BaseFloat> encoder_deriv_per_head(in_value.NumRows(), feature_dim);
    encoder_deriv_per_head.AddRowRanges(out_deriv.ColRange(h * feature_dim, feature_dim),
                indexes->backward_indexes);
    CuVector<BaseFloat> single_head_weight(in_value.NumRows());
    single_head_weight.CopyRowsFromMat(softmax_value.ColRange(h, 1));
    encoder_deriv_per_head.MulRowsVec(single_head_weight);
    encoder_deriv.AddMat(1.0, encoder_deriv_per_head);
  }
}



} // namespace nnet3
} // namespace kaldi
