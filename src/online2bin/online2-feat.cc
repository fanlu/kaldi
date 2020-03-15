// online2bin/online2-wav-nnet2-latgen-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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

#include "feat/wave-reader.h"
#include "online2/online-nnet2-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "util/kaldi-thread.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in wav file(s) and simulates online decoding with neural nets\n"
        "(nnet2 setup), with optional iVector-based speaker adaptation and\n"
        "optional endpointing.  Note: some configuration values and inputs are\n"
        "set via config files whose filenames are passed as options\n"
        "\n"
        "Usage: online2-feat [options] "
        "<spk2utt-rspecifier> <wav-rspecifier> <feat-wspecifier>\n"
        "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
        "you want to decode utterance by utterance.\n"
        "See egs/rm/s5/local/run_online_decoding_nnet2.sh for example\n"
        "See also online2-wav-nnet2-latgen-threaded\n";

    ParseOptions po(usage);

    // feature_config includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_config;

    BaseFloat chunk_length_secs = 0.05;
    bool online = true;

    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.  Set to <= 0 "
                "to use all input in one chunk.");
    po.Register("online", &online,
                "You can set this to false to disable online iVector estimation "
                "and have all the data for each utterance used, even at "
                "utterance start.  This is useful where you just want the best "
                "results and don't care about online operation.  Setting this to "
                "false has the same effect as setting "
                "--use-most-recent-ivector=true and --greedy-ivector-extractor=true "
                "in the file given to --ivector-extraction-config, and "
                "--chunk-length=-1.");
    po.Register("num-threads-startup", &g_num_threads,
                "Number of threads used when initializing iVector extractor.");

    feature_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      return 1;
    }

    std::string spk2utt_rspecifier = po.GetArg(1),
        wav_rspecifier = po.GetArg(2),
        feat_wspecifier = po.GetArg(3);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_config);
    if (!online) {
      feature_info.ivector_extractor_info.use_most_recent_ivector = true;
      feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
      chunk_length_secs = -1.0;
    }

    Matrix<double> global_cmvn_stats;
    if (feature_info.global_cmvn_stats_rxfilename != "")
      ReadKaldiObject(feature_info.global_cmvn_stats_rxfilename,
                      &global_cmvn_stats);


    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    OnlineTimingStats timing_stats;

    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();

      OnlineIvectorExtractorAdaptationState adaptation_state(
          feature_info.ivector_extractor_info);
      OnlineCmvnState cmvn_state(global_cmvn_stats);

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!wav_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find audio for utterance " << utt;
          num_err++;
          continue;
        }
        const WaveData &wave_data = wav_reader.Value(utt);
        // get the data for channel zero (if the signal is not mono, we only
        // take the first channel).
        SubVector<BaseFloat> data(wave_data.Data(), 0);

        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        feature_pipeline.SetAdaptationState(adaptation_state);
        feature_pipeline.SetCmvnState(cmvn_state);

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length;
        if (chunk_length_secs > 0) {
          chunk_length = int32(samp_freq * chunk_length_secs);
          if (chunk_length == 0) chunk_length = 1;
        } else {
          chunk_length = std::numeric_limits<int32>::max();
        }

        int32 samp_offset = 0;
        std::vector<std::pair<int32, BaseFloat> > delta_weights;

        while (samp_offset < data.Dim()) {
          int32 samp_remaining = data.Dim() - samp_offset;
          int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                         : samp_remaining;

          SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
          feature_pipeline.AcceptWaveform(samp_freq, wave_part);

          samp_offset += num_samp;
          if (samp_offset == data.Dim()) {
            // no more input. flush out last frames
            feature_pipeline.InputFinished();
          }

        }
        int32 feat_dim = feature_pipeline.InputFeature()->Dim();
        Matrix<BaseFloat> last_feats_(feature_pipeline.InputFeature()->NumFramesReady(), feat_dim, kUndefined);
        for(int32 i = 0; i< feature_pipeline.InputFeature()->NumFramesReady(); i++){
          SubVector<BaseFloat> dest(last_feats_, i);
          feature_pipeline.InputFeature()->GetFrame(i, &dest);
        }

        // In an application you might avoid updating the adaptation state if
        // you felt the utterance had low confidence.  See lat/confidence.h
        // feature_pipeline.GetAdaptationState(&adaptation_state);
        // feature_pipeline.GetCmvnState(&cmvn_state);
        
        feat_writer.Write(utt, last_feats_);
        KALDI_LOG << "write utterance's feature " << utt;
        num_done++;
      }
    }
    timing_stats.Print(online);

    KALDI_LOG << "feat " << num_done << " utterances, "
              << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
