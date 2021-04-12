from dataclasses import dataclass, field
from typing import List, Optional

from audio_utils.mel import TSTFTHParams


@dataclass
class ExperimentHParams():
  epochs: Optional[int] = 500
  iterations: Optional[int] = field(default_factory=int)
  # 0 if no saving, 1 for each and so on...
  iters_per_checkpoint: int = 1000
  # 0 if no saving, 1 for each and so on...
  epochs_per_checkpoint: int = 1
  seed: int = 1234
  cudnn_enabled: bool = True
  cudnn_benchmark: bool = False
  ignore_layers: List[str] = field(default_factory=list)


@dataclass
class DataHParams():
  use_saved_mels: bool = False
  cache_mels: bool = False


@dataclass
class ModelHParams():
  n_symbols: int = 0
  n_speakers: int = 0
  n_accents: int = 0
  symbols_embedding_dim: int = 512
  speakers_embedding_dim: int = 128  # 16
  #accents_embedding_dim: int = 512
  accents_use_own_symbols: bool = False
  use_speaker_embedding: bool = True

  # Encoder parameters
  encoder_kernel_size: int = 5
  encoder_n_convolutions: int = 3
  encoder_embedding_dim: int = 512

  # Decoder parameters
  n_frames_per_step: int = 1  # currently only 1 is supported
  decoder_rnn_dim: int = 1024
  prenet_dim: int = 256
  gate_threshold: float = 0.5
  p_attention_dropout: float = 0.1
  p_decoder_dropout: float = 0.1

  # Attention parameters
  attention_rnn_dim: int = 1024
  attention_dim: int = 128

  # Location Layer parameters
  attention_location_n_filters: int = 32
  attention_location_kernel_size: int = 31

  # Mel-post processing network parameters
  postnet_embedding_dim: int = 512
  postnet_kernel_size: int = 5
  postnet_n_convolutions: int = 5


@dataclass
class OptimizerHParams():
  use_saved_learning_rate: bool = False
  learning_rate: float = 1e-3
  weight_decay: float = 1e-6
  grad_clip_thresh: float = 1.0
  batch_size: int = 64
  # set model's padded outputs to padded values
  mask_padding: bool = True


@dataclass
class HParams(ExperimentHParams, DataHParams, TSTFTHParams, ModelHParams, OptimizerHParams):
  pass
