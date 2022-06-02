from dataclasses import dataclass, field
from typing import List, Optional

from tacotron.taco_stft import TSTFTHParams


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
  save_first_iteration: bool = True
  ignore_layers: List[str] = field(default_factory=list)


@dataclass
class DataHParams():
  use_saved_mels: bool = False
  cache_mels: bool = False


@dataclass
class ModelHParams():
  symbols_embedding_dim: int = 512

  use_speaker_embedding: bool = True
  speakers_embedding_dim: Optional[int] = 128  # 16

  # TODO rename to: train_stress_separately: bool = True
  use_stress_embedding: bool = True
  # use_stress_one_hot: bool = True
  symbols_are_ipa: bool = True

  # None for 1-hot encoding
  stress_embedding_dim: Optional[int] = None

  # Encoder parameters
  encoder_kernel_size: int = 5
  encoder_n_convolutions: int = 3
  # encoder_embedding_dim: int = 512 is equal to symbols_embedding_dim + stress_embedding_dim

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
  learning_rate: float = 1e-03
  grad_clip_thresh: float = 1.0
  batch_size: int = 64

  # set model's padded outputs to padded values
  mask_padding: bool = True

  # coefficients used for computing running averages of gradient and its square
  beta1: float = 0.9
  beta2: float = 0.999

  # term added to the denominator to improve numerical stability
  eps: float = 1e-08

  # L2 penalty (L2 regularization)
  weight_decay: float = 1e-06

  # whether to use the AMSGrad variant of adam from the paper "On the Convergence of Adam and Beyond"
  amsgrad: bool = False

  use_exponential_lr_decay: bool = False

  # One-based epoch after which the LR decaying is started, i.e., int in range [1, epochs)
  lr_decay_start_after_epoch: Optional[int] = 250
  lr_decay_gamma: Optional[float] = 0.97

  # is in range (0, learning_rate]
  lr_decay_min: Optional[float] = 1e-05


@dataclass
class HParams(ExperimentHParams, DataHParams, TSTFTHParams, ModelHParams, OptimizerHParams):
  pass
