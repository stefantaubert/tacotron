from pathlib import Path

import torch

DEFAULT_SEED = None
DEFAULT_REPETITIONS = 1
DEFAULT_MAX_DECODER_STEPS = 3000
DEFAULT_SAVE_MEL_INFO_COPY_PATH = Path("/tmp/mel_out.json")
# from paper
DEFAULT_MCD_NO_OF_COEFFS_PER_FRAME = 16

if torch.cuda.is_available():
  __DEFAULT_DEVICE = "cuda:0"
else:
  __DEFAULT_DEVICE = "cpu"

DEFAULT_DEVICE = __DEFAULT_DEVICE
