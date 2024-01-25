from pathlib import Path

import numpy as np
import wget

from tacotron.synthesizer import Synthesizer
from tacotron.utils import get_default_device
from tacotron_cli.io import load_checkpoint, try_load_checkpoint


def test_component():
  TACO_CKP = "https://zenodo.org/records/10107104/files/101000.pt"
  target_path = Path("/tmp/tacotron-test.pt")
  if not target_path.is_file():
    wget.download(TACO_CKP, str(target_path.absolute()))
  checkpoint = load_checkpoint(target_path, device=get_default_device())

  s = Synthesizer(checkpoint)
  text = 'ð|ˈɪ|s|SIL0|ˈɪ|z|SIL0|ə|SIL0|tː|ˈɛ|s|t|SIL0|ˈæ|b|?|SIL2|ə|n˘|d|SIL0|ˈaɪ˘|m|SIL0|ð|ˈɛr˘|SIL0|θ|ˈʌr|d˘|ˌi|-|wː|ˈʌː|nː|.|SIL2'

  result = s.infer(text.split("|"), "Linda Johnson", seed=0)

  assert result.sampling_rate == 22050
  assert result.reached_max_decoder_steps is False
  assert result.unmappable_durations is None
  assert result.unmappable_stresses is None
  assert result.unmappable_symbols is None
  assert result.unmappable_tones is None
  np.testing.assert_array_almost_equal(
    result.mel_outputs_postnet[:5, :5],
    np.array([
        [-6.9595537, -6.7366004, -6.482799, -6.498109, -6.52342],
        [-6.557069, -6.060888, -5.713961, -5.7108502, -5.830424],
        [-5.943193, -5.374925, -4.958975, -4.860589, -5.1093984],
        [-5.288998, -4.7378426, -4.6273413, -4.6779313, -5.0058713],
        [-4.703808, -3.656827, -3.7665925, -4.4009595, -4.8536625]
      ],
      dtype=float
    )
  )
  assert result.mel_outputs_postnet.shape == (80, 214)
  assert result.mel_outputs is None
  assert result.gate_outputs is None
  assert result.alignments is None
  assert result.duration_s == 2.4729251700680273
  assert result.inference_duration_s > 0
