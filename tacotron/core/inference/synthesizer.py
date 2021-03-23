import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from audio_utils import concatenate_audios, normalize_wav
from audio_utils.mel import mel_to_numpy
from tacotron.core.synthesizer import Synthesizer as TacoSynthesizer
from tacotron.core.training import CheckpointTacotron
from tacotron.utils import pass_lines
from tts_preparation import InferSentence, InferSentenceList


@dataclass
class InferenceResult():
  sentence: InferSentence
  wav: np.ndarray
  sampling_rate: int
  reached_max_decoder_steps: bool
  taco_inference_duration_s: float
  wg_inference_duration_s: float
  mel_outputs: np.ndarray
  mel_outputs_postnet: np.ndarray
  gate_outputs: np.ndarray
  alignments: np.ndarray


class Synthesizer():
  def __init__(self, tacotron_checkpoint: CheckpointTacotron, waveglow_checkpoint: CheckpointWaveglow, custom_taco_hparams: Optional[Dict[str, str]], custom_wg_hparams: Optional[Dict[str, str]], logger: logging.Logger):
    super().__init__()
    self._logger = logger

    self._taco_synt = TacoSynthesizer(
      checkpoint=tacotron_checkpoint,
      custom_hparams=custom_taco_hparams,
      logger=logger
    )

    self._wg_synt = WGSynthesizer(
      checkpoint=waveglow_checkpoint,
      custom_hparams=custom_wg_hparams,
      logger=logger
    )

    assert self._wg_synt.hparams.sampling_rate == self._taco_synt.hparams.sampling_rate

  def get_sampling_rate(self) -> int:
    return self._wg_synt.hparams.sampling_rate

  def _concatenate_wavs(self, result: List[InferenceResult], sentence_pause_s: float):
    wavs = [res.wav for res in result]
    if len(wavs) > 1:
      self._logger.info("Concatening audios...")
    output = concatenate_audios(wavs, sentence_pause_s, self._taco_synt.hparams.sampling_rate)

    return output

  def _infer_sentence(self, sentence: InferSentence, speaker: str, sigma: float, denoiser_strength: float):
    outputs, stats = self._taco_synt.infer(
      symbols=sentence.symbols,
      accents=sentence.accents,
      speaker=speaker,
    )

    mel_outputs, mel_outputs_postnet, gate_outputs, alignments = outputs
    reached_max_decoder_steps, taco_inference_duration_s = stats

    synthesized_sentence, wg_inference_duration_s = self._wg_synt.infer(
      mel=mel_outputs_postnet,
      sigma=sigma,
      denoiser_strength=denoiser_strength,
    )

    infer_res = InferenceResult(
      sentence=sentence,
      wav=synthesized_sentence,
      sampling_rate=self._wg_synt.hparams.sampling_rate,
      reached_max_decoder_steps=reached_max_decoder_steps,
      taco_inference_duration_s=taco_inference_duration_s,
      wg_inference_duration_s=wg_inference_duration_s,
      mel_outputs=mel_to_numpy(mel_outputs),
      mel_outputs_postnet=mel_to_numpy(mel_outputs_postnet),
      gate_outputs=mel_to_numpy(gate_outputs),
      alignments=mel_to_numpy(alignments),
    )

    return infer_res

  def infer(self, sentences: InferSentenceList, speaker: str, sigma: float, denoiser_strength: float, sentence_pause_s: float) -> Tuple[np.ndarray, List[InferenceResult]]:
    self._logger.debug(f"Selected speaker: {speaker}")

    result: List[InferenceResult] = []

    accent_id_dict = self._taco_synt.accents

    all_in_one = False

    if all_in_one:
      sentence = sentences.to_sentence(
        space_symbol=" ",
        space_accent="north_america",
      )
      self._logger.info(f"\n{sentence.get_formatted(accent_id_dict)}")
      infer_res = self._infer_sentence(sentence, speaker, sigma, denoiser_strength)
      result.append(infer_res)
    else:
      # Speed is: 1min inference for 3min wav result
      for sentence in sentences.items(True):
        pass_lines(self._logger.info, sentence.get_formatted(accent_id_dict))
        infer_res = self._infer_sentence(sentence, speaker, sigma, denoiser_strength)
        result.append(infer_res)

    output = self._concatenate_wavs(result, sentence_pause_s)
    output = normalize_wav(output)

    for infer_res in result:
      infer_res.wav = normalize_wav(infer_res.wav)

    return output, result
