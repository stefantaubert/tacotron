import datetime
import random
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Dict, List, Optional, Set

#import jiwer
#import jiwer.transforms as tr
import numpy as np
import pandas as pd
import torch
from mel_cepstral_distance import get_metrics_mels
from pandas import DataFrame
from scipy.io.wavfile import read
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from tacotron.audio_utils import align_mels_with_dtw, get_msd, plot_melspec_np
from tacotron.checkpoint_handling import CheckpointDict, get_iteration
from tacotron.image_utils import calculate_structual_similarity_np
from tacotron.synthesizer import Synthesizer
from tacotron.taco_stft import TacotronSTFT
from tacotron.typing import Entries, Entry
from tacotron.utils import cosine_dist_mels, make_same_dim, plot_alignment_np_new


@dataclass
class ValidationEntry():
  timepoint: datetime.datetime = None
  entry: Entry = None
  repetition: int = None
  repetitions: int = None
  seed: int = None
  #train_name: str = None
  iteration: int = None
  sampling_rate: int = None
  # grad_norm: float = None
  # loss: float = None
  inference_duration_s: float = None
  reached_max_decoder_steps: bool = None
  target_frames: int = None
  predicted_frames: int = None
  padded_mse: float = None
  padded_cosine_similarity: Optional[float] = None
  padded_structural_similarity: Optional[float] = None
  aligned_mse: float = None
  aligned_cosine_similarity: Optional[float] = None
  aligned_structural_similarity: Optional[float] = None
  mfcc_no_coeffs: int = None
  mfcc_dtw_mcd: float = None
  mfcc_dtw_penalty: float = None
  mfcc_dtw_frames: int = None
  msd: float = None
  # wer: float = None
  # mer: float = None
  # wil: float = None
  # wip: float = None
  # train_one_gram_rarity: float = None
  # train_two_gram_rarity: float = None
  # train_three_gram_rarity: float = None
  was_fast: bool = None


class ValidationEntries(List[ValidationEntry]):
  pass


def get_df(entries: ValidationEntries) -> DataFrame:
  if len(entries) == 0:
    return DataFrame()

  data = []
  for entry in entries:
    tmp = {}
    # tmp["Id"] = entry.entry.entry_id
    tmp["Basename"] = entry.entry.basename
    tmp["Timepoint"] = f"{entry.timepoint:%Y/%m/%d %H:%M:%S}"
    tmp["Iteration"] = entry.iteration
    tmp["Seed"] = entry.seed
    tmp["Repetition"] = entry.repetition
    tmp["Repetitions"] = entry.repetitions
    tmp["Language"] = entry.entry.symbols_language
    tmp["Symbols"] = ''.join(entry.entry.symbols)
    tmp["Stem"] = entry.entry.stem
    # tmp["Symbols format"] = repr(entry.entry.symbols_format)
    tmp["Speaker"] = entry.entry.speaker_name
    tmp["Speaker Gender"] = entry.entry.speaker_gender
    # tmp["Speaker Id"] = entry.entry.speaker_id
    tmp["Inference duration (s)"] = entry.inference_duration_s
    tmp["Reached max. steps"] = entry.reached_max_decoder_steps
    tmp["Sampling rate (Hz)"] = entry.sampling_rate
    if not entry.was_fast:
      tmp["# MFCC Coefficients"] = entry.mfcc_no_coeffs
      tmp["MFCC DTW MCD"] = entry.mfcc_dtw_mcd
      tmp["MFCC DTW PEN"] = entry.mfcc_dtw_penalty
      tmp["# MFCC DTW frames"] = entry.mfcc_dtw_frames
      tmp["# Target frames"] = entry.target_frames
    tmp["# Predicted frames"] = entry.predicted_frames
    if not entry.was_fast:
      tmp["# Difference frames"] = entry.predicted_frames - \
          entry.target_frames
      tmp["Frames deviation (%)"] = (
          entry.predicted_frames / entry.target_frames) - 1
      tmp["MSE (Padded)"] = entry.padded_mse
      tmp["Cosine Similarity (Padded)"] = entry.padded_cosine_similarity
      tmp["Structual Similarity (Padded)"] = entry.padded_structural_similarity
      tmp["MSE (Aligned)"] = entry.aligned_mse
      tmp["Cosine Similarity (Aligned)"] = entry.aligned_cosine_similarity
      tmp["Structual Similarity (Aligned)"] = entry.aligned_structural_similarity
      tmp["MSD"] = entry.msd
      # tmp["WER"] = entry.wer
      # tmp["MER"] = entry.mer
      # tmp["WIL"] = entry.wil
      # tmp["WIP"] = entry.wip
      # tmp["1-gram rarity (train set)"] = entry.train_one_gram_rarity
      # tmp["2-gram rarity (train set)"] = entry.train_two_gram_rarity
      # tmp["3-gram rarity (train set)"] = entry.train_three_gram_rarity
      # tmp["Combined rarity (train set)"] = entry.train_one_gram_rarity + \
      #     entry.train_two_gram_rarity + entry.train_three_gram_rarity
      # tmp["1-gram rarity (total set)"] = entry.utterance.one_gram_rarity
      # tmp["2-gram rarity (total set)"] = entry.utterance.two_gram_rarity
      # tmp["3-gram rarity (total set)"] = entry.utterance.three_gram_rarity
      # tmp["Combined rarity (total set)"] = entry.utterance.one_gram_rarity + \
      #     entry.utterance.two_gram_rarity + entry.utterance.three_gram_rarity
    tmp["# Symbols"] = len(entry.entry.symbols)
    tmp["Unique symbols"] = ' '.join(sorted(set(entry.entry.symbols)))
    tmp["# Unique symbols"] = len(set(entry.entry.symbols))
    #tmp["Train name"] = entry.train_name
    # tmp["Ds-Id"] = entry.entry.ds_entry_id
    tmp["Wav path"] = str(entry.entry.wav_absolute_path)
    # tmp["Wav path original"] = str(entry.entry.wav_original_absolute_path)
    data.append(tmp)

  df = DataFrame(
      data=[x.values() for x in data],
      columns=data[0].keys(),
  )

  return df


@dataclass
class ValidationEntryOutput():
  repetition: int = None
  wav_orig: np.ndarray = None
  mel_orig: np.ndarray = None
  mel_orig_aligned: np.ndarray = None
  orig_sr: int = None
  mel_orig_img: np.ndarray = None
  mel_orig_aligned_img: np.ndarray = None
  mel_postnet: np.ndarray = None
  mel_postnet_aligned: np.ndarray = None
  mel_postnet_sr: int = None
  mel_postnet_img: np.ndarray = None
  mel_postnet_aligned_img: np.ndarray = None
  mel_postnet_diff_img: np.ndarray = None
  mel_postnet_aligned_diff_img: np.ndarray = None
  mel_img: np.ndarray = None
  alignments_img: np.ndarray = None
  alignments_aligned_img: np.ndarray = None
  was_fast: bool = None
  # gate_out_img: np.ndarray


# def get_ngram_rarity(data: Entries, corpus: Entries, ngram: int) -> OrderedDictType[int, float]:
#   data_symbols_dict = OrderedDict({x.entry_id: x.symbols for x in data})
#   corpus_symbols_dict = OrderedDict({x.entry_id: x.symbols for x in corpus})

#   rarity = get_rarity_ngrams(
#     data=data_symbols_dict,
#     corpus=corpus_symbols_dict,
#     n_gram=ngram,
#     ignore_symbols=None,
#   )

#   return rarity


# def get_best_seeds(select_best_from: pd.DataFrame, entry_ids: OrderedSet[int], iteration: int, logger: Logger) -> List[int]:
#     df = select_best_from.loc[select_best_from['iteration'] == iteration]
#     result = []
#     for entry_id in entry_ids:
#         logger.info(f"Entry {entry_id}")
#         entry_df = df.loc[df['entry_id'] == entry_id]
#         # best_row = entry_df.iloc[[entry_df["mfcc_dtw_mcd"].argmin()]]
#         # worst_row = entry_df.iloc[[entry_df["mfcc_dtw_mcd"].argmax()]]

#         metric_values = []

#         for _, row in entry_df.iterrows():
#             best_value = row['mfcc_dtw_mcd'] + row['mfcc_dtw_penalty']
#             metric_values.append(best_value)

#         logger.info(f"Mean MCD: {entry_df['mfcc_dtw_mcd'].mean()}")
#         logger.info(f"Mean PEN: {entry_df['mfcc_dtw_penalty'].mean()}")
#         logger.info(f"Mean metric: {np.mean(metric_values)}")

#         best_idx = np.argmin(metric_values)
#         best_row = entry_df.iloc[best_idx]
#         test_x = best_row['mfcc_dtw_mcd'] + best_row['mfcc_dtw_penalty']
#         assert test_x == metric_values[best_idx]

#         worst_idx = np.argmax(metric_values)
#         worst_row = entry_df.iloc[worst_idx]
#         test_x = worst_row['mfcc_dtw_mcd'] + worst_row['mfcc_dtw_penalty']
#         assert test_x == metric_values[worst_idx]

#         seed = int(best_row["seed"])
#         logger.info(
#             f"The best seed was {seed} with an MCD of {float(best_row['mfcc_dtw_mcd']):.2f} and a PEN of {float(best_row['mfcc_dtw_penalty']):.5f} -> {metric_values[best_idx]:.5f}")
#         logger.info(
#             f"The worst seed was {int(worst_row['seed'])} with an MCD of {float(worst_row['mfcc_dtw_mcd']):.2f} and a PEN of {float(worst_row['mfcc_dtw_penalty']):.5f} -> {metric_values[worst_idx]:.5f}")
#         # print(f"The worst mcd was {}")
#         logger.info("------")
#         result.append(seed)
#     return result


# def wav_to_text(wav: np.ndarray) -> str:
#   return ""


def validate(checkpoint: CheckpointDict, data: Entries, custom_hparams: Optional[Dict[str, str]], entry_names: Set[str], speaker_name: Optional[str], full_run: bool, save_callback: Callable[[Entry, ValidationEntryOutput], None], max_decoder_steps: int, fast: bool, mcd_no_of_coeffs_per_frame: int, repetitions: int, seed: Optional[int], select_best_from: Optional[pd.DataFrame], device: torch.device, logger: Logger) -> ValidationEntries:
  seeds: List[int]
  validation_data: Entries
  iteration = get_iteration(checkpoint)

  if seed is None:
    seed = random.randint(1, 9999)
    logger.info(f"As no seed was given, using random seed: {seed}.")

  if full_run:
    validation_data = data
    seeds = [seed for _ in data]
  elif select_best_from is not None:
    raise NotImplementedError()
    # assert entry_names is not None
    # logger.info("Finding best seeds...")
    # validation_data = [x for x in data if x.audio_file_abs.stem in entry_names]
    # if len(validation_data) != len(entry_names):
    #   logger.error("Not all entry name's were found!")
    #   assert False
    # entry_ids_order_from_valdata = OrderedSet([x.entry_id for x in validation_data.items()])
    # seeds = get_best_seeds(select_best_from, entry_ids_order_from_valdata,
    #                        iteration, logger)

  #   entry_ids = [entry_id for entry_id, _ in entry_ids_w_seed]
  #   have_no_double_entries = len(set(entry_ids)) == len(entry_ids)
  #   assert have_no_double_entries
  #   validation_data = PreparedDataList([x for x in data.items() if x.entry_id in entry_ids])
  #   seeds = [s for _, s in entry_ids_w_seed]
  #   if len(validation_data) != len(entry_ids):
  #     logger.error("Not all entry_id's were found!")
  #     assert False
  elif len(entry_names) > 0:
    validation_data = [
        x for x in data if x.audio_file_abs.stem in entry_names]
    if len(validation_data) != len(entry_names):
      logger.error("Not all entry file's were found!")
      assert False
    seeds = [seed for _ in validation_data]
  elif speaker_name is not None:
    relevant_entries = [x for x in data if x.speaker_name == speaker_name]
    assert len(relevant_entries) > 0
    random.seed(seed)
    entry = random.choice(relevant_entries)
    validation_data = [entry]
    seeds = [seed]
  else:
    random.seed(seed)
    entry = random.choice(data)
    validation_data = [entry]
    seeds = [seed]

  assert len(seeds) == len(validation_data)

  if len(validation_data) == 0:
    logger.info("Nothing to synthesize!")
    return validation_data

  synth = Synthesizer(
      checkpoint=checkpoint,
      custom_hparams=custom_hparams,
      device=device,
      logger=logger,
  )

  taco_stft = TacotronSTFT(synth.hparams, device=device, logger=logger)
  validation_entries = ValidationEntries()

  if not fast:
    # jiwer_ground_truth_transform = tr.Compose([
    #     tr.ToLowerCase(),
    #     tr.ReduceToListOfListOfWords(),
    # ])

    # jiwer_inferred_asr_transform = tr.Compose([
    #     tr.ToLowerCase(),
    #     tr.ReduceToListOfListOfWords(),
    # ])

    # train_onegram_rarities = get_ngram_rarity(validation_data, trainset, 1)
    # train_twogram_rarities = get_ngram_rarity(validation_data, trainset, 2)
    # train_threegram_rarities = get_ngram_rarity(validation_data, trainset, 3)
    pass
  #asr_pipeline = asr.load('deepspeech2', lang='en')
  # asr_pipeline.model.summary()

  for repetition in range(repetitions):
    # rep_seed = seed + repetition
    rep_human_readable = repetition + 1
    logger.info(f"Starting repetition: {rep_human_readable}/{repetitions}")
    entry: Entry
    for entry, entry_seed in zip(tqdm(validation_data), seeds):
      rep_seed = entry_seed + repetition
      logger.info(
          f"Current --> entry name: {entry.basename}; seed: {rep_seed}; iteration: {iteration}; rep: {rep_human_readable}/{repetitions}")

      timepoint = datetime.datetime.now()
      inference_result = synth.infer(
          symbols=entry.symbols,
          speaker=entry.speaker_name,
          max_decoder_steps=max_decoder_steps,
          seed=rep_seed,
      )

      val_entry = ValidationEntry()

      val_entry.entry = entry
      val_entry.repetition = rep_human_readable
      val_entry.repetitions = repetitions
      val_entry.seed = rep_seed
      val_entry.iteration = iteration
      val_entry.timepoint = timepoint
      #val_entry.train_name = train_name
      val_entry.sampling_rate = synth.get_sampling_rate()
      val_entry.reached_max_decoder_steps = inference_result.reached_max_decoder_steps
      val_entry.inference_duration_s = inference_result.inference_duration_s
      val_entry.predicted_frames = inference_result.mel_outputs_postnet.shape[1]
      val_entry.was_fast = fast

      if not fast:

        mel_orig: np.ndarray = taco_stft.get_mel_tensor_from_file(
            entry.wav_absolute_path).cpu().numpy()

        target_frames = mel_orig.shape[1]

        dtw_mcd, dtw_penalty, dtw_frames = get_metrics_mels(mel_orig, inference_result.mel_outputs_postnet,
                                                            n_mfcc=mcd_no_of_coeffs_per_frame,
                                                            take_log=False,
                                                            use_dtw=True,
                                                            )

        padded_mel_orig, padded_mel_postnet = make_same_dim(
            mel_orig, inference_result.mel_outputs_postnet)

        aligned_mel_orig, aligned_mel_postnet, mel_dtw_dist, _, path_mel_postnet = align_mels_with_dtw(
            mel_orig, inference_result.mel_outputs_postnet)

        mel_aligned_length = len(aligned_mel_postnet)
        msd = get_msd(mel_dtw_dist, mel_aligned_length)

        padded_cosine_similarity = cosine_dist_mels(
            padded_mel_orig, padded_mel_postnet)
        aligned_cosine_similarity = cosine_dist_mels(
            aligned_mel_orig, aligned_mel_postnet)

        padded_mse = mean_squared_error(
            padded_mel_orig, padded_mel_postnet)
        aligned_mse = mean_squared_error(
            aligned_mel_orig, aligned_mel_postnet)

        padded_structural_similarity = None
        aligned_structural_similarity = None

        #ground_truth = ''.join(entry.symbols)
        #hypothesis = asr_pipeline.predict([inference_result.mel_outputs_postnet])[0]

        # measures = jiwer.compute_measures(
        #     truth=ground_truth,
        #     truth_transform=jiwer_ground_truth_transform,
        #     hypothesis=hypothesis,
        #     hypothesis_transform=jiwer_inferred_asr_transform,
        # )

        # wer = measures['wer']
        # mer = measures['mer']
        # wil = measures['wil']
        # wip = measures['wip']

        padded_mel_orig_img_raw_1, padded_mel_orig_img = plot_melspec_np(
            padded_mel_orig, title="padded_mel_orig")
        padded_mel_outputs_postnet_img_raw_1, padded_mel_outputs_postnet_img = plot_melspec_np(
            padded_mel_postnet, title="padded_mel_postnet")

        # imageio.imsave("/tmp/padded_mel_orig_img_raw_1.png", padded_mel_orig_img_raw_1)
        # imageio.imsave("/tmp/padded_mel_outputs_postnet_img_raw_1.png", padded_mel_outputs_postnet_img_raw_1)

        padded_structural_similarity, padded_mel_postnet_diff_img_raw = calculate_structual_similarity_np(
            img_a=padded_mel_orig_img_raw_1,
            img_b=padded_mel_outputs_postnet_img_raw_1,
        )

        # imageio.imsave("/tmp/padded_mel_diff_img_raw_1.png", padded_mel_postnet_diff_img_raw)

        aligned_mel_orig_img_raw, aligned_mel_orig_img = plot_melspec_np(
            aligned_mel_orig, title="aligned_mel_orig")
        aligned_mel_postnet_img_raw, aligned_mel_postnet_img = plot_melspec_np(
            aligned_mel_postnet, title="aligned_mel_postnet")

        # imageio.imsave("/tmp/aligned_mel_orig_img_raw.png", aligned_mel_orig_img_raw)
        # imageio.imsave("/tmp/aligned_mel_postnet_img_raw.png", aligned_mel_postnet_img_raw)

        aligned_structural_similarity, aligned_mel_diff_img_raw = calculate_structual_similarity_np(
            img_a=aligned_mel_orig_img_raw,
            img_b=aligned_mel_postnet_img_raw,
        )

        # imageio.imsave("/tmp/aligned_mel_diff_img_raw.png", aligned_mel_diff_img_raw)

        val_entry.target_frames = target_frames
        val_entry.padded_cosine_similarity = padded_cosine_similarity
        val_entry.mfcc_no_coeffs = mcd_no_of_coeffs_per_frame
        val_entry.mfcc_dtw_mcd = dtw_mcd
        val_entry.mfcc_dtw_penalty = dtw_penalty
        val_entry.mfcc_dtw_frames = dtw_frames
        val_entry.padded_structural_similarity = padded_structural_similarity
        val_entry.padded_mse = padded_mse
        val_entry.msd = msd
        # val_entry.wer = wer
        # val_entry.mer = mer
        # val_entry.wil = wil
        # val_entry.wip = wip
        val_entry.aligned_cosine_similarity = aligned_cosine_similarity
        val_entry.aligned_mse = aligned_mse
        val_entry.aligned_structural_similarity = aligned_structural_similarity
        # val_entry.train_one_gram_rarity = train_onegram_rarities[entry.entry_id]
        # val_entry.train_two_gram_rarity = train_twogram_rarities[entry.entry_id]
        # val_entry.train_three_gram_rarity = train_threegram_rarities[entry.entry_id]

      validation_entries.append(val_entry)

      validation_entry_output = ValidationEntryOutput()

      validation_entry_output.was_fast = fast
      validation_entry_output.repetition = rep_human_readable
      validation_entry_output.mel_postnet = inference_result.mel_outputs_postnet
      validation_entry_output.mel_postnet_sr = inference_result.sampling_rate

      if not fast:
        orig_sr, orig_wav = read(entry.wav_absolute_path)

        _, padded_mel_postnet_diff_img = calculate_structual_similarity_np(
            img_a=padded_mel_orig_img,
            img_b=padded_mel_outputs_postnet_img,
        )

        _, aligned_mel_postnet_diff_img = calculate_structual_similarity_np(
            img_a=aligned_mel_orig_img,
            img_b=aligned_mel_postnet_img,
        )

        _, mel_orig_img = plot_melspec_np(mel_orig, title="mel_orig")
        # alignments_img = plot_alignment_np(inference_result.alignments)
        _, post_mel_img = plot_melspec_np(
            inference_result.mel_outputs_postnet, title="mel_postnet")
        _, mel_img = plot_melspec_np(
            inference_result.mel_outputs, title="mel")
        _, alignments_img = plot_alignment_np_new(
            inference_result.alignments, title="alignments")

        aligned_alignments = inference_result.alignments[path_mel_postnet]
        _, aligned_alignments_img = plot_alignment_np_new(
            aligned_alignments, title="aligned_alignments")
        # imageio.imsave("/tmp/alignments.png", alignments_img)

        validation_entry_output.mel_orig = mel_orig
        validation_entry_output.wav_orig = orig_wav
        validation_entry_output.orig_sr = orig_sr
        validation_entry_output.mel_orig_img = mel_orig_img
        validation_entry_output.mel_postnet_img = post_mel_img
        validation_entry_output.mel_postnet_diff_img = padded_mel_postnet_diff_img
        validation_entry_output.alignments_img = alignments_img
        validation_entry_output.mel_img = mel_img
        validation_entry_output.mel_postnet_aligned_diff_img = aligned_mel_postnet_diff_img
        validation_entry_output.mel_orig_aligned = aligned_mel_orig
        validation_entry_output.mel_orig_aligned_img = aligned_mel_orig_img
        validation_entry_output.mel_postnet_aligned = aligned_mel_postnet
        validation_entry_output.mel_postnet_aligned_img = aligned_mel_postnet_img
        validation_entry_output.alignments_aligned_img = aligned_alignments_img

      save_callback(entry, validation_entry_output)

      if not fast:
        logger.info(f"MFCC MCD DTW: {val_entry.mfcc_dtw_mcd}")

  return validation_entries
