from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path

import imageio
import numpy as np
from ordered_set import OrderedSet
from scipy.io.wavfile import write
from tqdm import tqdm

from tacotron.globals import DEFAULT_CSV_SEPERATOR
from tacotron.image_utils import stack_images_vertically
from tacotron.parser import load_dataset
from tacotron.typing import Entry
from tacotron.utils import (get_checkpoint, get_last_checkpoint, prepare_logger,
                            set_torch_thread_to_max, split_hparams_string)
from tacotron.validation import ValidationEntries, ValidationEntryOutput, get_df, validate
from tacotron_cli.argparse_helper import (ConvertToOrderedSetAction, ConvertToSetAction,
                                          get_optional, parse_existing_directory, parse_non_empty,
                                          parse_non_empty_or_whitespace, parse_non_negative_integer,
                                          parse_path, parse_positive_integer)
from tacotron_cli.defaults import DEFAULT_MCD_NO_OF_COEFFS_PER_FRAME, DEFAULT_REPETITIONS
from tacotron_cli.helper import (add_device_argument, add_hparams_argument,
                                 add_max_decoder_steps_argument)
from tacotron_cli.io import try_load_checkpoint

# def get_repr_entries(entry_names: Optional[Set[str]]) -> str:
#     if entry_names is None:
#         return "none"
#     if len(entry_names) == 0:
#         return "empty"
#     return ",".join(list(sorted(map(str, entry_names))))


# def get_repr_speaker(speaker: Optional[Speaker]) -> Speaker:
#     if speaker is None:
#         return "none"
#     return speaker


# def get_run_name(ds: str, iterations: Set[int], full_run: bool, entry_names: Optional[Set[str]], speaker: Optional[str]) -> str:
#     if len(iterations) > 3:
#         its = ",".join(str(x) for x in sorted(iterations)[:3]) + ",..."
#     else:
#         its = ",".join(str(x) for x in sorted(iterations))

#     subdir_name = f"{datetime.datetime.now():%d.%m.%Y__%H-%M-%S}__ds={ds}__entries={get_repr_entries(entry_names)}__speaker={get_repr_speaker(speaker)}__its={its}__full={full_run}"
#     return subdir_name


# def get_val_dir(train_dir: Path, run_name: str) -> Path:
#     return _get_validation_root_dir(train_dir) / run_name


# def get_val_dir_new(train_dir: Path):
#   subdir_name = f"{datetime.datetime.now():%Y-%m-%d__%H-%M-%S}"
#   return get_subdir(_get_validation_root_dir(train_dir), subdir_name, create=True)


def get_val_entry_dir(val_dir: Path, result_name: str) -> Path:
  return val_dir / result_name


def save_stats(val_dir: Path, validation_entries: ValidationEntries) -> None:
  path = val_dir / "total.csv"
  df = get_df(validation_entries)
  df.to_csv(path, sep=DEFAULT_CSV_SEPERATOR, header=True)


# def save_mel_postnet_npy_paths(val_dir: Path, mel_postnet_npy_paths: List[Dict[str, Any]]) -> Path:
#     info_json = get_mel_out_dict(
#         root_dir=val_dir,
#         mel_info_dict=mel_postnet_npy_paths,
#     )

#     path = val_dir / "mel_postnet_npy.json"
#     save_json(path, info_json)
#     # text = '\n'.join(mel_postnet_npy_paths)
#     # save_txt(path, text)
#     return path


def get_result_name(entry: Entry, iteration: int, repetition: int) -> None:
  return f"it={iteration}_name={entry.basename}_rep={repetition}"


def save_results(entry: Entry, output: ValidationEntryOutput, val_dir: Path, iteration: int) -> None:
  result_name = get_result_name(entry, iteration, output.repetition)
  dest_dir = get_val_entry_dir(val_dir, result_name)
  dest_dir.mkdir(parents=True, exist_ok=True)

  mel_postnet_npy_path = dest_dir / "inferred.mel.npy"
  np.save(mel_postnet_npy_path, output.mel_postnet)

  if not output.was_fast:
    write(dest_dir / "original.wav", output.orig_sr, output.wav_orig)
    imageio.imsave(dest_dir / "original.png", output.mel_orig_img)
    imageio.imsave(dest_dir / "original_aligned.png",
                   output.mel_orig_aligned_img)
    imageio.imsave(dest_dir / "inferred.png", output.mel_postnet_img)
    imageio.imsave(dest_dir / "inferred_aligned.png",
                   output.mel_postnet_aligned_img)
    imageio.imsave(dest_dir / "mel.png", output.mel_img)
    imageio.imsave(dest_dir / "alignments.png", output.alignments_img)
    imageio.imsave(dest_dir / "alignments_aligned.png",
                   output.alignments_aligned_img)
    imageio.imsave(dest_dir / "diff.png", output.mel_postnet_diff_img)
    imageio.imsave(dest_dir / "diff_aligned.png",
                   output.mel_postnet_aligned_diff_img)
    np.save(dest_dir / "original.mel.npy", output.mel_orig)
    np.save(dest_dir / "original_aligned.mel.npy", output.mel_orig_aligned)
    np.save(dest_dir / "inferred_aligned.mel.npy",
            output.mel_postnet_aligned)

    stack_images_vertically(
        list_im=[
            dest_dir / "original.png",
            dest_dir / "inferred.png",
            dest_dir / "diff.png",
            dest_dir / "alignments.png",
            dest_dir / "mel.png",
        ],
        out_path=dest_dir / "comparison.png"
    )

    stack_images_vertically(
        list_im=[
            dest_dir / "original.png",
            dest_dir / "inferred.png",
            dest_dir / "original_aligned.png",
            dest_dir / "inferred_aligned.png",
            dest_dir / "diff_aligned.png",
            dest_dir / "alignments_aligned.png",
        ],
        out_path=dest_dir / "comparison_aligned.png"
    )

  # mel_info = get_mel_info_dict(
  #     identifier=result_name,
  #     path=mel_postnet_npy_path,
  #     sr=output.mel_postnet_sr,
  # )

  # mel_postnet_npy_paths.append(mel_info)


def init_validation_parser(parser: ArgumentParser) -> None:
  parser.description = "Validate checkpoint(s) using the validation set or any other dataset."
  parser.add_argument('checkpoints_dir',
                      metavar="CHECKPOINTS-FOLDER", type=parse_existing_directory, help="path to folder containing the checkpoints that should be validated")
  parser.add_argument('output_dir',
                      metavar="OUTPUT-FOLDER", type=parse_path, help="path to folder in which the resulting files should be written")
  parser.add_argument('dataset_dir', metavar="DATA-FOLDER",
                      type=parse_existing_directory, help="path to validation set folder or any other dataset")
  parser.add_argument("tier", metavar="TIER", type=parse_non_empty_or_whitespace,
                      help="name of grids tier that contains the symbol intervals")
  add_device_argument(parser)
  add_hparams_argument(parser)
  parser.add_argument('--full-run', action='store_true', help="validate all files in DATA-FOLDER")
  parser.add_argument('--files', type=parse_non_empty, nargs="*", metavar="UTTERANCE",
                      help="names of utterances in DATA-FOLDER that should be validated; if left unset a random utterance will be chosen", default=OrderedSet(), action=ConvertToSetAction)
  parser.add_argument('--speaker', type=get_optional(parse_non_empty),
                      help="chose random utterance only from this speaker (only relevant if no UTTERANCE is defined)", default=None)
  parser.add_argument('--custom-checkpoints',
                      type=parse_positive_integer, nargs="*", default=OrderedSet(), action=ConvertToOrderedSetAction, help="validate checkpoints with these iterations; is left unset the last iteration is chosen")
  add_max_decoder_steps_argument(parser)
  parser.add_argument('--include-stats', action='store_true',
                      help="include logging of statistics (increases synthesis duration)")
  #parser.add_argument('--select-best-from', type=parse_existing_file)
  parser.add_argument('--mcd-no-of-coeffs-per-frame', metavar="NUMBER-OF-COEFFICIENTS", type=parse_positive_integer,
                      default=DEFAULT_MCD_NO_OF_COEFFS_PER_FRAME, help="number of coefficients used for calculating MCD")
  parser.add_argument('--custom-seed', metavar="CUSTOM-SEED", type=get_optional(parse_non_negative_integer),
                      default=None, help="custom seed used for synthesis; if left unset a random seed will be chosen")
  parser.add_argument('--repetitions', type=parse_positive_integer, metavar="REPETITIONS", default=DEFAULT_REPETITIONS,
                      help="how often the synthesis should be done; the seed will be increased by one in each repetition")

  return validate_ns


def validate_ns(ns: Namespace) -> None:
  assert ns.repetitions > 0

  set_torch_thread_to_max()
  data = load_dataset(ns.dataset_dir, ns.tier)

  iterations: OrderedSet[int]
  if len(ns.custom_checkpoints) == 0:
    _, last_it = get_last_checkpoint(ns.checkpoints_dir)
    iterations = OrderedSet((last_it,))
  else:
    # if len(custom_checkpoints) == 0:
    #     iterations = set(get_all_checkpoint_iterations(checkpoint_dir))
    # else:
    iterations = ns.custom_checkpoints
  ns.output_dir.mkdir(parents=True, exist_ok=True)

  val_log_path = ns.output_dir / "log.txt"
  logger = prepare_logger(val_log_path, reset=True)
  logger.info("Validating...")
  logger.info(f"Checkpoints: {','.join(str(x) for x in sorted(iterations))}")

  result = ValidationEntries()
  save_callback = None

  select_best_from_df = None
  # if ns.select_best_from is not None:
  #   select_best_from_df = pd.read_csv(ns.select_best_from, sep="\t")

  custom_hparams = split_hparams_string(ns.custom_hparams)

  for iteration in tqdm(sorted(iterations)):
    logger.info(f"Current checkpoint: {iteration}")
    checkpoint_path = get_checkpoint(ns.checkpoints_dir, iteration)

    taco_checkpoint = try_load_checkpoint(checkpoint_path, ns.device, logger)
    if taco_checkpoint is None:
      return False

    save_callback = partial(
        save_results, val_dir=ns.output_dir, iteration=iteration)

    validation_entries = validate(
        checkpoint=taco_checkpoint,
        data=data,
        custom_hparams=custom_hparams,
        entry_names=ns.files,
        full_run=ns.full_run,
        speaker_name=ns.speaker,
        logger=logger,
        max_decoder_steps=ns.max_decoder_steps,
        fast=not ns.include_stats,
        save_callback=save_callback,
        mcd_no_of_coeffs_per_frame=ns.mcd_no_of_coeffs_per_frame,
        repetitions=ns.repetitions,
        seed=ns.custom_seed,
        device=ns.device,
        select_best_from=select_best_from_df,
    )

    result.extend(validation_entries)

  if len(result) == 0:
    return

  save_stats(ns.output_dir, result)

  # logger.info(
  #     "Wrote all inferred mel paths including sampling rate into these file(s):")
  # npy_path = save_mel_postnet_npy_paths(
  #     val_dir=output_dir,
  #     mel_postnet_npy_paths=mel_postnet_npy_paths
  # )
  # logger.info(npy_path)

  # if copy_mel_info_to is not None:
  #     copy_mel_info_to.parent.mkdir(parents=True, exist_ok=True)
  #     copyfile(npy_path, copy_mel_info_to)
  #     logger.info(copy_mel_info_to)

  logger.info(f"Saved output to: {ns.output_dir.absolute()}")

  return True
