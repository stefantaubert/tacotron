from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Set

import imageio
import numpy as np
import pandas as pd
from tacotron.utils import split_hparams_string
from scipy.io.wavfile import write
from speech_dataset_parser_api import parse_directory
from tacotron.globals import DEFAULT_CSV_SEPERATOR
from tacotron.image_utils import stack_images_vertically
from tacotron.parser import get_entries_from_sdp_entries
from tacotron.typing import Entry
from tacotron.utils import get_checkpoint, get_last_checkpoint, prepare_logger
from tacotron.validation import (ValidationEntries, ValidationEntryOutput,
                                 get_df, validate)
from tqdm import tqdm

from tacotron_cli.defaults import (DEFAULT_MAX_DECODER_STEPS,
                                   DEFAULT_MCD_NO_OF_COEFFS_PER_FRAME,
                                   DEFAULT_REPETITIONS, DEFAULT_SEED)
from tacotron_cli.io import load_checkpoint

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


def init_validate_parser(parser: ArgumentParser) -> None:
    parser.add_argument('checkpoints_dir',
                        metavar="CHECKPOINTS-FOLDER-PATH", type=Path)
    parser.add_argument('output_dir',
                        metavar="OUTPUT-FOLDER-PATH", type=Path)
    parser.add_argument('dataset_dir', metavar="DATA-FOLDER-PATH",
                        type=Path, help="train or val set folder")
    parser.add_argument("tier", metavar="TIER", type=str)
    parser.add_argument('--entry-names', type=str, nargs="*",
                        help="Utterance names or nothing if random", default=[])
    parser.add_argument('--speaker', type=str, help="ds_name,speaker_name")
    parser.add_argument('--custom-checkpoints',
                        type=int, nargs="*", default=[])
    parser.add_argument('--full-run', action='store_true')
    parser.add_argument('--max-decoder-steps', type=int,
                        default=DEFAULT_MAX_DECODER_STEPS)
    # parser.add_argument('--copy-mel_info_to', type=Path,
    #                     default=DEFAULT_SAVE_MEL_INFO_COPY_PATH)
    parser.add_argument('--custom-hparams', type=str)
    parser.add_argument('--select-best-from', type=str)
    parser.add_argument('--mcd-no-of-coeffs-per-frame', type=int,
                        default=DEFAULT_MCD_NO_OF_COEFFS_PER_FRAME)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--repetitions', type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)

    return validate_cli


def validate_cli(**args) -> None:
    args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
    validate_v2(**args)


def validate_v2(checkpoints_dir: Path, output_dir: Path, dataset_dir: Path, tier: str, entry_names: List[str] = None, speaker: Optional[str] = None, custom_checkpoints: Optional[List[int]] = None, custom_hparams: Optional[Dict[str, str]] = None, full_run: bool = False, max_decoder_steps: int = DEFAULT_MAX_DECODER_STEPS, mcd_no_of_coeffs_per_frame: int = DEFAULT_MCD_NO_OF_COEFFS_PER_FRAME, fast: bool = False, repetitions: int = DEFAULT_REPETITIONS, select_best_from: Optional[Path] = None, seed: Optional[int] = DEFAULT_SEED) -> None:
    assert repetitions > 0

    data = list(get_entries_from_sdp_entries(
        parse_directory(dataset_dir, tier, 16)))

    iterations: Set[int] = set()

    if len(custom_checkpoints) == 0:
        _, last_it = get_last_checkpoint(checkpoints_dir)
        iterations.add(last_it)
    else:
        # if len(custom_checkpoints) == 0:
        #     iterations = set(get_all_checkpoint_iterations(checkpoint_dir))
        # else:
        iterations = set(custom_checkpoints)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_log_path = output_dir / "log.txt"
    logger = prepare_logger(val_log_path, reset=True)
    logger.info("Validating...")
    logger.info(f"Checkpoints: {','.join(str(x) for x in sorted(iterations))}")

    result = ValidationEntries()
    save_callback = None

    select_best_from_df = None
    if select_best_from is not None:
        select_best_from_df = pd.read_csv(select_best_from, sep="\t")

    for iteration in tqdm(sorted(iterations)):
        logger.info(f"Current checkpoint: {iteration}")
        checkpoint_path = get_checkpoint(checkpoints_dir, iteration)
        taco_checkpoint = load_checkpoint(checkpoint_path)
        save_callback = partial(
            save_results, val_dir=output_dir, iteration=iteration)

        validation_entries = validate(
            checkpoint=taco_checkpoint,
            data=data,
            custom_hparams=custom_hparams,
            entry_names=set(entry_names),
            full_run=full_run,
            speaker_name=speaker,
            logger=logger,
            max_decoder_steps=max_decoder_steps,
            fast=fast,
            save_callback=save_callback,
            mcd_no_of_coeffs_per_frame=mcd_no_of_coeffs_per_frame,
            repetitions=repetitions,
            seed=seed,
            select_best_from=select_best_from_df,
        )

        result.extend(validation_entries)

    if len(result) == 0:
        return

    save_stats(output_dir, result)

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

    logger.info(f"Saved output to: {output_dir}")
