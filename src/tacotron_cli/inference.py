import datetime
import random
from argparse import ArgumentParser
from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from typing import List, Optional
from typing import OrderedDict as OrderedDictType

import imageio
import numpy as np
from image_utils import stack_images_vertically
from ordered_set import OrderedSet
from tacotron.audio_utils import plot_melspec_np
from tacotron.checkpoint_handling import get_learning_rate, get_speaker_mapping
from tacotron.synthesizer import Synthesizer
from tacotron.typing import Speaker, Symbols
from tacotron.utils import plot_alignment_np_new
from tqdm import tqdm

from tacotron_cli.io import load_checkpoint

Utterances = OrderedDictType[int, Symbols]
Paragraphs = OrderedDictType[int, Utterances]


def split_adv(s: str, sep: str) -> List[str]:
    if len(sep) == 0:
        return list(s)
    return s.split(sep)


def parse_paragraphs_from_text(text: str, sep: str) -> Paragraphs:
    lines = text.splitlines()
    result = OrderedDict()
    paragraph_nr = 1
    current_utterances = OrderedDict()
    for line_nr, line in enumerate(lines, start=1):
        if line == "":
            if len(current_utterances) > 0:
                assert paragraph_nr not in result
                result[paragraph_nr] = current_utterances
                paragraph_nr += 1
                current_utterances = OrderedDict()
        else:
            line_symbols = split_adv(line, sep)
            assert line_nr not in current_utterances
            current_utterances[line_nr] = line_symbols

    if len(current_utterances) > 0:
        result[paragraph_nr] = current_utterances
    return result


def init_inference_v2_parser(parser: ArgumentParser) -> None:
    parser.add_argument('checkpoint', metavar="CHECKPOINT-PATH", type=Path)
    parser.add_argument('text', metavar="TEXT-PATH", type=Path)
    parser.add_argument('--encoding', type=str, default="UTF-8")
    parser.add_argument('--custom-speaker', type=str, default=None)
    parser.add_argument('--custom-lines', type=int, nargs="*", default=[])
    parser.add_argument('--max-decoder-steps', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sep', type=str, default="")
    parser.add_argument('--custom-seed', type=int, default=None)
    parser.add_argument('-p', '--paragraph-directories', action='store_true')
    parser.add_argument('--include-stats', action='store_true')
    parser.add_argument('--prepend', type=str, default="",
                        help="prepend text to all output file names")
    parser.add_argument('--append', type=str, default="",
                        help="append text to all output file names")
    parser.add_argument('-out', '--output-directory', type=Path, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true')
    return infer_text


def infer_text(checkpoint: Path, text: Path, encoding: str, custom_speaker: Optional[Speaker], custom_lines: List[int], max_decoder_steps: int, batch_size: int, include_stats: bool, custom_seed: Optional[int], paragraph_directories: bool, output_directory: Optional[Path], prepend: str, append: str, sep: str, overwrite: bool) -> bool:
    logger = getLogger(__name__)

    if not checkpoint.is_file():
        logger.error("Checkpoint was not found!")
        return False

    if not text.is_file():
        logger.error("Text was not found!")
        return False

    custom_lines = OrderedSet(custom_lines)
    if not all(x >= 0 for x in custom_lines):
        logger.error(
            "Custom line values need to be greater than or equal to zero!")
        return False

    if not max_decoder_steps > 0:
        logger.error("Maximum decoder steps need to be greater than zero!")
        return False

    if not batch_size > 0:
        logger.error("Batch size need to be greater than zero!")
        return False

    if custom_seed is not None and not custom_seed >= 0:
        logger.error("Custom seed needs to be greater than or equal to zero!")
        return False

    try:
        logger.debug(f"Loading checkpoint...")
        checkpoint_dict = load_checkpoint(checkpoint)
    except Exception as ex:
        logger.error("Checkpoint couldn't be loaded!")
        return False

    if custom_speaker is not None:
        speaker_mapping = get_speaker_mapping(checkpoint_dict)
        if custom_speaker not in speaker_mapping:
            logger.error("Custom speaker was not found!")
            return False

    try:
        logger.debug(f"Loading text.")
        text_content = text.read_text(encoding)
    except Exception as ex:
        logger.error("Text couldn't be read!")
        return False

    if output_directory is None:
        output_directory = text.parent / text.stem

    if output_directory.is_file():
        logger.error("Output directory is a file!")
        return False

    paragraphs = parse_paragraphs_from_text(text_content, sep)

    line_nrs_to_infer = OrderedSet(
        line_nr for par in paragraphs.values() for line_nr in par.keys())
    if len(custom_lines) > 0:
        for custom_line in custom_lines:
            if custom_line not in line_nrs_to_infer:
                logger.error(f"Line {custom_line} is not inferable!")
                return False

        line_nrs_to_infer = custom_lines

    logger.info("Inferring...")
    logger.info(
        f"Checkpoint learning rate was: {get_learning_rate(checkpoint_dict)}")

    if custom_seed is not None:
        seed = custom_seed
    else:
        seed = random.randint(1, 9999)
        logger.info(f"Using random seed: {seed}.")

    if custom_speaker is not None:
        speaker = custom_speaker
    else:
        speaker_mapping = get_speaker_mapping(checkpoint_dict)
        speaker = next(iter(speaker_mapping.keys()))
        logger.debug(f"Speaker: {speaker}")

    synth = Synthesizer(
        checkpoint=checkpoint_dict,
        custom_hparams=None,
        logger=logger,
    )

    max_paragraph_nr = max(paragraphs.keys())
    max_line_nr = max(utt_nr for paragraph in paragraphs.values()
                      for utt_nr in paragraph.keys())
    zfill_paragraph = len(str(max_paragraph_nr))
    zfill_line_nr = len(str(max_line_nr))
    count_utterances = sum(1 for paragraph in paragraphs.values()
                           for _ in paragraph.keys())
    count_utterances_zfill = len(str(count_utterances))
    utterance_nr = 1
    unknown_symbols = set()
    with tqdm(total=len(line_nrs_to_infer), unit=" lines", ncols=100, desc="Inference") as progress_bar:
        for paragraph_nr, utterances in paragraphs.items():
            if paragraph_directories:
                min_utt = min(utterances.keys())
                max_utt = max(utterances.keys())
                name = f"{paragraph_nr}".zfill(zfill_paragraph)
                min_utt_str = f"{min_utt}".zfill(zfill_line_nr)
                max_utt_str = f"{max_utt}".zfill(zfill_line_nr)
                paragraph_folder = output_directory / \
                    f"{name}-{min_utt_str}-{max_utt_str}"
            else:
                paragraph_folder = output_directory
            for line_nr, utterance in utterances.items():
                if line_nr not in line_nrs_to_infer:
                    logger.debug(f"Skipped line {line_nr}.")
                    utterance_nr += 1
                    continue

                line_nr_filled = f"{line_nr}".zfill(zfill_line_nr)
                utterance_nr_filled = f"{utterance_nr}".zfill(
                    count_utterances_zfill)
                utt_path_stem = f"{prepend}{line_nr_filled}-{utterance_nr_filled}{append}"
                utterance_mel_path = paragraph_folder / f"{utt_path_stem}.npy"

                if utterance_mel_path.exists() and not overwrite:
                    logger.info(
                        f"Line {line_nr}: Skipped inference because line is already synthesized!")
                    continue

                if include_stats:
                    log_out = paragraph_folder / f"{utt_path_stem}.log"
                    align_img_path = paragraph_folder / \
                        f"{utt_path_stem}-1-alignments.png"
                    mel_prepost_img_path = paragraph_folder / \
                        f"{utt_path_stem}-2-prepost.png"
                    mel_postnet_img_path = paragraph_folder / \
                        f"{utt_path_stem}-3-postnet.png"
                    comp_img_path = paragraph_folder / f"{utt_path_stem}.png"

                    if not overwrite:
                        if log_out.exists():
                            logger.info(
                                f"Line {line_nr}: Log already exists! Skipped inference.")
                            continue

                        if mel_postnet_img_path.exists():
                            logger.info(
                                f"Line {line_nr}: Mel image already exists! Skipped inference.")
                            continue

                        if mel_prepost_img_path.exists():
                            logger.info(
                                f"Line {line_nr}: Mel pre-postnet image already exists! Skipped inference.")
                            continue

                        if align_img_path.exists():
                            logger.info(
                                f"Line {line_nr}: Alignments image already exists! Skipped inference.")
                            continue

                        if comp_img_path.exists():
                            logger.info(
                                f"Line {line_nr}: Comparison image already exists! Skipped inference.")
                            continue

                logger.debug(f"Infering {line_nr}...")

                inf_sent_output = synth.infer_v2(
                    symbols=utterance,
                    speaker=speaker,
                    include_stats=include_stats,
                    max_decoder_steps=max_decoder_steps,
                    seed=seed,
                )

                logger.debug(f"Saving {utterance_mel_path}...")
                paragraph_folder.mkdir(parents=True, exist_ok=True)
                np.save(utterance_mel_path, inf_sent_output.mel_outputs_postnet)

                unknown_symbols |= inf_sent_output.unknown_symbols

                if include_stats:
                    log_lines = []
                    log_lines.append(f"Timepoint: {datetime.datetime.now()}")
                    log_lines.append(
                        f"Reached max decoder steps: {inf_sent_output.reached_max_decoder_steps}")
                    log_lines.append(
                        f"Inference duration: {inf_sent_output.inference_duration_s}")
                    log_lines.append(
                        f"Sampling rate: {inf_sent_output.sampling_rate}")
                    if len(unknown_symbols) > 0:
                        log_lines.append(
                            f"Unknown symbols: {' '.join(inf_sent_output.unknown_symbols)}")
                    else:
                        log_lines.append("No unknown symbols.")

                    logger.debug(f"Saving {log_out}...")
                    log_out.write_text("\n".join(log_lines), encoding="UTF-8")

                    logger.debug(f"Saving {mel_postnet_img_path}...")
                    _, postnet_img = plot_melspec_np(
                        inf_sent_output.mel_outputs_postnet)
                    imageio.imsave(mel_postnet_img_path, postnet_img)

                    logger.debug(f"Saving {mel_prepost_img_path}...")
                    _, mel_img = plot_melspec_np(inf_sent_output.mel_outputs)
                    imageio.imsave(mel_prepost_img_path, mel_img)

                    logger.debug(f"Saving {align_img_path}...")
                    _, alignments_img = plot_alignment_np_new(
                        inf_sent_output.alignments)
                    imageio.imsave(align_img_path, alignments_img)

                    logger.debug(f"Saving {comp_img_path}...")
                    stack_images_vertically(
                        list_im=[
                            align_img_path,
                            mel_prepost_img_path,
                            mel_postnet_img_path,
                        ],
                        out_path=comp_img_path,
                    )
                progress_bar.update()
                utterance_nr += 1

    if len(unknown_symbols) > 0:
        logger.warning(
            f"Unknown symbols: {' '.join(sorted(unknown_symbols))} (#{len(unknown_symbols)})")
    logger.info(f"Done. Written output to: {output_directory.absolute()}")
    return True
