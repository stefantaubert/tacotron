import os
from argparse import ArgumentParser
from pathlib import Path

from general_utils import split_hparams_string, split_int_set_str

# from tacotron.app.eval_checkpoints import eval_checkpoints
from tacotron.app import (DEFAULT_MAX_DECODER_STEPS, continue_train, infer,
                          plot_embeddings, train, validate)
from tacotron.app.defaults import (DEFAULT_MCD_NO_OF_COEFFS_PER_FRAME,
                                   DEFAULT_REPETITIONS,
                                   DEFAULT_SAVE_MEL_INFO_COPY_PATH,
                                   DEFAULT_SEED)

BASE_DIR_VAR = "base_dir"


def init_plot_emb_parser(parser) -> None:
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--custom_checkpoint', type=int)
  return plot_embeddings


# def init_eval_checkpoints_parser(parser):
#   parser.add_argument('--train_name', type=str, required=True)
#   parser.add_argument('--custom_hparams', type=str)
#   parser.add_argument('--select', type=int)
#   parser.add_argument('--min_it', type=int)
#   parser.add_argument('--max_it', type=int)
#   return eval_checkpoints_main_cli


# def evaeckpoints_main_cli(**args):
#   argsl_ch["custom_hparams"] = split_hparams_string(args["custom_hparams"])
#   eval_checkpoints(**args)


# def init_restore_parser(parser: ArgumentParser) -> None:
#   parser.add_argument('--train_name', type=str, required=True)
#   parser.add_argument('--checkpoint_dir', type=Path, required=True)
#   return restore_model


def init_train_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--ttsp_dir', type=Path, required=True)
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--warm_start_train_name', type=str)
  parser.add_argument('--warm_start_checkpoint', type=int)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--weights_train_name', type=str)
  parser.add_argument('--weights_checkpoint', type=int)
  parser.add_argument('--map_from_speaker', type=str)
  parser.add_argument('--use_weights_map', action='store_true')
  return train_cli


def train_cli(**args) -> None:
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  train(**args)


def init_continue_train_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return continue_train_cli


def continue_train_cli(**args) -> None:
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  continue_train(**args)


def init_validate_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--entry_ids', type=str, help="Utterance ids or nothing if random")
  parser.add_argument('--speaker', type=str, help="ds_name,speaker_name")
  parser.add_argument('--ds', type=str, help="Choose if validation- or testset should be taken.",
                      choices=["val", "test"], default="val")
  parser.add_argument('--custom_checkpoints', type=str)
  parser.add_argument('--full_run', action='store_true')
  parser.add_argument('--max_decoder_steps', type=int, default=DEFAULT_MAX_DECODER_STEPS)
  parser.add_argument('--copy_mel_info_to', type=str, default=DEFAULT_SAVE_MEL_INFO_COPY_PATH)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--select_best_from', type=str)
  parser.add_argument('--mcd_no_of_coeffs_per_frame', type=int,
                      default=DEFAULT_MCD_NO_OF_COEFFS_PER_FRAME)
  parser.add_argument('--fast', action='store_true')
  parser.add_argument('--repetitions', type=int, default=DEFAULT_REPETITIONS)
  parser.add_argument('--seed', type=int, default=DEFAULT_SEED)

  return validate_cli


def validate_cli(**args) -> None:
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  args["entry_ids"] = split_int_set_str(args["entry_ids"])
  args["custom_checkpoints"] = split_int_set_str(args["custom_checkpoints"])
  validate(**args)


def init_inference_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--speaker', type=str, required=True, help="ds_name,speaker_name")
  parser.add_argument('--utterance_ids', type=str)
  parser.add_argument('--custom_checkpoint', type=int)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--full_run', action='store_true')
  parser.add_argument('--max_decoder_steps', type=int, default=DEFAULT_MAX_DECODER_STEPS)
  parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
  parser.add_argument('--copy_mel_info_to', type=str, default=DEFAULT_SAVE_MEL_INFO_COPY_PATH)

  return infer_cli


def infer_cli(**args) -> None:
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  args["utterance_ids"] = split_int_set_str(args["utterance_ids"])
  infer(**args)


def add_base_dir(parser: ArgumentParser) -> None:
  assert BASE_DIR_VAR in os.environ.keys()
  base_dir = Path(os.environ[BASE_DIR_VAR])
  parser.set_defaults(base_dir=base_dir)


def _add_parser_to(subparsers, name: str, init_method) -> None:
  parser = subparsers.add_parser(name, help=f"{name} help")
  invoke_method = init_method(parser)
  parser.set_defaults(invoke_handler=invoke_method)
  add_base_dir(parser)
  return parser


def _init_parser():
  result = ArgumentParser()
  subparsers = result.add_subparsers(help='sub-command help')

  _add_parser_to(subparsers, "train", init_train_parser)
  _add_parser_to(subparsers, "continue-train", init_continue_train_parser)
  _add_parser_to(subparsers, "validate", init_validate_parser)
  _add_parser_to(subparsers, "infer", init_inference_parser)
  # _add_parser_to(subparsers, "eval-checkpoints", init_taco_eval_checkpoints_parser)
  _add_parser_to(subparsers, "plot-embeddings", init_plot_emb_parser)
  #_add_parser_to(subparsers, "restore", init_restore_parser)

  return result


def _process_args(args) -> None:
  params = vars(args)
  invoke_handler = params.pop("invoke_handler")
  invoke_handler(**params)


if __name__ == "__main__":
  main_parser = _init_parser()

  received_args = main_parser.parse_args()

  _process_args(received_args)
