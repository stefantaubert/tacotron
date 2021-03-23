import os
from argparse import ArgumentParser
from argparse import ArgumentParser

from tacotron.app.tacotron.analysis import plot_embeddings
from tacotron.app.tacotron.defaults import (DEFAULT_DENOISER_STRENGTH,
                                            DEFAULT_SENTENCE_PAUSE_S, DEFAULT_SIGMA,
                                            DEFAULT_WAVEGLOW)
from tacotron.app.tacotron.eval_checkpoints import eval_checkpoints_main
from tacotron.app.tacotron.inference import infer_main2
from tacotron.app.tacotron.training import (continue_train_main, restore_model,
                                            train_main)
from tacotron.app.tacotron.validation import app_validate
from src.cli.utils import split_hparams_string, split_int_set_str


def init_plot_emb_parser(parser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--custom_checkpoint', type=int)
  return plot_embeddings


def init_eval_checkpoints_parser(parser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--select', type=int)
  parser.add_argument('--min_it', type=int)
  parser.add_argument('--max_it', type=int)
  return eval_checkpoints_main_cli


def eval_checkpoints_main_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  eval_checkpoints_main(**args)


def init_restore_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--checkpoint_dir', type=str, required=True)
  return restore_model


def init_train_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  parser.add_argument('--warm_start_train_name', type=str)
  parser.add_argument('--warm_start_checkpoint', type=int)
  parser.add_argument('--weights_train_name', type=str)
  parser.add_argument('--weights_checkpoint', type=int)
  parser.add_argument('--map_from_speaker', type=str)
  parser.add_argument('--use_weights_map', action='store_true')
  return train_cli


def train_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  train_main(**args)


def init_continue_train_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--custom_hparams', type=str)
  return continue_train_cli


def continue_train_cli(**args):
  args["custom_hparams"] = split_hparams_string(args["custom_hparams"])
  continue_train_main(**args)


def init_validate_parser(parser: ArgumentParser):
  parser.add_argument('--train_name', type=str, required=True)
  parser.add_argument('--entry_ids', type=str, help="Utterance ids or nothing if random")
  parser.add_argument('--speaker', type=str, help="ds_name,speaker_name")
  parser.add_argument('--ds', type=str, help="Choose if validation- or testset should be taken.",
                      choices=["val", "test"], default="val")
  parser.add_argument('--waveglow', type=str, help="Waveglow train_name", default=DEFAULT_WAVEGLOW)
  parser.add_argument('--custom_checkpoints', type=str)
  parser.add_argument("--denoiser_strength", default=DEFAULT_DENOISER_STRENGTH,
                      type=float, help='Removes model bias.')
  parser.add_argument("--sigma", default=DEFAULT_SIGMA, type=float)
  parser.add_argument('--full_run', action='store_true')

  parser.add_argument(
    '--custom_tacotron_hparams',
    type=str
  )

  parser.add_argument(
    '--custom_waveglow_hparams',
    type=str
  )

  return validate_cli


def validate_cli(**args):
  args["custom_tacotron_hparams"] = split_hparams_string(args["custom_tacotron_hparams"])
  args["custom_waveglow_hparams"] = split_hparams_string(args["custom_waveglow_hparams"])
  args["entry_ids"] = split_int_set_str(args["entry_ids"])
  args["custom_checkpoints"] = split_int_set_str(args["custom_checkpoints"])
  app_validate(**args)


def init_inference_parser(parser: ArgumentParser):
  parser.add_argument(
    '--train_name',
    type=str, required=True
  )

  parser.add_argument(
    '--text_name',
    type=str, required=True
  )

  parser.add_argument(
    '--speaker',
    type=str, required=True, help="ds_name,speaker_name"
  )

  parser.add_argument(
    '--sentence_ids',
    type=str,
  )

  parser.add_argument(
    '--waveglow',
    type=str, help="Waveglow train_name", default=DEFAULT_WAVEGLOW
  )

  parser.add_argument(
    '--custom_checkpoint',
    type=int
  )

  parser.add_argument(
    '--sentence_pause_s',
    type=float, default=DEFAULT_SENTENCE_PAUSE_S
  )

  parser.add_argument(
    '--sigma',
    type=float, default=DEFAULT_SIGMA
  )

  parser.add_argument(
    '--denoiser_strength',
    type=float, default=DEFAULT_DENOISER_STRENGTH
  )

  parser.add_argument(
    '--custom_tacotron_hparams',
    type=str
  )

  parser.add_argument(
    '--custom_waveglow_hparams',
    type=str
  )

  parser.add_argument('--full_run', action='store_true')

  return infer_cli


def infer_cli(**args):
  args["custom_tacotron_hparams"] = split_hparams_string(args["custom_tacotron_hparams"])
  args["custom_waveglow_hparams"] = split_hparams_string(args["custom_waveglow_hparams"])
  args["sentence_ids"] = split_int_set_str(args["sentence_ids"])
  infer_main2(**args)


BASE_DIR_VAR = "base_dir"


def add_base_dir(parser: ArgumentParser):
  assert BASE_DIR_VAR in os.environ.keys()
  base_dir = os.environ[BASE_DIR_VAR]
  parser.set_defaults(base_dir=base_dir)


def _add_parser_to(subparsers, name: str, init_method):
  parser = subparsers.add_parser(name, help=f"{name} help")
  invoke_method = init_method(parser)
  parser.set_defaults(invoke_handler=invoke_method)
  add_base_dir(parser)
  return parser


def _init_parser():
  result = ArgumentParser()
  subparsers = result.add_subparsers(help='sub-command help')

  _add_parser_to(subparsers, "tacotron-restore", init_taco_restore_parser)
  _add_parser_to(subparsers, "tacotron-train", init_taco_train_parser)
  _add_parser_to(subparsers, "tacotron-continue-train", init_taco_continue_train_parser)
  _add_parser_to(subparsers, "tacotron-validate", init_taco_val_parser)
  _add_parser_to(subparsers, "tacotron-infer", init_taco_infer_parser)
  _add_parser_to(subparsers, "tacotron-eval-checkpoints", init_taco_eval_checkpoints_parser)
  _add_parser_to(subparsers, "tacotron-plot-embeddings", init_taco_plot_emb_parser)

  return result


def _process_args(args):
  params = vars(args)
  invoke_handler = params.pop("invoke_handler")
  invoke_handler(**params)


if __name__ == "__main__":
  main_parser = _init_parser()

  received_args = main_parser.parse_args()
  #args = main_parser.parse_args("ljs-text --base_dir=/datasets/models/taco2pt_v2 --mel_name=ljs --ds_name=test_ljs --convert_to_ipa".split())

  _process_args(received_args)
