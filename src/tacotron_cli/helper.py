from argparse import ArgumentParser

from tacotron_cli.argparse_helper import (get_optional, parse_device, parse_non_empty,
                                          parse_positive_integer)
from tacotron_cli.defaults import DEFAULT_DEVICE, DEFAULT_MAX_DECODER_STEPS


def add_device_argument(parser: ArgumentParser) -> None:
  parser.add_argument("--device", type=parse_device, metavar="DEVICE", default=DEFAULT_DEVICE,
                      help="use this device, e.g., \"cpu\" or \"cuda:0\"")


def add_hparams_argument(parser: ArgumentParser) -> None:
  parser.add_argument('--custom-hparams', type=get_optional(parse_non_empty),
                      metavar="CUSTOM-HYPERPARAMETERS", default=None, help="custom hyperparameters comma separated")


def add_max_decoder_steps_argument(parser: ArgumentParser) -> None:
  parser.add_argument('--max-decoder-steps', type=parse_positive_integer, metavar="MAX-DECODER-STEPS",
                      default=DEFAULT_MAX_DECODER_STEPS, help="maximum step count before synthesis is stopped")
