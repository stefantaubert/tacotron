import argparse
import faulthandler
import logging
import platform
import sys
from argparse import ArgumentParser, Namespace
from importlib.metadata import version
from logging import getLogger
from pathlib import Path
from pkgutil import iter_modules
from tempfile import gettempdir
from time import perf_counter
from typing import Callable, Generator, List, Tuple

from tacotron_cli.analysis import init_plot_emb_parser
from tacotron_cli.argparse_helper import get_optional, parse_path
# from tacotron_cli.eval_checkpoints import eval_checkpoints
from tacotron_cli.inference import init_inference_v2_parser
from tacotron_cli.logging_configuration import (configure_root_logger, get_file_logger,
                                                try_init_file_logger)
from tacotron_cli.training import init_continue_train_parser, init_train_parser
from tacotron_cli.validation import init_validate_parser
from tacotron_cli.weights import init_add_missing_weights_parser

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


def _add_parser_to(subparsers, name: str, init_method) -> None:
  parser = subparsers.add_parser(name, help=f"{name} help")
  invoke_method = init_method(parser)
  parser.set_defaults(invoke_handler=invoke_method)
  return parser


def _init_parser():
  result = ArgumentParser()
  subparsers = result.add_subparsers(help='sub-command help')

  _add_parser_to(subparsers, "train", init_train_parser)
  _add_parser_to(subparsers, "continue-train", init_continue_train_parser)
  _add_parser_to(subparsers, "validate", init_validate_parser)
  _add_parser_to(subparsers, "infer-text", init_inference_v2_parser)
  # _add_parser_to(subparsers, "eval-checkpoints", init_taco_eval_checkpoints_parser)
  _add_parser_to(subparsers, "plot-embeddings", init_plot_emb_parser)
  _add_parser_to(subparsers, "add-missing-symbols",
                 init_add_missing_weights_parser)
  #_add_parser_to(subparsers, "restore", init_restore_parser)

  return result


__version__ = version("tacotron")

INVOKE_HANDLER_VAR = "invoke_handler"

CONSOLE_PNT_GREEN = "\x1b[1;49;32m"
CONSOLE_PNT_RED = "\x1b[1;49;31m"
CONSOLE_PNT_RST = "\x1b[0m"


Parsers = Generator[Tuple[str, str, Callable[[ArgumentParser],
                                             Callable[..., bool]]], None, None]


def formatter(prog):
  return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=40)


def get_parsers() -> Parsers:
  yield "train", "train", init_train_parser
  yield "continue-train", "continue-train", init_continue_train_parser
  yield "validate", "validate", init_validate_parser
  yield "infer-text", "infer-text", init_inference_v2_parser
  yield "plot-embeddings", "plot-embeddings", init_plot_emb_parser
  yield "add-missing-symbols", "add-missing-symbols", init_add_missing_weights_parser


def print_features():
  parsers = get_parsers()
  for command, description, method in parsers:
    print(f"- `{command}`: {description}")


def _init_parser():
  main_parser = ArgumentParser(
    formatter_class=formatter,
    description="This program provides methods to modify a text file.",
  )
  main_parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
  subparsers = main_parser.add_subparsers(help="description")
  default_log_path = Path(gettempdir()) / "txt-utils.log"

  methods = get_parsers()
  for command, description, method in methods:
    method_parser = subparsers.add_parser(
      command, help=description, formatter_class=formatter)
    method_parser.set_defaults(**{
      INVOKE_HANDLER_VAR: method(method_parser),
    })
    logging_group = method_parser.add_argument_group("logging arguments")
    logging_group.add_argument("--log", type=get_optional(parse_path), metavar="FILE",
                               nargs="?", const=None, help="path to write the log", default=default_log_path)
    logging_group.add_argument("--debug", action="store_true",
                               help="include debugging information in log")

  return main_parser


def parse_args(args: List[str]) -> None:
  configure_root_logger()
  logger = getLogger()

  local_debugging = debug_file_exists()
  if local_debugging:
    logger.debug(f"Received arguments: {str(args)}")

  parser = _init_parser()

  try:
    ns = parser.parse_args(args)
  except SystemExit:
    # invalid command supplied
    return

  if hasattr(ns, INVOKE_HANDLER_VAR):
    invoke_handler: Callable[..., bool] = getattr(ns, INVOKE_HANDLER_VAR)
    delattr(ns, INVOKE_HANDLER_VAR)
    log_to_file = ns.log is not None
    if log_to_file:
      log_to_file = try_init_file_logger(ns.log, local_debugging or ns.debug)
      if not log_to_file:
        logger.warning("Logging to file is not possible.")

    flogger = get_file_logger()
    if not local_debugging:
      sys_version = sys.version.replace('\n', '')
      flogger.debug(f"CLI version: {__version__}")
      flogger.debug(f"Python version: {sys_version}")
      flogger.debug("Modules: %s", ', '.join(sorted(p.name for p in iter_modules())))

      my_system = platform.uname()
      flogger.debug(f"System: {my_system.system}")
      flogger.debug(f"Node Name: {my_system.node}")
      flogger.debug(f"Release: {my_system.release}")
      flogger.debug(f"Version: {my_system.version}")
      flogger.debug(f"Machine: {my_system.machine}")
      flogger.debug(f"Processor: {my_system.processor}")

    flogger.debug(f"Received arguments: {str(args)}")
    flogger.debug(f"Parsed arguments: {str(ns)}")

    start = perf_counter()
    success = invoke_handler(ns)

    if success:
      logger.info(f"{CONSOLE_PNT_GREEN}Everything was successful!{CONSOLE_PNT_RST}")
      flogger.info("Everything was successful!")
    else:
      if log_to_file:
        logger.error(
          "Not everything was successful! See log for details.")
      else:
        logger.error(
          "Not everything was successful!")
      flogger.error("Not everything was successful!")

    duration = perf_counter() - start
    flogger.debug(f"Total duration (s): {duration}")

    if log_to_file:
      logger.info(f"Written log to: {ns.log.absolute()}")

  else:
    parser.print_help()


def run():
  arguments = sys.argv[1:]
  parse_args(arguments)


def run_prod():
  run()


def debug_file_exists():
  return (Path(gettempdir()) / "tacotron-debug").is_file()


def create_debug_file():
  (Path(gettempdir()) / "tacotron-debug").write_text("", "UTF-8")


if __name__ == "__main__":
  run_prod()
