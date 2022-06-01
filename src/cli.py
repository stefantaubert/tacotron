import faulthandler
import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path


# from tacotron.app.eval_checkpoints import eval_checkpoints
from tacotron.app import plot_embeddings
from tacotron.app.inference import infer_text
from tacotron.app.training import init_continue_train_parser, init_train_parser
from tacotron.app.validation import init_validate_parser
from tacotron.app.weights import map_missing_symbols_v2

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


def init_inference_v2_parser(parser: ArgumentParser) -> None:
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('text', type=Path)
    parser.add_argument('--encoding', type=str, default="UTF-8")
    parser.add_argument('--custom-speaker', type=str, default=None)
    parser.add_argument('--custom-lines', type=int, nargs="*", default=[])
    parser.add_argument('--max-decoder-steps', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=64)
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


def init_add_missing_weights_parser(parser: ArgumentParser) -> None:
    parser.add_argument('checkpoint1', type=Path)
    parser.add_argument('checkpoint2', type=Path)
    parser.add_argument('--mode', type=str,
                        choices=["copy", "predict"], default="copy")
    parser.add_argument('-out', '--custom-output', type=Path, default=None)
    return map_missing_symbols_v2


def add_base_dir(parser: ArgumentParser) -> None:
    if BASE_DIR_VAR in os.environ.keys():
        base_dir = Path(os.environ[BASE_DIR_VAR])
        parser.set_defaults(base_dir=base_dir)


def _add_parser_to(subparsers, name: str, init_method) -> None:
    parser = subparsers.add_parser(name, help=f"{name} help")
    add_base_dir(parser)
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


def configure_logger() -> None:
    loglevel = logging.DEBUG if __debug__ else logging.INFO
    main_logger = logging.getLogger()
    main_logger.setLevel(loglevel)
    main_logger.manager.disable = logging.NOTSET
    if len(main_logger.handlers) > 0:
        console = main_logger.handlers[0]
    else:
        console = logging.StreamHandler()
        main_logger.addHandler(console)

    logging_formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] (%(levelname)s) %(message)s',
        '%Y/%m/%d %H:%M:%S',
    )
    console.setFormatter(logging_formatter)
    console.setLevel(loglevel)


def _process_args(args: Namespace) -> None:
    print("Received args:")
    print(args)
    params = vars(args)
    if "invoke_handler" in params:
        invoke_handler = params.pop("invoke_handler")
        invoke_handler(**params)


if __name__ == "__main__":
    configure_logger()
    faulthandler.enable()
    main_parser = _init_parser()

    received_args = main_parser.parse_args()

    _process_args(received_args)
