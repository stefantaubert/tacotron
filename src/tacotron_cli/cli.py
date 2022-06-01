import faulthandler
import logging
from argparse import ArgumentParser, Namespace

from tacotron_cli.analysis import init_plot_emb_parser
# from tacotron_cli.eval_checkpoints import eval_checkpoints
from tacotron_cli.inference import init_inference_v2_parser
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
