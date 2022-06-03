# import os

# from tacotron_cli.io import get_checkpoints_dir, load_prep_settings
# from tacotron_cli.pre.merge_ds import (get_merged_dir, load_merged_accents_ids,
#                                        load_merged_speakers_json,
#                                        load_merged_symbol_converter)
# from tacotron_cli.pre.prepare import get_prep_dir, load_valset
# from tacotron_cli.tacotron.io import get_train_dir
# from tacotron_cli.utils import prepare_logger
# from tacotron.eval_checkpoints import eval_checkpoints


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


# def eval_checkpoints_main(base_dir: Path, train_name: str, select: int, min_it: int, max_it: int):
#   train_dir = get_train_dir(base_dir, train_name, create=False)
#   assert train_dir.is_dir()

#   merge_name, prep_name = load_prep_settings(train_dir)
#   merge_dir = get_merged_dir(base_dir, merge_name, create=False)
#   prep_dir = get_prep_dir(merge_dir, prep_name, create=False)
#   valset = load_valset(prep_dir)

#   symbols_conv = load_merged_symbol_converter(merge_dir)
#   speakers = load_merged_speakers_json(merge_dir)
#   accents = load_merged_accents_ids(merge_dir)

#   logger = prepare_logger()

#   eval_checkpoints(
#     custom_hparams=None,
#     checkpoint_dir=get_checkpoints_dir(train_dir),
#     select=select,
#     min_it=min_it,
#     max_it=max_it,
#     n_symbols=len(symbols_conv),
#     n_speakers=len(speakers),
#     n_accents=len(accents),
#     valset=valset,
#     logger=logger
#   )


# if __name__ == "__main__":
#   eval_checkpoints_main(
#     base_dir="/datasets/models/taco2pt_v5",
#     train_name="debug",
#     select=1,
#     min_it=0,
#     max_it=0
#   )
