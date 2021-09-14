# import os

# from tacotron.app.io import get_checkpoints_dir, load_prep_settings
# from tacotron.app.pre.merge_ds import (get_merged_dir, load_merged_accents_ids,
#                                        load_merged_speakers_json,
#                                        load_merged_symbol_converter)
# from tacotron.app.pre.prepare import get_prep_dir, load_valset
# from tacotron.app.tacotron.io import get_train_dir
# from tacotron.app.utils import prepare_logger
# from tacotron.core.eval_checkpoints import eval_checkpoints


# def eval_checkpoints_main(base_dir: Path, train_name: str, select: int, min_it: int, max_it: int):
#   train_dir = get_train_dir(base_dir, train_name, create=False)
#   assert os.path.isdir(train_dir)

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
