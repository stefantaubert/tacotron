# from tacotron.core.model_checkpoint import convert_v1_to_v2_model
# from typing import Dict, Optional

# from tacotron.app.pre.prepare import (get_merge_dir, load_prep_accents_ids,
#                                       load_prep_speakers_json,
#                                       load_prep_symbol_converter)


# def convert_model(base_dir: str, prep_name: str, model_path: str, custom_hparams: Optional[Dict[str, str]]):
#   prep_dir = get_merge_dir(base_dir, prep_name)

#   convert_v1_to_v2_model(
#     old_model_path=model_path,
#     custom_hparams=custom_hparams,
#     speakers=load_prep_speakers_json(prep_dir),
#     accents=load_prep_accents_ids(prep_dir),
#     symbols=load_prep_symbol_converter(prep_dir)
#   )


# if __name__ == "__main__":
#   convert_model(
#     base_dir="/datasets/models/taco2pt_v5",
#     prep_name="ljs_ipa",
#     model_path="/datasets/models/taco2pt_v5/tacotron/ljs_ipa_warm/checkpoints/29379.pt",
#     custom_hparams={
#       "batch_size": 26,
#       "iters_per_checkpoint": 500,
#       "epochs": 1000
#     }
#   )
