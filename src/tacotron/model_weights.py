# from dataclasses import asdict
# from logging import Logger, getLogger
# from typing import Dict
# from typing import OrderedDict as OrderedDictType
# from typing import Set

# import torch
# from tacotron.model import (SYMBOL_EMBEDDING_LAYER_NAME)
# from text_utils import SymbolIdDict, SymbolsMap
# from text_utils.types import Symbol, SymbolId
# from torch import Tensor


# def map_weights(model_symbols_id_map: OrderedDictType[SymbolId, SymbolId], model_weights: Tensor, trained_weights: Tensor, logger: Logger) -> None:
#     for map_to_id, map_from_id in model_symbols_id_map.items():
#         assert 0 <= map_to_id < model_weights.shape[0]
#         assert 0 <= map_from_id < trained_weights.shape[0]
#         old_weights = model_weights[map_to_id].cpu().numpy()[:5]
#         model_weights[map_to_id] = trained_weights[map_from_id]
#         logger.debug(f"Mapped {map_from_id} to {map_to_id}.")
#         logger.debug(f"Old {old_weights}")
#         logger.debug(f"New {model_weights[map_to_id].cpu().numpy()[:5]}")


# def get_mapped_symbol_weights(model_symbols: SymbolIdDict, trained_weights: Tensor, trained_symbols: SymbolIdDict, custom_mapping: Optional[SymbolsMap], hparams: HParams, logger: Logger) -> Tensor:
#   symbols_match_not_model = trained_weights.shape[0] != len(trained_symbols)
#   if symbols_match_not_model:
#     logger.exception(
#       f"Weights mapping: symbol space from pretrained model ({trained_weights.shape[0]}) did not match amount of symbols ({len(trained_symbols)}).")
#     raise Exception()

#   if custom_mapping is None:
#     symbols_map = SymbolsMap.from_intersection(
#       map_to=model_symbols.get_all_symbols(),
#       map_from=trained_symbols.get_all_symbols(),
#     )
#   else:
#     symbols_map = custom_mapping
#     symbols_map.remove_unknown_symbols(
#       known_to_symbol=model_symbols.get_all_symbols(),
#       known_from_symbols=trained_symbols.get_all_symbols()
#     )

#   # Remove all empty mappings
#   symbols_wo_mapping = symbols_map.get_symbols_with_empty_mapping()
#   symbols_map.pop_batch(symbols_wo_mapping)

#   symbols_id_map = symbols_map.convert_to_symbols_ids_map(
#     to_symbols=model_symbols,
#     from_symbols=trained_symbols,
#   )

#   model_weights = get_symbol_weights(hparams)

#   map_weights(
#     model_symbols_id_map=symbols_id_map,
#     model_weights=model_weights,
#     trained_weights=trained_weights,
#     logger=logger
#   )

#   not_existing_symbols = model_symbols.get_all_symbols() - symbols_map.keys()
#   no_mapping = symbols_wo_mapping | not_existing_symbols
#   if len(no_mapping) > 0:
#     logger.warning(f"Following symbols were not mapped: {no_mapping}")
#   else:
#     logger.info("All symbols were mapped.")

#   return model_weights


# def get_mapped_speaker_weights(model_speaker_id_dict: SpeakersDict, trained_weights: Tensor, trained_speaker: SpeakersDict, map_from_speaker_name: Speaker, hparams: HParams, logger: Logger) -> Tensor:
#   map_from_id = trained_speaker.get_id(map_from_speaker_name)
#   speakers_map: OrderedDictType[int, int] = OrderedDict(
#     {new_speaker_id: map_from_id for new_speaker_id in model_speaker_id_dict.values()})

#   weights = get_speaker_weights(hparams)

#   map_weights(
#     model_symbols_id_map=speakers_map,
#     model_weights=weights,
#     trained_weights=trained_weights,
#     logger=logger
#   )

#   return weights


# class AddSymbolEmbeddings():
#     def __init__(self) -> None:
#         self.symbols_id_mapping = None
#         self.input_weights = None
#         self.input_symbols = None
#         self.average_shift = None

#     def get_symbols_mapping(self, input_symbols: SymbolIdDict, target_symbols: SymbolIdDict):
#         symbols_mapping = SymbolsMap.from_intersection(
#             map_from=input_symbols.get_all_symbols(),
#             map_to=target_symbols.get_all_symbols(),
#         )

#         self.symbols_id_mapping = symbols_mapping.convert_to_symbols_ids_map(
#             from_symbols=input_symbols,
#             to_symbols=target_symbols,
#         )

#         return self.symbols_id_mapping

#     def get_difference_vector_to_add(self, input_weights: Tensor, target_weights: Tensor, index_mapping: Dict[int, int]) -> Tensor:
#         assert input_weights.shape[1] == target_weights.shape[1]
#         difference_vectors = torch.stack([target_weights[target_index] - input_weights[input_index]
#                                           for target_index, input_index in index_mapping.items()])
#         average_difference_vector = torch.mean(difference_vectors, 0)

#         return average_difference_vector

#     def fit(self, input_weights: Tensor, input_symbols: SymbolIdDict, target_weights: Tensor, target_symbols: SymbolIdDict):
#         symbols_mapping = self.get_symbols_mapping(
#             input_symbols, target_symbols)
#         self.input_weights = input_weights
#         self.input_symbols = input_symbols
#         self.average_shift = self.get_difference_vector_to_add(
#             input_weights, target_weights, symbols_mapping)

#     def predict(self, input_symbol: Symbol) -> Tensor:
#         symbol_index = self.input_symbols.get_id(input_symbol)
#         embedding_input_symbols = self.input_weights[symbol_index]
#         predicted_embedding = embedding_input_symbols + self.average_shift
#         return predicted_embedding


# def map_symbols(input_model: CheckpointTacotron, target_model: CheckpointTacotron, symbols: Set[Symbol]) -> None:
#     logger = getLogger(__name__)
#     input_embedding: Tensor = input_model.model_state_dict[SYMBOL_EMBEDDING_LAYER_NAME]
#     input_symbols = input_model.get_symbols()

#     target_embedding: Tensor = target_model.model_state_dict[SYMBOL_EMBEDDING_LAYER_NAME]
#     target_symbols = target_model.get_symbols()

#     mapper = AddSymbolEmbeddings()
#     mapper.fit(
#         input_weights=input_embedding,
#         input_symbols=input_symbols,
#         target_symbols=target_symbols,
#         target_weights=target_embedding,
#     )

#     for symbol in symbols:
#         logger.info(f"Mapping \"{symbol}\"...")
#         target_hparams = target_model.get_hparams(logger)
#         new_vector = mapper.predict(symbol)
#         logger.info(new_vector[:7])
#         s = torch.reshape(
#             new_vector, (1, target_hparams.symbols_embedding_dim))
#         target_embedding = torch.cat((target_embedding, s))
#         target_model.model_state_dict[SYMBOL_EMBEDDING_LAYER_NAME] = target_embedding
#         target_symbols.add_symbol(symbol)
#         target_model.symbol_id_dict = target_symbols.raw()
#         target_hparams.n_symbols += 1
#         target_model.hparams = asdict(target_hparams)

#     return target_model
