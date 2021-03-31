import dataclasses
import json
import logging
import math
import os
import random
import tarfile
import unicodedata
from collections import Counter
from dataclasses import asdict, astuple, dataclass, replace
from logging import Logger
from math import floor, sqrt
from pathlib import Path
from typing import (Any, Dict, Generic, List, Optional, Set, Tuple, Type,
                    TypeVar, Union)

import numpy as np
import pandas as pd
import torch
import wget
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split
from speech_dataset_preprocessing.globals import DEFAULT_CSV_SEPERATOR
from torch import Tensor, nn
from torch.optim.optimizer import \
    Optimizer  # pylint: disable=no-name-in-module
from torch.utils.data import DataLoader
from tqdm import tqdm

_T = TypeVar('_T')
PYTORCH_EXT = ".pt"


def disable_matplot_logger():
  disable_matplot_font_logger()
  disable_matplot_colorbar_logger()


def disable_numba_logger():
  disable_numba_core_logger()


formatter = logging.Formatter(
  '[%(asctime)s.%(msecs)03d] (%(levelname)s) %(message)s',
  datefmt='%Y/%m/%d %H:%M:%S'
)


def get_default_logger():
  return logging.getLogger("default")


def prepare_logger(log_file_path: Optional[str] = None, reset: bool = False, logger: Logger = get_default_logger()):
  init_logger(logger)
  add_console_out_to_logger(logger)
  if log_file_path is not None:
    if reset:
      reset_file_log(log_file_path)
    add_file_out_to_logger(logger, log_file_path)
  return logger


def init_logger(logger: logging.Logger = get_default_logger()):
  root_logger = logging.getLogger()
  root_logger.setLevel(logging.DEBUG)
  # disable is required (don't know why) because otherwise DEBUG messages would be ignored!
  logger.manager.disable = logging.NOTSET

  # to disable double logging
  logger.propagate = False

  # take it from the above logger (root)
  logger.setLevel(logging.DEBUG)

  for h in logger.handlers:
    logger.removeHandler(h)

  disable_matplot_logger()
  disable_numba_logger()

  return logger


def add_console_out_to_logger(logger: logging.Logger = get_default_logger()):
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.NOTSET)
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)
  logger.debug("init console logger")


def add_file_out_to_logger(logger: logging.Logger = get_default_logger(), log_file_path: str = "/tmp/log.txt"):
  fh = logging.FileHandler(log_file_path)
  fh.setLevel(logging.INFO)
  fh.setFormatter(formatter)
  logger.addHandler(fh)
  logger.debug(f"init logger to {log_file_path}")


def reset_file_log(log_file_path: str):
  if os.path.isfile(log_file_path):
    os.remove(log_file_path)


if __name__ == "__main__":
  test_logger = logging.getLogger("test")

  add_console_out_to_logger(test_logger)


def plot_alignment_np(alignment) -> np.ndarray:
  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment, aspect='auto', origin='lower',
                 interpolation='none')
  fig.colorbar(im, ax=ax)
  ax.set_xlabel("Decoder timestep")
  ax.set_ylabel("Encoder timestep")

  plt.tight_layout()  # font logging occurs here
  plot_np = figure_to_numpy_rgb(fig)
  plt.close()
  return plot_np


def get_last_checkpoint(checkpoint_dir: str) -> Tuple[str, int]:
  '''
  Returns the full path of the last checkpoint and its iteration.
  '''
  # checkpoint_dir = get_checkpoint_dir(training_dir_path)
  its = get_all_checkpoint_iterations(checkpoint_dir)
  at_least_one_checkpoint_exists = len(its) > 0
  if not at_least_one_checkpoint_exists:
    raise Exception("No checkpoint iteration found!")
  last_iteration = max(its)
  last_checkpoint = get_pytorch_filename(last_iteration)
  checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
  return checkpoint_path, last_iteration


def get_all_checkpoint_iterations(checkpoint_dir: str) -> List[int]:
  filenames = get_filenames(checkpoint_dir)
  checkpoints_str = [get_pytorch_basename(x)
                     for x in filenames if is_pytorch_file(x)]
  checkpoints = list(sorted(list(map(int, checkpoints_str))))
  return checkpoints


def get_checkpoint(checkpoint_dir: str, iteration: int) -> str:
  checkpoint_path = os.path.join(
    checkpoint_dir, get_pytorch_filename(iteration))
  if not os.path.isfile(checkpoint_path):
    raise Exception(f"Checkpoint with iteration {iteration} not found!")
  return checkpoint_path


def get_custom_or_last_checkpoint(checkpoint_dir: str, custom_iteration: Optional[int]) -> Tuple[str, int]:
  return (get_checkpoint(checkpoint_dir, custom_iteration), custom_iteration) if custom_iteration is not None else get_last_checkpoint(checkpoint_dir)


def get_value_in_type(old_value: _T, new_value: str) -> _T:
  old_type = type(old_value)
  new_value_with_original_type = old_type(new_value)
  return new_value_with_original_type


def get_only_known_params(params: Dict[str, str], hparams: _T) -> Dict[str, str]:
  available_params = asdict(hparams)
  res = {k: v for k, v in params.items() if k in available_params.keys()}
  return res


def get_dataclass_from_dict(params: Dict[str, str], dc: Type[_T]) -> Tuple[_T, Set[str]]:
  field_names = {x.name for x in dataclasses.fields(dc)}
  res = {k: v for k, v in params.items() if k in field_names}
  ignored = {k for k in params.keys() if k not in field_names}
  return dc(**res), ignored


def check_has_unknown_params(params: Dict[str, str], hparams: _T) -> bool:
  available_params = asdict(hparams)
  for custom_hparam in params.keys():
    if custom_hparam not in available_params.keys():
      return True
  return False


def set_types_according_to_dataclass(params: Dict[str, str], hparams: _T) -> None:
  available_params = asdict(hparams)
  for custom_hparam, new_value in params.items():
    assert custom_hparam in available_params.keys()
    hparam_value = available_params[custom_hparam]
    params[custom_hparam] = get_value_in_type(hparam_value, new_value)


def update_learning_rate_optimizer(optimizer: Optimizer, learning_rate: float):
  for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate


def overwrite_custom_hparams(hparams_dc: _T, custom_hparams: Optional[Dict[str, str]]) -> _T:
  if custom_hparams is None:
    return hparams_dc

  # custom_hparams = get_only_known_params(custom_hparams, hparams_dc)
  if check_has_unknown_params(custom_hparams, hparams_dc):
    raise Exception()

  set_types_according_to_dataclass(custom_hparams, hparams_dc)

  result = replace(hparams_dc, **custom_hparams)
  return result


def get_mask_from_lengths(lengths):
  max_len = torch.max(lengths).item()
  ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
  mask = (ids < lengths.unsqueeze(1)).bool()
  return mask


def get_uniform_weights(dimension: int, emb_dim: int) -> Tensor:
  # TODO check cuda is correct here
  weight = torch.zeros(size=(dimension, emb_dim), device="cuda")
  std = sqrt(2.0 / (dimension + emb_dim))
  val = sqrt(3.0) * std  # uniform bounds for std
  nn.init.uniform_(weight, -val, val)
  return weight


def get_xavier_weights(dimension: int, emb_dim: int) -> Tensor:
  weight = torch.zeros(size=(dimension, emb_dim), device="cuda")
  torch.nn.init.xavier_uniform_(weight)
  return weight


def update_weights(emb: nn.Embedding, weights: Tensor) -> None:
  emb.weight = nn.Parameter(weights)


def weights_to_embedding(weights: Tensor) -> nn.Embedding:
  embedding = nn.Embedding(weights.shape[0], weights.shape[1])
  update_weights(embedding, weights)
  return embedding


def copy_state_dict(state_dict: Dict[str, Tensor], to_model: nn.Module, ignore: List[str]):
  # TODO: ignore as set
  model_dict = {k: v for k, v in state_dict.items() if k not in ignore}
  update_state_dict(to_model, model_dict)


def update_state_dict(model: nn.Module, updates: Dict[str, Tensor]):
  dummy_dict = model.state_dict()
  dummy_dict.update(updates)
  model.load_state_dict(dummy_dict)


def log_hparams(hparams: _T, logger: Logger):
  logger.info("=== HParams ===")
  for param, val in asdict(hparams).items():
    logger.info(f"- {param} = {val}")
  logger.info("===============")


def get_formatted_current_total(current: int, total: int) -> str:
  return f"{str(current).zfill(len(str(total)))}/{total}"


def validate_model(model: nn.Module, criterion: nn.Module, val_loader: DataLoader, batch_parse_method) -> Tuple[float, Tuple[float, nn.Module, Tuple, Tuple]]:
  model.eval()
  res = []
  with torch.no_grad():
    total_val_loss = 0.0
    # val_loader count is: ceil(validation set length / batch size)
    for batch in tqdm(val_loader):
      x, y = batch_parse_method(batch)
      y_pred = model(x)
      loss = criterion(y_pred, y)
      # if distributed_run:
      #   reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
      # else:
      #  reduced_val_loss = loss.item()
      reduced_val_loss = loss.item()
      res.append((reduced_val_loss, model, y, y_pred))
      total_val_loss += reduced_val_loss
    avg_val_loss = total_val_loss / len(val_loader)
  model.train()

  return avg_val_loss, res


@dataclass
class SaveIterationSettings():
  epochs: int
  batch_iterations: int
  save_first_iteration: bool
  save_last_iteration: bool
  iters_per_checkpoint: int
  epochs_per_checkpoint: int


def check_save_it(epoch: int, iteration: int, settings: SaveIterationSettings) -> bool:
  if check_is_first(iteration) and settings.save_first_iteration:
    return True

  if check_is_last(iteration, settings.epochs, settings.batch_iterations) and settings.save_last_iteration:
    return True

  if check_is_save_iteration(iteration, settings.iters_per_checkpoint):
    return True

  is_last_batch_iteration = check_is_last_batch_iteration(iteration, settings.batch_iterations)
  if is_last_batch_iteration and check_is_save_epoch(epoch, settings.epochs_per_checkpoint):
    return True

  return False


def get_next_save_it(iteration: int, settings: SaveIterationSettings) -> Optional[int]:
  result = iteration
  while result <= settings.epochs * settings.batch_iterations:
    epoch = iteration_to_epoch(result, settings.batch_iterations)
    if check_save_it(epoch, result, settings):
      return result
    result += 1
  return None


def check_is_first(iteration: int) -> bool:
  assert iteration >= 0
  # iteration=0 means no training was done yet
  return iteration == 1


def check_is_last(iteration: int, epochs: int, batch_iterations: int) -> bool:
  assert iteration >= 0
  return iteration == epochs * batch_iterations


def check_is_save_iteration(iteration: int, iters_per_checkpoint: int) -> bool:
  assert iteration >= 0
  save_iterations = iters_per_checkpoint > 0
  return iteration > 0 and save_iterations and iteration % iters_per_checkpoint == 0


def check_is_save_epoch(epoch: int, epochs_per_checkpoint: int) -> bool:
  assert epoch >= 0

  save_epochs = epochs_per_checkpoint > 0
  return save_epochs and ((epoch + 1) % epochs_per_checkpoint == 0)


def check_is_last_batch_iteration(iteration: int, batch_iterations: int):
  assert iteration >= 0
  assert batch_iterations > 0
  if iteration == 0:
    return False
  batch_iteration = iteration_to_batch_iteration(iteration, batch_iterations)
  is_last_batch_iteration = batch_iteration + 1 == batch_iterations
  return is_last_batch_iteration


def get_continue_epoch(current_iteration: int, batch_iterations: int) -> int:
  return iteration_to_epoch(current_iteration + 1, batch_iterations)


def skip_batch(continue_batch_iteration: int, batch_iteration: int):
  result = batch_iteration < continue_batch_iteration
  return result


def iteration_to_epoch(iteration: int, batch_iterations: int) -> int:
  """result: [0, inf)"""
  # Iteration 0 has no epoch.
  assert iteration > 0

  iteration_zero_based = iteration - 1
  epoch = floor(iteration_zero_based / batch_iterations)
  return epoch


def iteration_to_batch_iteration(iteration: int, batch_iterations: int) -> int:
  """result: [0, iterations)"""
  # Iteration 0 has no batch iteration.
  assert iteration > 0

  iteration_zero_based = iteration - 1
  batch_iteration = iteration_zero_based % batch_iterations
  return batch_iteration


def get_continue_batch_iteration(iteration: int, batch_iterations: int) -> int:
  return iteration_to_batch_iteration(iteration + 1, batch_iterations)


def filter_checkpoints(iterations: List[int], select: Optional[int], min_it: Optional[int], max_it: Optional[int]) -> List[int]:
  if select is None:
    select = 0
  if min_it is None:
    min_it = 0
  if max_it is None:
    max_it = max(iterations)
  process_checkpoints = [checkpoint for checkpoint in iterations if checkpoint %
                         select == 0 and min_it <= checkpoint <= max_it]

  return process_checkpoints


def init_global_seeds(seed: int) -> None:
  # torch.backends.cudnn.deterministic = True
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.random.manual_seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  # only on multi GPU
  # torch.cuda.manual_seed_all(seed)


def init_cuddn(enabled: bool):
  torch.backends.cudnn.enabled = enabled


def init_cuddn_benchmark(enabled: bool):
  torch.backends.cudnn.benchmark = enabled


def get_pytorch_filename(name: Union[str, int]) -> str:
  return f"{name}{PYTORCH_EXT}"


def get_pytorch_basename(filename: str):
  return filename[:-len(PYTORCH_EXT)]


def is_pytorch_file(filename: str):
  return filename.endswith(PYTORCH_EXT)


def split_train_test_val(wholeset: List[_T], test_size: float, validation_size: float, seed: int, shuffle: bool, verbose: bool) -> Tuple[List[_T], List[_T], List[_T]]:
  assert seed >= 0
  assert 0 <= test_size <= 1
  assert 0 <= validation_size <= 1
  assert test_size + validation_size < 1

  trainset, testset, valset = wholeset, [], []

  if validation_size:
    is_ok = assert_fraction_is_big_enough(validation_size, len(trainset), verbose)
    trainset, valset = train_test_split(
      trainset, test_size=validation_size, random_state=seed, shuffle=shuffle)
    if not is_ok:
      check_is_not_empty(trainset)
      check_is_not_empty(valset)
      if verbose:
        print(f"Split was however successfull, trainsize {len(trainset)}, valsize: {len(valset)}.")
  if test_size:
    adj_test_size = test_size / (1 - validation_size)
    is_ok = assert_fraction_is_big_enough(adj_test_size, len(trainset), verbose)
    trainset, testset = train_test_split(
      trainset, test_size=adj_test_size, random_state=seed, shuffle=shuffle)
    if not is_ok:
      check_is_not_empty(trainset)
      check_is_not_empty(valset)
      if verbose:
        print(
          f"Split was however successfull, trainsize {len(trainset)}, testsize: {len(testset)}.")

  return trainset, testset, valset


def check_is_not_empty(dataset: List[_T]) -> None:
  if len(dataset) == 0:
    raise Exception("Aborting splitting, as a size of 0 resulted.")


def assert_fraction_is_big_enough(fraction: float, size: int, verbose: bool) -> bool:
  """tests that the fraction is bigger than the smallest fraction possible with that size to get at least one example in splitting"""
  calculation_inaccuracy = 10e-5
  min_frac = 1 / size
  y = min(fraction, 1 - fraction)
  if y + calculation_inaccuracy < min_frac:
    if verbose:
      print(f"Warn: Split-fraction {fraction} is to small, it should be >= {min_frac}.")
    return False
  return True


def parse_tuple_list(tuple_list: Optional[str] = None) -> Optional[List[Tuple]]:
  """ tuple_list: "a,b;c,d;... """
  if tuple_list is None:
    return None

  step1: List[str] = tuple_list.split(';')
  result: List[Tuple] = [tuple(x.split(',')) for x in step1]
  result = list(sorted(set(result)))
  return result


def split_hparams_string(hparams: Optional[str]) -> Optional[Dict[str, str]]:
  if hparams is None:
    return None

  assignments = hparams.split(",")
  result = dict([x.split("=") for x in assignments])
  return result


def to_gpu(x) -> torch.autograd.Variable:
  x = x.contiguous()

  if torch.cuda.is_available():
    x = x.cuda(non_blocking=True)
  return torch.autograd.Variable(x)


def split_int_set_str(ints: Optional[str]) -> Optional[Set[int]]:
  """ tuple_list: "1,2,4" """
  if ints is None:
    return None
  if len(ints) == 0:
    return set()
  ints_list = ints.split(",")
  ints_set = set(map(int, ints_list))
  return ints_set


def get_chunk_name(i, chunksize, maximum) -> str:
  assert i >= 0
  assert chunksize > 0
  assert maximum >= 0
  start = i // chunksize
  start *= chunksize
  end = start + chunksize - 1
  if end > maximum:
    end = maximum
  res = f"{start}-{end}"
  return res


def have_common_entries(l: Union[Tuple[_T], List[_T]], s: Union[Tuple[_T], List[_T]]) -> bool:
  res = len(set(l).union(set(s))) > 0
  return res


def contains_only_allowed_symbols(l: Union[Tuple[_T], List[_T]], allowed: Union[Tuple[_T], List[_T]]) -> bool:
  res = len(set(l).difference(set(allowed))) == 0
  return res


def disable_numba_core_logger():
  """
  Disables:
    DEBUG:numba.core.ssa:on stmt: $92load_attr.32 = getattr(value=y, attr=shape)
    DEBUG:numba.core.ssa:on stmt: $const94.33 = const(int, 1)
    DEBUG:numba.core.ssa:on stmt: $96binary_subscr.34 = static_getitem(value=$92load_attr.32, index=1, index_var=$const94.33, fn=<built-in function getitem>)
    DEBUG:numba.core.ssa:on stmt: n_channels = $96binary_subscr.34
    DEBUG:numba.core.ssa:on stmt: $100load_global.35 = global(range: <class 'range'>)
    DEBUG:numba.core.ssa:on stmt: $104call_function.37 = call $100load_global.35(n_out, func=$100load_global.35, args=[Var(n_out, interpn.py:24)], kws=(), vararg=None)
    DEBUG:numba.core.ssa:on stmt: $106get_iter.38 = getiter(value=$104call_function.37)
    DEBUG:numba.core.ssa:on stmt: $phi108.0 = $106get_iter.38
    DEBUG:numba.core.ssa:on stmt: jump 108
    DEBUG:numba.core.byteflow:block_infos State(pc_initial=446 nstack_initial=1):
    AdaptBlockInfo(insts=((446, {'res': '$time_register446.1'}), (448, {'res': '$time_increment448.2'}), (450, {'lhs': '$time_register446.1', 'rhs': '$time_increment448.2', 'res': '$450inplace_add.3'}), (452, {'value': '$450inplace_add.3'}),
    (454, {})), outgoing_phis={}, blockstack=(), active_try_block=None, outgoing_edgepushed={108: ('$phi446.0',)})
    DEBUG:numba.core.byteflow:block_infos State(pc_initial=456 nstack_initial=0):
    AdaptBlockInfo(insts=((456, {'res': '$const456.0'}), (458, {'retval': '$const456.0', 'castval': '$458return_value.1'})), outgoing_phis={}, blockstack=(), active_try_block=None, outgoing_edgepushed={})
    DEBUG:numba.core.interpreter:label 0:
        x = arg(0, name=x)                       ['x']
        y = arg(1, name=y)                       ['y']
        sample_ratio = arg(2, name=sample_ratio) ['sample_ratio']
    ...
  """
  logging.getLogger('numba.core').disabled = True


def disable_matplot_font_logger():
  '''
  Disables:
    DEBUG:matplotlib.font_manager:findfont: score(<Font 'Noto Sans Oriya UI' (NotoSansOriyaUI-Bold.ttf) normal normal 700 normal>) = 10.335
    DEBUG:matplotlib.font_manager:findfont: score(<Font 'Noto Serif Khmer' (NotoSerifKhmer-Regular.ttf) normal normal 400 normal>) = 10.05
    DEBUG:matplotlib.font_manager:findfont: score(<Font 'Samyak Gujarati' (Samyak-Gujarati.ttf) normal normal 500 normal>) = 10.14
    ...
  '''
  logging.getLogger('matplotlib.font_manager').disabled = True


def disable_matplot_colorbar_logger():
  '''
  Disables:
    DEBUG:matplotlib.colorbar:locator: <matplotlib.colorbar._ColorbarAutoLocator object at 0x7f78f08e6370>
    DEBUG:matplotlib.colorbar:Using auto colorbar locator <matplotlib.colorbar._ColorbarAutoLocator object at 0x7f78f08e6370> on colorbar
    DEBUG:matplotlib.colorbar:Setting pcolormesh
  '''
  logging.getLogger('matplotlib.colorbar').disabled = True


def cast_as(obj, _: _T) -> _T:
  return obj


def pass_lines(method: Any, text: str) -> None:
  lines = text.split("\n")
  pass_lines_list(method, lines)


def pass_lines_list(method: Any, lines: List[str]) -> None:
  for l in lines:
    method(l)


def figure_to_numpy_rgb(figure: Figure) -> np.ndarray:
  figure.canvas.draw()
  data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
  return data


def get_filenames(parent_dir: str) -> List[str]:
  assert os.path.isdir(parent_dir)
  _, _, filenames = next(os.walk(parent_dir))
  filenames.sort()
  return filenames


def get_filepaths(parent_dir: str) -> List[str]:
  names = get_filenames(parent_dir)
  res = [os.path.join(parent_dir, x) for x in names]
  return res


def get_subfolder_names(parent_dir: str) -> List[str]:
  assert os.path.isdir(parent_dir)
  _, subfolder_names, _ = next(os.walk(parent_dir))
  subfolder_names.sort()
  return subfolder_names


def get_subfolders(parent_dir: str) -> List[str]:
  """return full paths"""
  names = get_subfolder_names(parent_dir)
  res = [os.path.join(parent_dir, x) for x in names]
  return res


def console_out_len(text: str):
  res = len([c for c in text if unicodedata.combining(c) == 0])
  return res

# TODO: tests


def make_batches_v_h(arr: List[_T], v: int, h: int) -> List[List[_T]]:
  vertical_merge_count = math.ceil(len(arr) / v)
  # print("v", vertical_merge_count)
  horizontal_merge_count = math.ceil(vertical_merge_count / h)
  # print("h", horizontal_merge_count)

  current = 0
  vertical_batches = []

  for _ in range(vertical_merge_count):
    vertical_batch = arr[current:current + v]
    current += v
    vertical_batches.append(vertical_batch)
  # print(vertical_batches)

  current = 0
  horizontal_batches = []
  for _ in range(horizontal_merge_count):
    horizontal_batch = vertical_batches[current:current + h]
    current += h
    horizontal_batches.append(horizontal_batch)

  return horizontal_batches

# TODO: tests


def make_batches_h_v(arr: List[_T], v: int, h: int) -> List[List[_T]]:
  horizontal_merge_count = math.ceil(len(arr) / h)
  # print("v", vertical_merge_count)
  vertical_merge_count = math.ceil(horizontal_merge_count / v)
  # print("h", horizontal_merge_count)

  current = 0
  horizontal_batches = []
  for _ in range(horizontal_merge_count):
    horizontal_batch = arr[current:current + h]
    current += h
    horizontal_batches.append(horizontal_batch)

  current = 0
  vertical_batches = []

  for _ in range(vertical_merge_count):
    vertical_batch = horizontal_batches[current:current + v]
    current += v
    vertical_batches.append(vertical_batch)
  # print(vertical_batches)

  return vertical_batches


class GenericList(list, Generic[_T]):
  def save(self, file_path: str, header: bool = False):
    data = [astuple(xi) for xi in self.items()]
    dataframe = pd.DataFrame(data)
    header_cols = None
    if header and len(self) > 0:
      first_entry = self.items()[0]
      header_cols = list(first_entry.__dataclass_fields__.keys())
    save_df(dataframe, file_path, header_columns=header_cols)

  @classmethod
  def load(cls, member_class: Type[_T], file_path: str):
    data = try_load_df(file_path)
    data_is_not_empty = data is not None
    if data_is_not_empty:
      data_loaded: List[_T] = [member_class(*xi) for xi in data.values]
      res = cls(data_loaded)
      res.load_init()
    else:
      res = cls()
    return res

  def load_init(self):
    return self

  def items(self, with_tqdm: bool = False) -> List[_T]:
    if with_tqdm:
      return tqdm(self)
    return self

  def get_random_entry(self) -> _T:
    idx = random.choice(range(len(self)))
    return self[idx]


def try_load_df(path: str) -> Optional[pd.DataFrame]:
  try:
    return load_df(path)
  except pd.errors.EmptyDataError:
    return None


def load_df(path: str) -> pd.DataFrame:
  data = pd.read_csv(path, header=None, sep=DEFAULT_CSV_SEPERATOR)
  return data


def save_df(dataframe: pd.DataFrame, path: str, header_columns: Optional[List[str]]):
  dataframe.to_csv(path, header=header_columns, index=None, sep=DEFAULT_CSV_SEPERATOR)


def get_sorted_list_from_set(unsorted_set: Set[_T]) -> List[_T]:
  res: List[_T] = list(sorted(list(unsorted_set)))
  return res


def remove_duplicates_list_orderpreserving(l: List[str]) -> List[str]:
  result = []
  for x in l:
    if x not in result:
      result.append(x)
  assert len(result) == len(set(result))
  return result


def get_counter(l: List[List[_T]]) -> Counter:
  items = []
  for sublist in l:
    items.extend(sublist)
  symbol_counter = Counter(items)
  return symbol_counter


def get_unique_items(of_list: List[Union[List[_T], Set[_T]]]) -> Set[_T]:
  items: Set[_T] = set()
  for sub_entries in of_list:
    items = items.union(set(sub_entries))
  return items


def cosine_dist_mels(a: np.ndarray, b: np.ndarray) -> float:
  a, b = make_same_dim(a, b)
  scores = []
  for channel_nr in range(a.shape[0]):
    channel_a = a[channel_nr]
    channel_b = b[channel_nr]
    score = cosine(channel_a, channel_b)
    if np.isnan(score):
      score = 1
    scores.append(score)
  score = np.mean(scores)
  # scores = cdist(pred_np, orig_np, 'cosine')
  final_score = 1 - score
  return final_score


def make_same_dim(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  dim_a = a.shape[1]
  dim_b = b.shape[1]
  diff = abs(dim_a - dim_b)
  if diff > 0:
    adding_array = np.zeros(shape=(a.shape[0], diff))
    if dim_a < dim_b:
      a = np.concatenate((a, adding_array), axis=1)
    else:
      b = np.concatenate((b, adding_array), axis=1)
  assert a.shape == b.shape
  return a, b


def get_basename(filepath: str) -> str:
  '''test.wav -> test'''
  basename, _ = os.path.splitext(os.path.basename(filepath))
  return basename


def get_parent_dirname(filepath: str) -> str:
  last_dir_name = Path(filepath).parts[-1]
  return last_dir_name


def create_parent_folder(file: str) -> str:
  path = Path(file)
  os.makedirs(path.parent, exist_ok=True)
  return path.parent


def str_to_int(val: str) -> int:
  '''maps a string to int'''
  mapped = [(i + 1) * ord(x) for i, x in enumerate(val)]
  res = sum(mapped)
  return res


def get_subdir(training_dir_path: str, subdir: str, create: bool = True) -> str:
  result = os.path.join(training_dir_path, subdir)
  if create:
    os.makedirs(result, exist_ok=True)
  return result


def download_tar(download_url, dir_path, tarmode: str = "r:gz") -> None:
  print("Starting download of {}...".format(download_url))
  os.makedirs(dir_path, exist_ok=True)
  dest = wget.download(download_url, dir_path)
  downloaded_file = os.path.join(dir_path, dest)
  print("\nFinished download to {}".format(downloaded_file))
  print("Unpacking...")
  tar = tarfile.open(downloaded_file, tarmode)
  tar.extractall(dir_path)
  tar.close()
  os.remove(downloaded_file)
  print("Done.")


def save_txt(path: str, text: str) -> None:
  with open(path, 'w', encoding='utf-8') as f:
    f.write(text)


def args_to_str(args) -> str:
  res = ""
  for arg, value in sorted(vars(args).items()):
    res += "{}: {}\n".format(arg, value)
  return res


def parse_json(path: str) -> dict:
  assert os.path.isfile(path)
  with open(path, 'r', encoding='utf-8') as f:
    tmp = json.load(f)
  return tmp


def save_json(path: str, mapping_dict: Dict) -> None:
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(mapping_dict, f, ensure_ascii=False, indent=2)


def read_lines(path: str) -> List[str]:
  assert os.path.isfile(path)
  with open(path, "r", encoding='utf-8') as f:
    lines = f.readlines()
  res = [x.strip("\n") for x in lines]
  return res


def read_text(path: str) -> str:
  res = '\n'.join(read_lines(path))
  return res
