import logging
import os
import random
from dataclasses import asdict, dataclass
from logging import Logger
from math import floor, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import matplotlib.ticker as ticker
import numpy as np
import torch
from general_utils import (disable_imageio_logger, disable_matplot_logger,
                           disable_numba_logger, get_filenames)
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.spatial.distance import cosine
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_T = TypeVar('_T')
PYTORCH_EXT = ".pt"


formatter = logging.Formatter(
  '[%(asctime)s.%(msecs)03d] (%(levelname)s) %(message)s',
  datefmt='%Y/%m/%d %H:%M:%S'
)


def get_default_logger():
  return logging.getLogger("default")


def prepare_logger(log_file_path: Optional[str] = None, reset: bool = False, logger: Logger = get_default_logger()) -> None:
  init_logger(logger)
  add_console_out_to_logger(logger)
  if log_file_path is not None:
    if reset:
      reset_file_log(log_file_path)
    add_file_out_to_logger(logger, log_file_path)
  return logger


def init_logger(logger: logging.Logger = get_default_logger()) -> None:
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
  disable_imageio_logger()

  return logger


def add_console_out_to_logger(logger: logging.Logger = get_default_logger()) -> None:
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.NOTSET)
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)
  logger.debug("init console logger")


def add_file_out_to_logger(logger: logging.Logger = get_default_logger(), log_file_path: str = "/tmp/log.txt") -> None:
  fh = logging.FileHandler(log_file_path)
  fh.setLevel(logging.INFO)
  fh.setFormatter(formatter)
  logger.addHandler(fh)
  logger.debug(f"init logger to {log_file_path}")


def reset_file_log(log_file_path: str) -> None:
  if log_file_path.is_file():
    os.remove(log_file_path)


def plot_alignment_np(alignment: np.ndarray) -> np.ndarray:
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


def plot_alignment_np_new(alignment: np.ndarray, mel_dim_x=16, mel_dim_y=5, factor=1, title: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
  alignment = alignment.T

  height, width = alignment.shape
  width_factor = width / 1000
  fig, axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(mel_dim_x * factor * width_factor, mel_dim_y * factor),
  )

  img = axes.imshow(
    X=alignment,
    aspect='auto',
    origin='lower',
    interpolation='none'
  )

  axes.set_yticks(np.arange(0, height, step=5))
  axes.set_xticks(np.arange(0, width, step=50))
  axes.xaxis.set_major_locator(ticker.NullLocator())
  axes.yaxis.set_major_locator(ticker.NullLocator())
  plt.tight_layout()  # font logging occurs here
  figa_core = figure_to_numpy_rgb(fig)

  fig.colorbar(img, ax=axes)
  axes.xaxis.set_major_locator(ticker.AutoLocator())
  axes.yaxis.set_major_locator(ticker.AutoLocator())

  if title is not None:
    axes.set_title(title)
  axes.set_xlabel("Encoder timestep")
  axes.set_ylabel("Decoder timestep")
  plt.tight_layout()  # font logging occurs here
  figa_labeled = figure_to_numpy_rgb(fig)
  plt.close()

  return figa_core, figa_labeled


def get_last_checkpoint(checkpoint_dir: Path) -> Tuple[str, int]:
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
  checkpoint_path = checkpoint_dir / last_checkpoint
  return checkpoint_path, last_iteration


def get_all_checkpoint_iterations(checkpoint_dir: Path) -> List[int]:
  filenames = get_filenames(checkpoint_dir)
  checkpoints_str = [get_pytorch_basename(str(x))
                     for x in filenames if is_pytorch_file(str(x))]
  checkpoints = list(sorted(list(map(int, checkpoints_str))))
  return checkpoints


def get_checkpoint(checkpoint_dir: Path, iteration: int) -> str:
  checkpoint_path = checkpoint_dir / get_pytorch_filename(iteration)
  if not checkpoint_path.is_file():
    raise Exception(f"Checkpoint with iteration {iteration} not found!")
  return checkpoint_path


def get_custom_or_last_checkpoint(checkpoint_dir: Path, custom_iteration: Optional[int]) -> Tuple[str, int]:
  return (get_checkpoint(checkpoint_dir, custom_iteration), custom_iteration) if custom_iteration is not None else get_last_checkpoint(checkpoint_dir)


def get_mask_from_lengths(lengths) -> None:
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


def copy_state_dict(state_dict: Dict[str, Tensor], to_model: nn.Module, ignore: List[str]) -> None:
  # TODO: ignore as set
  model_dict = {k: v for k, v in state_dict.items() if k not in ignore}
  update_state_dict(to_model, model_dict)


def update_state_dict(model: nn.Module, updates: Dict[str, Tensor]) -> None:
  dummy_dict = model.state_dict()
  dummy_dict.update(updates)
  model.load_state_dict(dummy_dict)


def log_hparams(hparams: _T, logger: Logger) -> None:
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
  epochs: Optional[int]
  iterations: Optional[int]
  batch_iterations: int
  save_first_iteration: bool
  save_last_iteration: bool
  iters_per_checkpoint: int
  epochs_per_checkpoint: int


def check_save_it(epoch: int, iteration: int, settings: SaveIterationSettings) -> bool:
  if check_is_first(iteration) and settings.save_first_iteration:
    return True

  if settings.epochs is not None and check_is_last_epoch_based(iteration, settings.epochs, settings.batch_iterations) and settings.save_last_iteration:
    return True

  if settings.iterations is not None and check_is_last_iterations_based(iteration, settings.iterations) and settings.save_last_iteration:
    return True

  if check_is_save_iteration(iteration, settings.iters_per_checkpoint):
    return True

  is_last_batch_iteration = check_is_last_batch_iteration(iteration, settings.batch_iterations)
  if is_last_batch_iteration and check_is_save_epoch(epoch, settings.epochs_per_checkpoint):
    return True

  return False


def get_next_save_it(iteration: int, settings: SaveIterationSettings) -> Optional[int]:
  current_iteration = iteration
  last_iteration = get_last_iteration(
    settings.epochs, settings.batch_iterations, settings.iterations)
  while current_iteration <= last_iteration:
    epoch = iteration_to_epoch(current_iteration, settings.batch_iterations)
    if check_save_it(epoch, current_iteration, settings):
      return current_iteration
    current_iteration += 1
  return None


def get_last_iteration(epochs: Optional[int], batch_iterations: Optional[int], iterations: Optional[int]) -> int:
  if epochs is not None:
    return epochs * batch_iterations

  assert iterations is not None
  return iterations


def check_is_first(iteration: int) -> bool:
  assert iteration >= 0
  # iteration=0 means no training was done yet
  return iteration == 1


def check_is_last_epoch_based(iteration: int, epochs: int, batch_iterations: int) -> bool:
  assert iteration >= 0
  return iteration == epochs * batch_iterations


def check_is_last_iterations_based(iteration: int, iterations: int) -> bool:
  assert iteration >= 0
  return iteration == iterations


def check_is_save_iteration(iteration: int, iters_per_checkpoint: int) -> bool:
  assert iteration >= 0
  save_iterations = iters_per_checkpoint > 0
  return iteration > 0 and save_iterations and iteration % iters_per_checkpoint == 0


def check_is_save_epoch(epoch: int, epochs_per_checkpoint: int) -> bool:
  assert epoch >= 0
  save_epochs = epochs_per_checkpoint > 0
  return save_epochs and ((epoch + 1) % epochs_per_checkpoint == 0)


def check_is_last_batch_iteration(iteration: int, batch_iterations: int) -> None:
  assert iteration >= 0
  assert batch_iterations > 0
  if iteration == 0:
    return False
  batch_iteration = iteration_to_batch_iteration(iteration, batch_iterations)
  is_last_batch_iteration = batch_iteration + 1 == batch_iterations
  return is_last_batch_iteration


def get_continue_epoch(current_iteration: int, batch_iterations: int) -> int:
  return iteration_to_epoch(current_iteration + 1, batch_iterations)


def skip_batch(continue_batch_iteration: int, batch_iteration: int) -> None:
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
  random.seed(seed)
  np.random.seed(seed)
  torch.random.manual_seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  # only on multi GPU
  # torch.cuda.manual_seed_all(seed)


def init_cuddn(enabled: bool) -> None:
  torch.backends.cudnn.enabled = enabled


def init_cuddn_benchmark(enabled: bool) -> None:
  torch.backends.cudnn.benchmark = enabled


def get_pytorch_filename(name: Union[str, int]) -> str:
  return f"{name}{PYTORCH_EXT}"


def get_pytorch_basename(filename: str) -> None:
  return filename[:-len(PYTORCH_EXT)]


def is_pytorch_file(filename: str) -> None:
  return filename.endswith(PYTORCH_EXT)


def to_gpu(x) -> torch.autograd.Variable:
  x = x.contiguous()

  if torch.cuda.is_available():
    x = x.cuda(non_blocking=True)
  return torch.autograd.Variable(x)


def figure_to_numpy_rgb(figure: Figure) -> np.ndarray:
  figure.canvas.draw()
  data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
  return data


def cosine_dist_mels(a: np.ndarray, b: np.ndarray) -> float:
  assert a.shape == b.shape
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
