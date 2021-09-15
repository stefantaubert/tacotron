import logging
import time
from logging import Logger
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from tacotron.checkpoint import Checkpoint, get_iteration
from tacotron.core.dataloader import (SymbolsMelCollate, parse_batch,
                                      prepare_trainloader, prepare_valloader)
from tacotron.core.hparams import ExperimentHParams, HParams, OptimizerHParams
from tacotron.core.logger import Tacotron2Logger
from tacotron.core.model import (SPEAKER_EMBEDDING_LAYER_NAME,
                                 SYMBOL_EMBEDDING_LAYER_NAME, Tacotron2)
from tacotron.core.model_checkpoint import CheckpointTacotron
from tacotron.core.model_weights import (get_mapped_speaker_weights,
                                         get_mapped_symbol_weights)
from tacotron.globals import DEFAULT_PADDING_SYMBOL
from tacotron.utils import (SaveIterationSettings, check_save_it,
                            copy_state_dict, get_continue_batch_iteration,
                            get_continue_epoch, get_formatted_current_total,
                            get_last_iteration, get_next_save_it, init_cuddn,
                            init_cuddn_benchmark, init_global_seeds,
                            iteration_to_epoch, log_hparams,
                            overwrite_custom_hparams, skip_batch,
                            update_weights, validate_model)
from text_utils import SpeakersDict, SymbolIdDict, SymbolsMap
from torch import nn
from torch.nn import Parameter
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from tts_preparation import PreparedDataList


class Tacotron2Loss(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.mse_criterion = nn.MSELoss()
    self.bce_criterion = nn.BCEWithLogitsLoss()

  def forward(self, y_pred: Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], y: Tuple[torch.FloatTensor, torch.FloatTensor]) -> torch.Tensor:
    mel_target, gate_target = y[0], y[1]
    mel_target.requires_grad = False
    gate_target.requires_grad = False
    gate_target = gate_target.view(-1, 1)

    mel_out, mel_out_postnet, gate_out, _ = y_pred
    gate_out = gate_out.view(-1, 1)

    mel_out_mse = self.mse_criterion(mel_out, mel_target)
    mel_out_post_mse = self.mse_criterion(mel_out_postnet, mel_target)
    gate_loss = self.bce_criterion(gate_out, gate_target)

    return mel_out_mse + mel_out_post_mse + gate_loss


def validate(model: nn.Module, criterion: nn.Module, val_loader: DataLoader, iteration: int, taco_logger: Tacotron2Logger, logger: Logger) -> None:
  logger.debug("Validating...")
  avg_val_loss, res = validate_model(model, criterion, val_loader, parse_batch)
  logger.info(f"Validation loss {iteration}: {avg_val_loss:9f}")

  logger.debug("Logging to tensorboard...")
  log_only_last_validation_batch = True
  if log_only_last_validation_batch:
    taco_logger.log_validation(*res[-1], iteration)
  else:
    for entry in tqdm(res):
      taco_logger.log_validation(*entry, iteration)
  logger.debug("Finished.")

  return avg_val_loss


def train(warm_model: Optional[CheckpointTacotron], custom_hparams: Optional[Dict[str, str]], taco_logger: Tacotron2Logger, symbols: SymbolIdDict, speakers: SpeakersDict, trainset: PreparedDataList, valset: PreparedDataList, save_callback: Callable[[str], None], weights_checkpoint: Optional[CheckpointTacotron], weights_map: Optional[SymbolsMap], map_from_speaker_name: Optional[str], logger: Logger, checkpoint_logger: Logger) -> None:
  logger.info("Starting new model...")
  _train(
    custom_hparams=custom_hparams,
    taco_logger=taco_logger,
    trainset=trainset,
    valset=valset,
    save_callback=save_callback,
    speakers=speakers,
    symbols=symbols,
    weights_checkpoint=weights_checkpoint,
    weights_map=weights_map,
    warm_model=warm_model,
    map_from_speaker_name=map_from_speaker_name,
    checkpoint=None,
    logger=logger,
    checkpoint_logger=checkpoint_logger
  )


def continue_train(checkpoint: CheckpointTacotron, custom_hparams: Optional[Dict[str, str]], taco_logger: Tacotron2Logger, trainset: PreparedDataList, valset: PreparedDataList, save_callback: Callable[[str], None], logger: Logger, checkpoint_logger: Logger) -> None:
  logger.info("Continuing training from checkpoint...")
  _train(
    custom_hparams=custom_hparams,
    taco_logger=taco_logger,
    trainset=trainset,
    valset=valset,
    save_callback=save_callback,
    speakers=checkpoint.get_speakers(),
    symbols=checkpoint.get_symbols(),
    weights_checkpoint=None,
    weights_map=None,
    warm_model=None,
    map_from_speaker_name=None,
    checkpoint=checkpoint,
    logger=logger,
    checkpoint_logger=checkpoint_logger
  )


def init_torch(hparams: ExperimentHParams) -> None:
  init_cuddn(hparams.cudnn_enabled)
  init_cuddn_benchmark(hparams.cudnn_benchmark)


def log_symbol_weights(model: Tacotron2, logger: Logger) -> None:
  logger.info(f"Symbolweights (cuda: {model.symbol_embeddings.weight.is_cuda})")
  logger.info(str(model.state_dict()[SYMBOL_EMBEDDING_LAYER_NAME]))


def _train(custom_hparams: Optional[Dict[str, str]], taco_logger: Tacotron2Logger, trainset: PreparedDataList, valset: PreparedDataList, save_callback: Callable[[str], None], speakers: SpeakersDict, symbols: SymbolIdDict, checkpoint: Optional[CheckpointTacotron], warm_model: Optional[CheckpointTacotron], weights_checkpoint: Optional[CheckpointTacotron], weights_map: SymbolsMap, map_from_speaker_name: Optional[str], logger: Logger, checkpoint_logger: Logger) -> None:
  """Training and validation logging results to tensorboard and stdout
  Params
  ------
  output_directory (string): directory to save checkpoints
  log_directory (string) directory to save tensorboard logs
  checkpoint_path(string): checkpoint path
  n_gpus (int): number of gpus
  rank (int): rank of current gpu
  hparams (object): comma separated list of "name=value" pairs.
  """

  complete_start = time.time()

  if checkpoint is not None:
    hparams = checkpoint.get_hparams(logger)
  else:
    hparams = HParams(
      n_speakers=len(speakers),
      n_symbols=len(symbols)
    )
  # TODO: it should not be recommended to change the batch size on a trained model
  hparams = overwrite_custom_hparams(hparams, custom_hparams)

  assert hparams.n_speakers > 0
  assert hparams.n_symbols > 0

  log_hparams(hparams, logger)
  init_global_seeds(hparams.seed)
  init_torch(hparams)

  model, optimizer, scheduler = load_model_and_optimizer_and_scheduler(
    hparams=hparams,
    checkpoint=checkpoint,
    logger=logger,
  )

  iteration = get_iteration(checkpoint)

  if checkpoint is None:
    if warm_model is not None:
      logger.info("Loading states from pretrained model...")
      warm_start_model(model, warm_model, hparams, logger)

    if weights_checkpoint is not None:
      logger.info("Mapping symbol embeddings...")

      pretrained_symbol_weights = get_mapped_symbol_weights(
        model_symbols=symbols,
        trained_weights=weights_checkpoint.get_symbol_embedding_weights(),
        trained_symbols=weights_checkpoint.get_symbols(),
        custom_mapping=weights_map,
        hparams=hparams,
        logger=logger
      )

      update_weights(model.symbol_embeddings, pretrained_symbol_weights)

      logger.info("Checking if mapping speaker embeddings...")
      weights_checkpoint_hparams = weights_checkpoint.get_hparams(
        logger)
      map_speaker_weights = hparams.use_speaker_embedding and weights_checkpoint_hparams.use_speaker_embedding and map_from_speaker_name is not None
      if map_speaker_weights:
        logger.info("Mapping speaker embeddings...")
        pretrained_speaker_weights = get_mapped_speaker_weights(
          model_speaker_id_dict=speakers,
          trained_weights=weights_checkpoint.get_speaker_embedding_weights(),
          trained_speaker=weights_checkpoint.get_speakers(),
          map_from_speaker_name=map_from_speaker_name,
          hparams=hparams,
          logger=logger,
        )

        update_weights(model.speakers_embeddings, pretrained_speaker_weights)
      logger.info(f"Done. Mapped speaker weights: {map_speaker_weights}")

  log_symbol_weights(model, logger)

  collate_fn = SymbolsMelCollate(
    n_frames_per_step=hparams.n_frames_per_step,
    padding_symbol_id=symbols.get_id(DEFAULT_PADDING_SYMBOL),
  )

  val_loader = prepare_valloader(hparams, collate_fn, valset, logger)
  train_loader = prepare_trainloader(hparams, collate_fn, trainset, logger)

  batch_iterations = len(train_loader)
  enough_traindata = batch_iterations > 0
  if not enough_traindata:
    msg = "Not enough training data!"
    logger.error(msg)
    raise Exception(msg)

  save_it_settings = SaveIterationSettings(
    epochs=hparams.epochs,
    iterations=hparams.iterations,
    batch_iterations=batch_iterations,
    save_first_iteration=hparams.save_first_iteration,
    save_last_iteration=True,
    iters_per_checkpoint=hparams.iters_per_checkpoint,
    epochs_per_checkpoint=hparams.epochs_per_checkpoint
  )

  last_iteration = get_last_iteration(hparams.epochs, batch_iterations, hparams.iterations)
  last_epoch_one_based = iteration_to_epoch(last_iteration, batch_iterations) + 1

  criterion = Tacotron2Loss()
  batch_durations: List[float] = []

  train_start = time.perf_counter()
  start = train_start
  model.train()
  continue_epoch = get_continue_epoch(iteration, batch_iterations)

  for epoch in range(continue_epoch, last_epoch_one_based):
    current_lr = get_lr(optimizer)
    logger.info(f"The learning rate for epoch {epoch + 1} is: {current_lr}")
    # logger.debug("==new epoch==")
    next_batch_iteration = get_continue_batch_iteration(iteration, batch_iterations)
    skip_bar = None
    if next_batch_iteration > 0:
      logger.debug(f"Current batch is {next_batch_iteration} of {batch_iterations}")
      logger.debug("Skipping batches...")
      skip_bar = tqdm(total=next_batch_iteration)
    for batch_iteration, batch in enumerate(train_loader):
      # logger.debug(f"Used batch with fingerprint: {sum(batch[0][0])}")
      need_to_skip_batch = skip_batch(
        batch_iteration=batch_iteration,
        continue_batch_iteration=next_batch_iteration
      )

      if need_to_skip_batch:
        assert skip_bar is not None
        skip_bar.update(1)
        # debug_logger.debug(f"Skipped batch {batch_iteration + 1}/{next_batch_iteration + 1}.")
        continue
      # debug_logger.debug(f"Current batch: {batch[0][0]}")

      # update_learning_rate_optimizer(optimizer, hparams.learning_rate)

      model.zero_grad()
      x, y = parse_batch(batch)
      y_pred = model(x)

      loss = criterion(y_pred, y)
      reduced_loss = loss.item()

      loss.backward()

      grad_norm = clip_grad_norm_(
        parameters=model.parameters(),
        max_norm=hparams.grad_clip_thresh,
        norm_type=2.0,
      )

      optimizer.step()

      iteration += 1

      end = time.perf_counter()
      duration = end - start
      start = end

      batch_durations.append(duration)
      avg_batch_dur = np.mean(batch_durations)
      avg_epoch_dur = avg_batch_dur * batch_iterations
      remaining_its = last_iteration - iteration
      estimated_remaining_duration = avg_batch_dur * remaining_its

      next_it = get_next_save_it(iteration, save_it_settings)
      next_checkpoint_save_time = 0
      if next_it is not None:
        next_checkpoint_save_time = (next_it - iteration) * avg_batch_dur

      logger.info(" | ".join([
        f"Ep: {get_formatted_current_total(epoch + 1, last_epoch_one_based)}",
        f"It.: {get_formatted_current_total(batch_iteration + 1, batch_iterations)}",
        f"Tot. it.: {get_formatted_current_total(iteration, last_iteration)} ({iteration / last_iteration * 100:.2f}%)",
        f"Utts.: {iteration * hparams.batch_size}",
        f"Loss: {reduced_loss:.6f}",
        f"Grad norm: {grad_norm:.6f}",
        # f"Dur.: {duration:.2f}s/it",
        f"Avg. dur.: {avg_batch_dur:.2f}s/it & {avg_epoch_dur / 60:.0f}m/epoch",
        f"Tot. dur.: {(time.perf_counter() - train_start) / 60 / 60:.2f}h/{estimated_remaining_duration / 60 / 60:.0f}h ({estimated_remaining_duration / 60 / 60 / 24:.1f}days)",
        f"Next ckp.: {next_checkpoint_save_time / 60:.0f}m",
      ]))

      taco_logger.log_training(reduced_loss, grad_norm, hparams.learning_rate,
                               duration, iteration)
      was_last_batch_in_epoch = batch_iteration + 1 == len(train_loader)

      if was_last_batch_in_epoch and scheduler is not None:
        # TODO is not on the logical optimal position. should be done after saving and then after loading (but only if saving was done after the last batch iteration)!
        adjust_lr(
          hparams=hparams,
          optimizer=optimizer,
          epoch=epoch,
          scheduler=scheduler,
          logger=logger,
        )

      save_it = check_save_it(epoch, iteration, save_it_settings)
      if save_it:
        checkpoint = CheckpointTacotron.from_instances(
          model=model,
          optimizer=optimizer,
          hparams=hparams,
          iteration=iteration,
          symbols=symbols,
          speakers=speakers,
          scheduler=scheduler,
        )

        save_callback(checkpoint)

        valloss = validate(model, criterion, val_loader, iteration, taco_logger, logger)

        # if rank == 0:
        log_checkpoint_score(
          iteration=iteration,
          gradloss=grad_norm,
          trainloss=reduced_loss,
          valloss=valloss,
          epoch_one_based=epoch + 1,
          batch_it_one_based=batch_iteration + 1,
          batch_size=hparams.batch_size,
          checkpoint_logger=checkpoint_logger
        )

      is_last_it = iteration == last_iteration
      if is_last_it:
        break

  duration_s = time.time() - complete_start
  logger.info(f'Finished training. Total duration: {duration_s / 60:.2f}m')


def adjust_lr(hparams, optimizer, epoch, scheduler, logger) -> None:
  assert hparams.lr_decay_start_after_epoch is not None
  assert hparams.lr_decay_start_after_epoch >= 1
  assert hparams.lr_decay_min is not None
  assert 0 < hparams.lr_decay_min <= hparams.learning_rate

  decrease_lr = epoch + 1 >= hparams.lr_decay_start_after_epoch
  if decrease_lr:
    new_lr_would_be_too_small = scheduler.get_lr()[0] < hparams.lr_decay_min
    if new_lr_would_be_too_small:
      if get_lr(optimizer) != hparams.lr_decay_min:
        set_lr(optimizer, hparams.lr_decay_min)
        logger.info(f"Reached closest value to min_lr {hparams.lr_decay_min}")
    else:
      scheduler.step()

  #logger.info(f"After adj: Epoch: {epoch + 1}, Current LR: {get_lr(optimizer)}, Scheduler next LR would be: {scheduler.get_lr()[0]}")


def get_lr(optimizer: Optimizer) -> float:
  vals = []
  for g in optimizer.param_groups:
    vals.append(g['lr'])
  divergend_lrs = set(vals)
  assert len(divergend_lrs) == 1
  return divergend_lrs.pop()


def set_lr(optimizer: Optimizer, lr: float) -> None:
  for g in optimizer.param_groups:
    g['lr'] = lr


def load_model(hparams: HParams, state_dict: Optional[Dict], logger: logging.Logger) -> None:
  model = Tacotron2(hparams, logger).cuda()
  if state_dict is not None:
    model.load_state_dict(state_dict)

  return model


def load_optimizer(model_parameters: Iterator[Parameter], hparams: OptimizerHParams, state_dict: Optional[Dict]) -> Adam:
  optimizer = Adam(
    params=model_parameters,
    lr=hparams.learning_rate,
    betas=(hparams.beta1, hparams.beta2),
    eps=hparams.eps,
    weight_decay=hparams.weight_decay,
    amsgrad=hparams.amsgrad,
  )

  if state_dict is not None:
    optimizer.load_state_dict(state_dict)

  return optimizer


def load_scheduler(optimizer: Adam, hparams: OptimizerHParams, state_dict: Optional[Dict]) -> ExponentialLR:
  scheduler: ExponentialLR
  if state_dict is not None:
    scheduler = ExponentialLR(
      optimizer=optimizer,
      gamma=state_dict["gamma"],
      last_epoch=state_dict["last_epoch"],
      verbose=state_dict["verbose"],
    )
    scheduler.load_state_dict(state_dict)
  else:
    assert hparams.lr_decay_gamma is not None

    scheduler = ExponentialLR(
      optimizer=optimizer,
      gamma=hparams.lr_decay_gamma,
      last_epoch=-1,
      verbose=True,
    )

  return scheduler


def load_model_and_optimizer_and_scheduler(hparams: HParams, checkpoint: Optional[Checkpoint], logger: Logger) -> Tuple[Tacotron2, Adam, Optional[ExponentialLR]]:
  model = load_model(
    hparams=hparams,
    logger=logger,
    #state_dict=checkpoint.state_dict if checkpoint is not None else None,
    state_dict=None,
  )

  optimizer = load_optimizer(
    model_parameters=model.parameters(),
    hparams=hparams,
    #state_dict=checkpoint.optimizer if checkpoint is not None else None
    state_dict=None,
  )

  scheduler = None

  if hparams.use_exponential_lr_decay:
    scheduler = load_scheduler(
      optimizer=optimizer,
      hparams=hparams,
      #state_dict=checkpoint.scheduler_state_dict if checkpoint is not None else None
        state_dict=None,
    )

  if checkpoint is not None:
    model.load_state_dict(checkpoint.model_state_dict)
    optimizer.load_state_dict(checkpoint.optimizer_state_dict)

    if hparams.use_exponential_lr_decay:
      scheduler.load_state_dict(checkpoint.scheduler_state_dict)

  return model, optimizer, scheduler


def warm_start_model(model: nn.Module, warm_model: CheckpointTacotron, hparams: HParams, logger: Logger) -> None:
  warm_model_hparams = warm_model.get_hparams(logger)
  use_speaker_emb = hparams.use_speaker_embedding and warm_model_hparams.use_speaker_embedding

  speakers_embedding_dim_mismatch = use_speaker_emb and (
    warm_model_hparams.speakers_embedding_dim != hparams.speakers_embedding_dim)

  if speakers_embedding_dim_mismatch:
    msg = "Mismatch in speaker embedding dimensions!"
    logger.exception(msg)
    raise Exception(msg)

  symbols_embedding_dim_mismatch = warm_model_hparams.symbols_embedding_dim != hparams.symbols_embedding_dim
  if symbols_embedding_dim_mismatch:
    msg = "Mismatch in symbol embedding dimensions!"
    logger.exception(msg)
    raise Exception(msg)

  copy_state_dict(
    state_dict=warm_model.model_state_dict,
    to_model=model,
    ignore=hparams.ignore_layers + [
      SYMBOL_EMBEDDING_LAYER_NAME,
      SPEAKER_EMBEDDING_LAYER_NAME
    ]
  )


def log_checkpoint_score(iteration: int, gradloss: float, trainloss: float, valloss: float, epoch_one_based: int, batch_it_one_based: int, batch_size: int, checkpoint_logger: Logger) -> None:
  loss_avg = (trainloss + valloss) / 2
  msg = f"{iteration}\tepoch: {epoch_one_based}\tit-{batch_it_one_based}\tgradloss: {gradloss:.6f}\ttrainloss: {trainloss:.6f}\tvalidationloss: {valloss:.6f}\tavg-train-val: {loss_avg:.6f}\tutterances: {iteration*batch_size}"
  checkpoint_logger.info(msg)
