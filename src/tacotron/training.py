import time
from logging import Logger, getLogger
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from torch import FloatTensor, nn
from torch.backends import cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from tacotron.checkpoint_handling import (CheckpointDict, create, get_hparams, get_iteration,
                                          get_model_state, get_optimizer_state, get_scheduler_state,
                                          get_speaker_embedding_weights, get_speaker_mapping,
                                          get_stress_mapping, get_symbol_embedding_weights,
                                          get_symbol_mapping)
from tacotron.dataloader import (SymbolsMelCollate, create_speaker_mapping,
                                 create_symbol_and_stress_mapping, create_symbol_mapping,
                                 get_speaker_mappings_count, get_stress_mappings_count,
                                 get_symbol_mappings_count, parse_batch, prepare_trainloader,
                                 prepare_valloader)
from tacotron.hparams import ExperimentHParams, HParams, OptimizerHParams
from tacotron.logger import Tacotron2Logger
from tacotron.model import SPEAKER_EMBEDDING_LAYER_NAME, SYMBOL_EMBEDDING_LAYER_NAME, Tacotron2
from tacotron.typing import Entries, SymbolToSymbolMapping
from tacotron.utils import (SaveIterationSettings, check_save_it, copy_state_dict,
                            get_continue_batch_iteration, get_continue_epoch, get_last_iteration,
                            get_next_save_it, get_symbol_printable, init_cuddn,
                            init_cuddn_benchmark, init_global_seeds, iteration_to_epoch,
                            log_hparams, overwrite_custom_hparams, skip_batch,
                            try_copy_tensors_to_device_iterable, try_copy_to)

AVG_COUNT = 30
AVG_COUNT_LONG_TERM = 300


class Tacotron2Loss(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.mse_criterion = nn.MSELoss()
    self.bce_criterion = nn.BCEWithLogitsLoss()

  def forward(self, y_pred: Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor], y: Tuple[FloatTensor, FloatTensor]) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
    mel_target, gate_target = y[0], y[1]
    mel_target.requires_grad = False
    gate_target.requires_grad = False
    gate_target = gate_target.view(-1, 1)

    mel_out, mel_out_postnet, gate_out, alignments = y_pred
    gate_out = gate_out.view(-1, 1)

    mel_out_mse = self.mse_criterion(mel_out, mel_target)
    mel_out_post_mse = self.mse_criterion(mel_out_postnet, mel_target)
    gate_bce = self.bce_criterion(gate_out, gate_target)

    # total_loss = mel_out_mse + mel_out_post_mse + gate_bce

    return mel_out_mse, mel_out_post_mse, gate_bce


def validate(model: nn.Module, criterion: nn.Module, val_loader: DataLoader, iteration: int, device: torch.device, taco_logger: Tacotron2Logger, logger: Logger) -> None:
  logger.debug("Validating...")
  avg_val_loss, res = validate_model(
      model, criterion, val_loader, device, parse_batch)
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


def validate_model(model: nn.Module, criterion: Tacotron2Loss, val_loader: DataLoader, device: torch.device, batch_parse_method) -> Tuple[float, Tuple[float, nn.Module, Tuple, Tuple]]:
  res = []
  logger = getLogger(__name__)
  with torch.no_grad():
    total_val_loss = 0.0
    # val_loader count is: ceil(validation set length / batch size)
    for batch in tqdm(val_loader):
      batch = tuple(try_copy_tensors_to_device_iterable(batch, device))
      x, y = batch_parse_method(batch)
      y_pred = model(x)
      mel_out_mse, mel_out_post_mse, gate_bce = criterion(y_pred, y)
      total_loss = mel_out_mse + mel_out_post_mse + gate_bce
      total_loss_float = total_loss.item()
      logger.debug(f"Mel MSE: {mel_out_mse.item()}")
      logger.debug(f"Mel post MSE: {mel_out_post_mse.item()}")
      logger.debug(f"Gate BCE: {gate_bce.item()}")
      logger.debug(f"Total loss: {total_loss_float}")
      res.append((total_loss_float, model, y, y_pred))
      total_val_loss += total_loss_float
    avg_val_loss = total_val_loss / len(val_loader)

  return avg_val_loss, res


def init_torch(hparams: ExperimentHParams) -> None:
  init_cuddn(hparams.cudnn_enabled)
  init_cuddn_benchmark(hparams.cudnn_benchmark)


def log_symbol_weights(model: Tacotron2, logger: Logger) -> None:
  logger.info(
      f"Symbolweights (cuda: {model.symbol_embeddings.weight.is_cuda})")
  logger.info(str(model.state_dict()[SYMBOL_EMBEDDING_LAYER_NAME]))


def start_training(custom_hparams: Optional[Dict[str, str]], taco_logger: Tacotron2Logger, trainset: Entries, valset: Entries, save_callback: Callable[[CheckpointDict], None], checkpoint: Optional[CheckpointDict], pretrained_model: Optional[CheckpointDict], warm_start: bool, map_symbol_weights: bool, custom_symbol_weights_map: Optional[SymbolToSymbolMapping], map_speaker_weights: bool, map_from_speaker_name: Optional[str], device: torch.device, logger: Logger, checkpoint_logger: Logger) -> None:
  complete_start = time.time()

  if checkpoint is not None:
    hparams = get_hparams(checkpoint)
  else:
    hparams = HParams()
  # TODO: it should not be recommended to change the batch size on a trained model
  hparams = overwrite_custom_hparams(hparams, custom_hparams)

  log_hparams(hparams, logger)
  init_global_seeds(hparams.seed)
  init_torch(hparams)

  n_stresses = None
  stress_mapping = None
  if hparams.use_stress_embedding:
    if checkpoint is None:
      symbol_mapping, stress_mapping = create_symbol_and_stress_mapping(
          valset, trainset, hparams.symbols_are_ipa)
    else:
      symbol_mapping = get_symbol_mapping(checkpoint)
      stress_mapping = get_stress_mapping(checkpoint)
    n_stresses = get_stress_mappings_count(stress_mapping)
  else:
    if checkpoint is None:
      symbol_mapping = create_symbol_mapping(valset, trainset)
    else:
      symbol_mapping = get_symbol_mapping(checkpoint)

  n_symbols = get_symbol_mappings_count(symbol_mapping)

  n_speakers = None
  speaker_mapping = None
  if hparams.use_speaker_embedding:
    if checkpoint is None:
      speaker_mapping = create_speaker_mapping(valset, trainset)
    else:
      speaker_mapping = get_speaker_mapping(checkpoint)
    n_speakers = get_speaker_mappings_count(speaker_mapping)

  logger.info(
      f"Symbols: {' '.join(get_symbol_printable(symbol) for symbol in symbol_mapping.keys())} (#{len(symbol_mapping)}, dim: {hparams.symbols_embedding_dim})")
  if hparams.use_stress_embedding:
    logger.info(
        f"Stresses: {' '.join(stress_mapping.keys())} (#{len(stress_mapping)})")
  else:
    logger.info("Use no stress embedding.")
  if hparams.use_speaker_embedding:
    logger.info(
        f"Speakers: {', '.join(sorted(speaker_mapping.keys()))} (#{len(speaker_mapping)}, dim: {hparams.speakers_embedding_dim})")
  else:
    logger.info("Use no speaker embedding.")

  model = load_model(
      hparams=hparams,
      checkpoint=checkpoint,
      n_symbols=n_symbols,
      n_stresses=n_stresses,
      n_speakers=n_speakers,
  )

  model = cast(Tacotron2, try_copy_to(model, device))
  model = model.train()

  optimizer = load_optimizer(
      model=model,
      hparams=hparams,
      checkpoint=checkpoint
  )

  scheduler = None

  if hparams.use_exponential_lr_decay:
    scheduler = load_scheduler(
        optimizer=optimizer,
        hparams=hparams,
        checkpoint=checkpoint,
    )

  if checkpoint is None:
    iteration = 0
  else:
    iteration = get_iteration(checkpoint)

  if checkpoint is None:
    if warm_start:
      logger.info("Warm starting from pretrained model...")
      if pretrained_model is None:
        logger.error(
            "Warm start: For warm start a pretrained model must be provided!")
        return

      success = warm_start_model(
          model, pretrained_model, hparams, logger)
      if not success:
        return
    else:
      logger.info("Didn't used warm start.")

    if map_symbol_weights:
      logger.info("Mapping symbol weights...")
      if pretrained_model is None:
        logger.error(
            "Mapping symbol weights: For mapping symbol weights a pretrained model must be provided!")
        return

      pre_hparams = get_hparams(pretrained_model)
      if pre_hparams.stress_embedding_dim != hparams.stress_embedding_dim:
        logger.error(
            "Mapping symbol weights: Stress embedding dimensions do not match!")
        return

      pre_stress_mapping = get_stress_mapping(pretrained_model)
      if pre_stress_mapping.keys() != stress_mapping.keys():
        logger.error(
            f"Mapping symbol weights: Stress mappings are not equal! '{' '.join(pre_stress_mapping.keys())}' vs. '{' '.join(stress_mapping.keys())}'")
        return

      pre_symbol_weights = get_symbol_embedding_weights(pretrained_model)
      pre_symbol_mapping = get_symbol_mapping(pretrained_model)

      logger.info(
          f"Symbols in pretrained model: {' '.join(get_symbol_printable(symbol) for symbol in sorted(pre_symbol_mapping.keys()))} (#{len(pre_symbol_mapping)})")
      # map padding
      with torch.no_grad():
        model.symbol_embeddings.weight[0] = pre_symbol_weights[0]
      mapped_symbols = set()
      if custom_symbol_weights_map is not None:
        for to_symbol, from_symbol in custom_symbol_weights_map.items():
          if from_symbol not in pre_symbol_mapping:
            logger.info(
                f"Skipped '{get_symbol_printable(from_symbol)}' -> '{get_symbol_printable(to_symbol)}' because former does not exist in the pretrained model.")
            continue

          if to_symbol not in symbol_mapping:
            logger.info(
                f"Skipped '{get_symbol_printable(from_symbol)}' -> '{get_symbol_printable(to_symbol)}' because latter does not exist in the current model.")
            continue

          from_id = pre_symbol_mapping[from_symbol]
          to_id = symbol_mapping[to_symbol]
          assert from_id > 0 and to_id > 0
          # logger.debug("Current model")
          # logger.debug(model.symbol_embeddings.weight[to_id][:5])
          # logger.debug("Pretrained model")
          # logger.debug(pre_symbol_weights[from_id][:5])
          with torch.no_grad():
            model.symbol_embeddings.weight[to_id] = pre_symbol_weights[from_id]

          # logger.debug(model.symbol_embeddings.weight[to_id][:5])
          # logger.debug("Pretrained model")
          # logger.debug(pre_symbol_weights[from_id][:5])
          logger.info(
              f"Mapped '{get_symbol_printable(from_symbol)}' ({from_id}) to '{get_symbol_printable(to_symbol)}' ({to_id}).")
          mapped_symbols.add(to_symbol)
      else:
        common_symbols = set(pre_symbol_mapping.keys()).intersection(
            symbol_mapping.keys())
        logger.info(
            f"Common symbols that will be mapped: {' '.join(get_symbol_printable(symbol) for symbol in sorted(common_symbols))}")
        for common_symbol in common_symbols:
          from_id = pre_symbol_mapping[common_symbol]
          to_id = symbol_mapping[common_symbol]
          assert from_id > 0 and to_id > 0
          with torch.no_grad():
            model.symbol_embeddings.weight[to_id] = pre_symbol_weights[from_id]
        mapped_symbols = mapped_symbols.union(common_symbols)

      nonmapped_symbols = set(
          symbol_mapping.keys()).difference(mapped_symbols)
      if len(nonmapped_symbols) > 0:
        logger.info(
            f"Mapped symbols: {' '.join(get_symbol_printable(symbol) for symbol in sorted(mapped_symbols))} (#{len(mapped_symbols)})")
        logger.info(
            f"Non-mapped symbols: {' '.join(get_symbol_printable(symbol) for symbol in sorted(nonmapped_symbols))} (#{len(nonmapped_symbols)})")
      else:
        logger.info(f"Mapped all {len(symbol_mapping)} symbols!")
      logger.info("Mapped symbol embeddings.")
    else:
      logger.info("Didn't mapped symbol embeddings.")

    if map_speaker_weights:
      logger.info("Mapping speaker weights...")
      if pretrained_model is None:
        logger.error(
            "Mapping speaker weights: For mapping speaker weights a pretrained model must be provided!")
        return

      if map_from_speaker_name is None:
        logger.error(
            "Mapping speaker weights: A speaker name is required for mapping speaker weights.")
        return

      pre_hparams = get_hparams(pretrained_model)
      both_models_use_speaker_emb = hparams.use_speaker_embedding and pre_hparams.use_speaker_embedding
      if not both_models_use_speaker_emb:
        logger.error(
            "Mapping speaker weights: Couldn't map speaker embeddings because one of the models use no speaker embeddings!")
        return

      pre_speaker_mapping = get_speaker_mapping(pretrained_model)
      if map_from_speaker_name not in pre_speaker_mapping:
        logger.error(
            f"Mapping speaker weights: Speaker '{map_from_speaker_name}' was not found in weights checkpoint.")
        return

      pre_speaker_embedding = get_speaker_embedding_weights(
          pretrained_model)
      # map padding
      with torch.no_grad():
        model.speakers_embeddings.weight[0] = pre_speaker_embedding[0]

      from_id = pre_speaker_mapping[map_from_speaker_name]
      assert from_id > 0
      assert speaker_mapping is not None
      for to_speaker in speaker_mapping.keys():
        to_id = speaker_mapping[to_speaker]
        assert to_id > 0
        with torch.no_grad():
          model.speakers_embeddings.weight[to_id] = pre_speaker_embedding[from_id]
        logger.info(
            f"Mapped '{map_from_speaker_name}' ({from_id}) to '{to_speaker}' ({to_id}).")
      logger.info("Mapped speaker embeddings.")
    else:
      logger.info("Didn't mapped speaker embeddings.")

  # log_symbol_weights(model, logger)

  collate_fn = SymbolsMelCollate(
      n_frames_per_step=hparams.n_frames_per_step,
      use_stress=hparams.use_stress_embedding,
      use_speakers=hparams.use_speaker_embedding,
  )

  val_loader = prepare_valloader(hparams, collate_fn, valset,
                                 symbol_mapping, stress_mapping, speaker_mapping, device, logger)
  train_loader = prepare_trainloader(
      hparams, collate_fn, trainset, symbol_mapping, stress_mapping, speaker_mapping, device, logger)

  batch_iterations = len(train_loader)
  enough_traindata = batch_iterations > 0
  if not enough_traindata:
    msg = "Not enough training data!"
    logger.error(msg)
    return

  save_it_settings = SaveIterationSettings(
      epochs=hparams.epochs,
      iterations=hparams.iterations,
      batch_iterations=batch_iterations,
      save_first_iteration=hparams.save_first_iteration,
      save_last_iteration=True,
      iters_per_checkpoint=hparams.iters_per_checkpoint,
      epochs_per_checkpoint=hparams.epochs_per_checkpoint
  )

  last_iteration = get_last_iteration(
      hparams.epochs, batch_iterations, hparams.iterations)
  last_epoch_one_based = iteration_to_epoch(
      last_iteration, batch_iterations) + 1

  criterion = Tacotron2Loss()
  batch_durations: List[float] = []

  train_start = time.perf_counter()
  start = train_start

  continue_epoch = get_continue_epoch(iteration, batch_iterations)

  mel_mse_losses = []
  mel_post_mse_losses = []
  gate_bce_losses = []
  total_losses = []

  logger.info(f"Use cuDNN: {cudnn.enabled}")

  for epoch in range(continue_epoch, last_epoch_one_based):
    current_lr = get_lr(optimizer)
    logger.info(
        f"The learning rate for epoch {epoch + 1} is: {current_lr}")
    # logger.debug("==new epoch==")
    next_batch_iteration = get_continue_batch_iteration(
        iteration, batch_iterations)
    skip_bar = None
    if next_batch_iteration > 0:
      logger.debug(
          f"Current batch is {next_batch_iteration} of {batch_iterations}")
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
      batch = tuple(try_copy_tensors_to_device_iterable(batch, device))
      x, y = parse_batch(batch)
      y_pred = model(x)

      mel_out_mse, mel_out_post_mse, gate_bce = criterion(y_pred, y)
      total_loss = mel_out_mse + mel_out_post_mse + gate_bce

      mel_mse_losses.append(mel_out_mse.item())
      mel_post_mse_losses.append(mel_out_post_mse.item())
      gate_bce_losses.append(gate_bce.item())
      total_losses.append(total_loss.item())

      total_loss.backward()

      grad_norm = clip_grad_norm_(
          parameters=model.parameters(),
          max_norm=hparams.grad_clip_thresh,
          norm_type=2.0,
      )

      optimizer.step()

      iteration += 1

      end = time.perf_counter()
      batch_durations.append(end - start)
      start = end

      next_it = get_next_save_it(iteration, save_it_settings)
      statistics = {
          "Epoch": epoch + 1,
          "Epochs": last_epoch_one_based,
          "Epoch iteration": batch_iteration + 1,
          "Epoch iterations": batch_iterations,
          "Iteration": iteration,
          "Iterations": last_iteration,
          "Iteration (%)": round(iteration / last_iteration * 100, 2),
          "Seen utterances": iteration * hparams.batch_size,
          "Utterances": last_iteration * hparams.batch_size,
          "Learning rate": current_lr,
          "Mel MSE": mel_mse_losses[-1],
          "Mel MSE AVG": np.mean(mel_mse_losses[-AVG_COUNT:]),
          "Mel MSE long AVG": np.mean(mel_mse_losses[-AVG_COUNT_LONG_TERM:]),
          "Mel postnet MSE": mel_post_mse_losses[-1],
          "Mel postnet MSE AVG": np.mean(mel_post_mse_losses[-AVG_COUNT:]),
          "Mel postnet MSE long AVG": np.mean(mel_post_mse_losses[-AVG_COUNT_LONG_TERM:]),
          "Gate BCE": gate_bce_losses[-1],
          "Gate BCE AVG": np.mean(gate_bce_losses[-AVG_COUNT:]),
          "Gate BCE long AVG": np.mean(gate_bce_losses[-AVG_COUNT_LONG_TERM:]),
          "Total loss": total_losses[-1],
          "Total loss AVG": np.mean(total_losses[-AVG_COUNT:]),
          "Total loss long AVG": np.mean(total_losses[-AVG_COUNT_LONG_TERM:]),
          "Grad norm": grad_norm,
          "Iteration duration (s)": round(batch_durations[-1], 2),
          "Iteration duration AVG (s)": round(np.mean(batch_durations[-AVG_COUNT:]), 2),
          "Epoch duration AVG (min)": round(np.mean(batch_durations[-AVG_COUNT:]) * batch_iterations / 60, 2),
          "Current training duration (h)": round((time.perf_counter() - train_start) / 60 / 60, 2),
          "Estimated remaining duration (h)": round(np.mean(batch_durations[-AVG_COUNT:]) * (last_iteration - iteration) / 60 / 60, 2),
          "Estimated remaining duration (days)": round(np.mean(batch_durations[-AVG_COUNT:]) * (last_iteration - iteration) / 60 / 60 / 24, 2),
          "Estimated duration until next checkpoint (h)": "N/A" if next_it is None else round(np.mean(batch_durations[-AVG_COUNT:]) * (next_it - iteration) / 60 / 60, 2),
      }

      logger.info("---------------------------------------------------")
      logger.info(f"Iteration {iteration}")
      logger.info("---------------------------------------------------")
      for param, val in statistics.items():
        logger.info(f"├─ {param}: {val}")

      # logger.info(" | ".join([
      #   f"Ep: {get_formatted_current_total(epoch + 1, last_epoch_one_based)}",
      #   f"It.: {get_formatted_current_total(batch_iteration + 1, batch_iterations)}",
      #   f"Tot. it.: {get_formatted_current_total(iteration, last_iteration)} ({iteration / last_iteration * 100:.2f}%)",
      #   f"Utts.: {iteration * hparams.batch_size}",
      #   f"Loss: {total_loss_float:.6f}",
      #   f"Avg Loss: {np.mean(total_losses[-20:]):.6f}",
      #   f"Grad norm: {grad_norm:.6f}",
      #   # f"Dur.: {duration:.2f}s/it",
      #   f"Avg. dur.: {avg_batch_dur:.2f}s/it & {avg_epoch_dur / 60:.0f}m/epoch",
      #   f"Tot. dur.: {(time.perf_counter() - train_start) / 60 / 60:.2f}h/{estimated_remaining_duration / 60 / 60:.0f}h ({estimated_remaining_duration / 60 / 60 / 24:.1f}days)",
      #   f"Next ckp.: {next_checkpoint_save_time / 60:.0f}m",
      # ]))

      taco_logger.log_training(total_losses[-1], grad_norm, hparams.learning_rate,
                               batch_durations[-1], iteration)
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
        checkpoint = create(
            model=model,
            optimizer=optimizer,
            hparams=hparams,
            iteration=iteration,
            speaker_mapping=speaker_mapping,
            learning_rate=get_lr(optimizer),
            stress_mapping=stress_mapping,
            symbol_mapping=symbol_mapping,
            scheduler=scheduler,
        )

        save_callback(checkpoint)

        model.eval()
        valloss = validate(model, criterion, val_loader,
                           iteration, device, taco_logger, logger)
        model.train()

        # if rank == 0:
        log_checkpoint_score(
            iteration=iteration,
            gradloss=grad_norm,
            trainloss=total_losses[-1],
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


def adjust_lr(hparams: HParams, optimizer: Optimizer, epoch: int, scheduler, logger: Logger) -> None:
  assert hparams.lr_decay_start_after_epoch is not None
  assert hparams.lr_decay_start_after_epoch >= 1
  assert hparams.lr_decay_min is not None
  assert 0 < hparams.lr_decay_min <= hparams.learning_rate

  decrease_lr = epoch + 1 >= hparams.lr_decay_start_after_epoch
  if decrease_lr:
    new_lr_would_be_too_small = scheduler.get_lr()[
        0] < hparams.lr_decay_min
    if new_lr_would_be_too_small:
      if get_lr(optimizer) != hparams.lr_decay_min:
        set_lr(optimizer, hparams.lr_decay_min)
        logger.info(
            f"Reached closest value to min_lr {hparams.lr_decay_min}")
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


def load_model(hparams: HParams, checkpoint: Optional[CheckpointDict], n_symbols: int, n_stresses: Optional[int], n_speakers: Optional[int]) -> Tacotron2:
  model = Tacotron2(hparams, n_symbols, n_stresses, n_speakers)
  if checkpoint is not None:
    model_state = get_model_state(checkpoint)
    model.load_state_dict(model_state)

  return model


def load_optimizer(model: Tacotron2, hparams: OptimizerHParams, checkpoint: Optional[CheckpointDict]) -> Adam:
  # see: https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783/3
  # for parameter in model.parameters():
  #  assert check_is_on_gpu(parameter)

  optimizer = Adam(
      params=model.parameters(),
      lr=hparams.learning_rate,
      betas=(hparams.beta1, hparams.beta2),
      eps=hparams.eps,
      weight_decay=hparams.weight_decay,
      amsgrad=hparams.amsgrad,
  )

  if checkpoint is not None:
    optimizer_state = get_optimizer_state(checkpoint)
    optimizer.load_state_dict(optimizer_state)

  return optimizer


def load_scheduler(optimizer: Adam, hparams: OptimizerHParams, checkpoint: Optional[CheckpointDict]) -> ExponentialLR:
  assert hparams.lr_decay_gamma is not None
  scheduler: ExponentialLR
  if checkpoint is not None:
    scheduler_state = get_scheduler_state(checkpoint)
    scheduler = ExponentialLR(
        optimizer=optimizer,
        gamma=scheduler_state["gamma"],
        last_epoch=scheduler_state["last_epoch"],
        verbose=scheduler_state["verbose"],
    )
    scheduler.load_state_dict(scheduler_state)
  else:
    scheduler = ExponentialLR(
        optimizer=optimizer,
        gamma=hparams.lr_decay_gamma,
        last_epoch=-1,
        verbose=True,
    )

  return scheduler


def warm_start_model(model: Tacotron2, warm_model: CheckpointDict, hparams: HParams, logger: Logger) -> bool:
  warm_model_hparams = get_hparams(warm_model)

  symbols_embedding_dim_mismatch = warm_model_hparams.symbols_embedding_dim != hparams.symbols_embedding_dim
  if symbols_embedding_dim_mismatch:
    msg = "Warm start: Mismatch in symbol embedding dimensions!"
    logger.error(msg)
    return False

  if hparams.use_stress_embedding and not warm_model_hparams.use_stress_embedding:
    msg = "Warm start: Warm model did not used a stress embedding!"
    logger.error(msg)
    return False

  if hparams.use_speaker_embedding:
    if not warm_model_hparams.use_speaker_embedding:
      msg = "Warm start: Warm model did not used a speaker embedding!"
      logger.error(msg)
      return False

    speakers_embedding_dim_mismatch = warm_model_hparams.speakers_embedding_dim != hparams.speakers_embedding_dim
    if speakers_embedding_dim_mismatch:
      msg = "Warm start: Mismatch in speaker embedding dimensions!"
      logger.error(msg)
      return False

  model_state = get_model_state(warm_model)
  copy_state_dict(
      state_dict=model_state,
      to_model=model,
      ignore=hparams.ignore_layers + [
          SYMBOL_EMBEDDING_LAYER_NAME,
          SPEAKER_EMBEDDING_LAYER_NAME
      ]
  )

  return True


def log_checkpoint_score(iteration: int, gradloss: float, trainloss: float, valloss: float, epoch_one_based: int, batch_it_one_based: int, batch_size: int, checkpoint_logger: Logger) -> None:
  loss_avg = (trainloss + valloss) / 2
  msg = f"{iteration}\tepoch: {epoch_one_based}\tit-{batch_it_one_based}\tgradloss: {gradloss:.6f}\ttrainloss: {trainloss:.6f}\tvalidationloss: {valloss:.6f}\tavg-train-val: {loss_avg:.6f}\tutterances: {iteration*batch_size}"
  checkpoint_logger.info(msg)
