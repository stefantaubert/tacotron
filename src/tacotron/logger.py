import random
from pathlib import Path

import matplotlib.pylab as plt
import torch

from tacotron.utils import figure_to_numpy_rgb

#from torch.utils.tensorboard import SummaryWriter


def plot_alignment_to_numpy(alignment, info=None) -> None:
  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment, aspect='auto', origin='lower',
                 interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
    xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()  # font logging occurs here

  data = figure_to_numpy_rgb(fig)
  plt.close()
  return data


def plot_spectrogram_to_numpy(spectrogram) -> None:
  fig, ax = plt.subplots(figsize=(12, 3))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                 interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  data = figure_to_numpy_rgb(fig)
  plt.close()
  return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs) -> None:
  fig, ax = plt.subplots(figsize=(12, 3))
  ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5,
             color='green', marker='+', s=1, label='target')
  ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
             color='red', marker='.', s=1, label='predicted')

  plt.xlabel("Frames (Green target, Red predicted)")
  plt.ylabel("Gate State")
  plt.tight_layout()

  data = figure_to_numpy_rgb(fig)
  plt.close()
  return data

# class Tacotron2Logger(SummaryWriter):


class Tacotron2Logger():
  def __init__(self, logdir: Path):
    # super().__init__(logdir)
    logdir.mkdir(parents=True, exist_ok=True)

  def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                   iteration):
    return
    self.add_scalar("training.loss", reduced_loss, iteration)
    self.add_scalar("grad.norm", grad_norm, iteration)
    self.add_scalar("learning.rate", learning_rate, iteration)
    self.add_scalar("duration", duration, iteration)

  def log_validation(self, reduced_loss, model, y, y_pred, iteration):
    return
    self.add_scalar("validation.loss", reduced_loss, iteration)
    _, mel_outputs, gate_outputs, alignments = y_pred
    mel_targets, gate_targets = y

    # plot distribution of parameters
    for tag, value in model.named_parameters():
      tag = tag.replace('.', '/')
      # if this fails, then the gradloss is too big, most likely the embeddings return nan
      self.add_histogram(tag, value.data.cpu().numpy(), iteration)

    # plot alignment, mel target and predicted, gate target and predicted
    idx = random.randint(0, alignments.size(0) - 1)
    self.add_image("alignment", plot_alignment_to_numpy(
        alignments[idx].data.cpu().numpy().T), iteration, dataformats='HWC')
    self.add_image("mel_target", plot_spectrogram_to_numpy(
        mel_targets[idx].data.cpu().numpy()), iteration, dataformats='HWC')
    self.add_image("mel_predicted", plot_spectrogram_to_numpy(
        mel_outputs[idx].data.cpu().numpy()), iteration, dataformats='HWC')
    self.add_image("gate", plot_gate_outputs_to_numpy(gate_targets[idx].data.cpu().numpy(
    ), torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()), iteration, dataformats='HWC')
