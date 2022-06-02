from logging import getLogger
from math import ceil
from typing import Optional, Tuple

import torch
from torch import (FloatTensor, IntTensor, LongTensor, Tensor,  # pylint: disable=no-name-in-module
                   nn)
from torch.autograd import Variable
from torch.nn import functional as F

from tacotron.hparams import HParams
from tacotron.layers import ConvNorm, LinearNorm
from tacotron.utils import (get_mask_from_lengths, get_uniform_weights, get_xavier_weights,
                            weights_to_embedding)

SYMBOL_EMBEDDING_LAYER_NAME = "symbol_embeddings.weight"
SPEAKER_EMBEDDING_LAYER_NAME = "speakers_embeddings.weight"


class LocationLayer(nn.Module):
  def __init__(self, hparams: HParams):
    super().__init__()
    self.location_conv = ConvNorm(
        in_channels=2,
        out_channels=hparams.attention_location_n_filters,
        kernel_size=hparams.attention_location_kernel_size,
        padding=int((hparams.attention_location_kernel_size - 1) / 2),
        bias=False,
        stride=1,
        dilation=1
    )

    self.location_dense = LinearNorm(
        in_dim=hparams.attention_location_n_filters,
        out_dim=hparams.attention_dim,
        bias=False,
        w_init_gain='tanh'
    )

  def forward(self, attention_weights_cat):
    processed_attention = self.location_conv(attention_weights_cat)
    processed_attention = processed_attention.transpose(1, 2)
    processed_attention = self.location_dense(processed_attention)
    return processed_attention


class Attention(nn.Module):
  def __init__(self, hparams: HParams, dims: int):
    super().__init__()
    self.query_layer = LinearNorm(
        in_dim=hparams.attention_rnn_dim,
        out_dim=hparams.attention_dim,
        bias=False,
        w_init_gain='tanh'
    )

    self.memory_layer = LinearNorm(
        in_dim=dims,
        out_dim=hparams.attention_dim,
        bias=False,
        w_init_gain='tanh'
    )

    self.v = LinearNorm(
        in_dim=hparams.attention_dim,
        out_dim=1,
        bias=False
    )

    self.location_layer = LocationLayer(hparams)
    self.score_mask_value = -float("inf")

  def get_alignment_energies(self, query, processed_memory,
                             attention_weights_cat):
    """
    PARAMS
    ------
    query: decoder output (batch, n_mel_channels * n_frames_per_step)
    processed_memory: processed encoder outputs (B, T_in, attention_dim)
    attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

    RETURNS
    -------
    alignment (batch, max_time)
    """

    processed_query = self.query_layer(query.unsqueeze(1))
    processed_attention_weights = self.location_layer(
        attention_weights_cat)
    energies = self.v(torch.tanh(processed_query +
                      processed_attention_weights + processed_memory))

    energies = energies.squeeze(-1)
    return energies

  def forward(self, attention_hidden_state, memory, processed_memory,
              attention_weights_cat, mask):
    """
    PARAMS
    ------
    attention_hidden_state: attention rnn last output
    memory: encoder outputs + speaker embeddings
    processed_memory: processed encoder outputs
    attention_weights_cat: previous and cummulative attention weights
    mask: binary mask for padded data
    """
    alignment = self.get_alignment_energies(
        attention_hidden_state, processed_memory, attention_weights_cat)

    if mask is not None:
      alignment.data.masked_fill_(mask, self.score_mask_value)

    attention_weights = F.softmax(alignment, dim=1)
    attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
    attention_context = attention_context.squeeze(1)

    return attention_context, attention_weights


class Prenet(nn.Module):
  def __init__(self, hparams: HParams):
    super().__init__()
    self.layers = nn.ModuleList([
        LinearNorm(
            in_dim=hparams.n_mel_channels * hparams.n_frames_per_step,
            out_dim=hparams.prenet_dim,
            bias=False
        ),
        LinearNorm(
            in_dim=hparams.prenet_dim,
            out_dim=hparams.prenet_dim,
            bias=False
        ),
    ])

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
      x = F.relu(x)
      x = F.dropout(x, p=0.5, training=True)
    return x


class Postnet(nn.Module):
  """Postnet
    - Five 1-d convolution with 512 channels and kernel size 5
  """

  def __init__(self, hparams: HParams):
    super().__init__()
    self.convolutions = nn.ModuleList()

    self.convolutions.append(
        nn.Sequential(
            ConvNorm(
                in_channels=hparams.n_mel_channels,
                out_channels=hparams.postnet_embedding_dim,
                kernel_size=hparams.postnet_kernel_size,
                stride=1,
                padding=int((hparams.postnet_kernel_size - 1) / 2),
                dilation=1,
                w_init_gain='tanh'
            ),
            nn.BatchNorm1d(
                num_features=hparams.postnet_embedding_dim
            )
        )
    )

    for i in range(1, hparams.postnet_n_convolutions - 1):
      self.convolutions.append(
          nn.Sequential(
              ConvNorm(
                  in_channels=hparams.postnet_embedding_dim,
                  out_channels=hparams.postnet_embedding_dim,
                  kernel_size=hparams.postnet_kernel_size,
                  stride=1,
                  padding=int((hparams.postnet_kernel_size - 1) / 2),
                  dilation=1,
                  w_init_gain='tanh'
              ),
              nn.BatchNorm1d(
                  num_features=hparams.postnet_embedding_dim
              )
          )
      )

    self.convolutions.append(
        nn.Sequential(
            ConvNorm(
                in_channels=hparams.postnet_embedding_dim,
                out_channels=hparams.n_mel_channels,
                kernel_size=hparams.postnet_kernel_size,
                stride=1,
                padding=int((hparams.postnet_kernel_size - 1) / 2),
                dilation=1,
                w_init_gain='linear'
            ),
            nn.BatchNorm1d(
                num_features=hparams.n_mel_channels
            )
        )
    )

  def forward(self, x):
    for i in range(len(self.convolutions) - 1):
      x = F.dropout(torch.tanh(
          self.convolutions[i](x)), 0.5, self.training)
    x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

    return x


class Encoder(nn.Module):
  """Encoder module:
    - Three 1-d convolution banks
    - Bidirectional LSTM
  """

  def __init__(self, hparams: HParams, stress_embedding_dim: Optional[int]):
    super().__init__()

    encoder_embedding_dim = hparams.symbols_embedding_dim
    if hparams.use_stress_embedding:
      encoder_embedding_dim += stress_embedding_dim
    convolutions = []
    for _ in range(hparams.encoder_n_convolutions):
      conv_norm = ConvNorm(
          in_channels=encoder_embedding_dim,
          out_channels=encoder_embedding_dim,
          kernel_size=hparams.encoder_kernel_size,
          stride=1,
          padding=int((hparams.encoder_kernel_size - 1) / 2),
          dilation=1,
          w_init_gain='relu'
      )
      batch_norm = nn.BatchNorm1d(encoder_embedding_dim)
      conv_layer = nn.Sequential(conv_norm, batch_norm)
      convolutions.append(conv_layer)
    self.convolutions = nn.ModuleList(convolutions)

    self.lstm = nn.LSTM(
        input_size=encoder_embedding_dim,
        hidden_size=ceil(encoder_embedding_dim / 2),
        # hidden_size=128,
        num_layers=1,
        batch_first=True,
        bidirectional=True,
    )

  def forward(self, x, input_lengths):
    for conv in self.convolutions:
      x = F.dropout(F.relu(conv(x)), 0.5, self.training)

    x = x.transpose(1, 2)

    # pytorch tensor are not reversible, hence the conversion
    input_lengths = input_lengths.cpu().numpy()
    x = nn.utils.rnn.pack_padded_sequence(
        x, input_lengths, batch_first=True)

    self.lstm.flatten_parameters()
    outputs, _ = self.lstm(x)
    outputs, _ = nn.utils.rnn.pad_packed_sequence(
        outputs, batch_first=True)

    return outputs

  def inference(self, x):
    for conv in self.convolutions:
      x = F.dropout(F.relu(conv(x)), 0.5, self.training)

    x = x.transpose(1, 2)

    self.lstm.flatten_parameters()
    outputs, _ = self.lstm(x)

    return outputs


class Decoder(nn.Module):
  def __init__(self, hparams: HParams, stress_embedding_dim: Optional[int]):
    super().__init__()
    self.n_mel_channels = hparams.n_mel_channels
    self.n_frames_per_step = hparams.n_frames_per_step
    self.attention_rnn_dim = hparams.attention_rnn_dim
    self.decoder_rnn_dim = hparams.decoder_rnn_dim
    self.gate_threshold = hparams.gate_threshold
    self.p_attention_dropout = hparams.p_attention_dropout
    self.p_decoder_dropout = hparams.p_decoder_dropout

    self.prenet = Prenet(hparams)

    encoder_embedding_dim = hparams.symbols_embedding_dim
    if hparams.use_stress_embedding:
      encoder_embedding_dim += stress_embedding_dim

    lstm_hidden_size = ceil(encoder_embedding_dim / 2)
    lstm_out_dim = lstm_hidden_size * 2

    self.merged_dimensions = lstm_out_dim

    if hparams.use_speaker_embedding:
      self.merged_dimensions += hparams.speakers_embedding_dim

    self.attention_rnn = nn.LSTMCell(
        input_size=hparams.prenet_dim + self.merged_dimensions,
        hidden_size=hparams.attention_rnn_dim
    )

    self.attention_layer = Attention(hparams, self.merged_dimensions)

    # Deep Voice 2: "one site-speciﬁc embedding as the initial decoder GRU hidden state" -> is in Tacotron 2 now a LSTM
    self.decoder_rnn = nn.LSTMCell(
        input_size=hparams.attention_rnn_dim + self.merged_dimensions,
        hidden_size=hparams.decoder_rnn_dim,
        bias=True
    )

    self.linear_projection = LinearNorm(
        in_dim=hparams.decoder_rnn_dim + self.merged_dimensions,
        out_dim=hparams.n_mel_channels * hparams.n_frames_per_step
    )

    self.gate_layer = LinearNorm(
        in_dim=hparams.decoder_rnn_dim + self.merged_dimensions,
        out_dim=1,
        bias=True,
        w_init_gain='sigmoid'
    )

  def get_go_frame(self, memory):
    """ Gets all zeros frames to use as first decoder input
    PARAMS
    ------
    memory: decoder outputs

    RETURNS
    -------
    decoder_input: all zeros frames
    """
    B = memory.size(0)
    decoder_input = Variable(memory.data.new(
        B, self.n_mel_channels * self.n_frames_per_step).zero_())
    return decoder_input

  def initialize_decoder_states(self, memory, mask):
    """ Initializes attention rnn states, decoder rnn states, attention
    weights, attention cumulative weights, attention context, stores memory
    and stores processed memory
    PARAMS
    ------
    memory: Encoder outputs
    mask: Mask for padded data if training, expects None for inference
    """
    B = memory.size(0)
    MAX_TIME = memory.size(1)

    self.attention_hidden = Variable(
        memory.data.new(B, self.attention_rnn_dim).zero_())
    self.attention_cell = Variable(
        memory.data.new(B, self.attention_rnn_dim).zero_())

    self.decoder_hidden = Variable(
        memory.data.new(B, self.decoder_rnn_dim).zero_())
    self.decoder_cell = Variable(
        memory.data.new(B, self.decoder_rnn_dim).zero_())

    self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
    self.attention_weights_cum = Variable(
        memory.data.new(B, MAX_TIME).zero_())

    self.attention_context = Variable(
        memory.data.new(B, self.merged_dimensions).zero_())

    self.memory = memory
    self.processed_memory = self.attention_layer.memory_layer(memory)
    self.mask = mask

  def parse_decoder_inputs(self, decoder_inputs):
    """ Prepares decoder inputs, i.e. mel outputs
    PARAMS
    ------
    decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

    RETURNS
    -------
    inputs: processed decoder inputs

    """
    # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
    decoder_inputs = decoder_inputs.transpose(1, 2)
    decoder_inputs = decoder_inputs.view(decoder_inputs.size(
        0), int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
    # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
    decoder_inputs = decoder_inputs.transpose(0, 1)
    return decoder_inputs

  def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Prepares decoder outputs for output
    PARAMS
    ------
    mel_outputs:
    gate_outputs: gate output energies
    alignments:

    RETURNS
    -------
    mel_outputs:
    gate_outpust: gate output energies
    alignments:
    """
    # (T_out, B) -> (B, T_out)
    alignments = torch.stack(alignments).transpose(0, 1)
    # (T_out, B) -> (B, T_out)
    gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
    gate_outputs = gate_outputs.contiguous()
    # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
    mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
    # decouple frames per step
    mel_outputs = mel_outputs.view(
        mel_outputs.size(0), -1, self.n_mel_channels)
    # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
    mel_outputs = mel_outputs.transpose(1, 2)

    return mel_outputs, gate_outputs, alignments

  def decode(self, decoder_input):
    """ Decoder step using stored states, attention and memory
    PARAMS
    ------
    decoder_input: previous mel output

    RETURNS
    -------
    mel_output:
    gate_output: gate output energies
    attention_weights:
    """
    cell_input = torch.cat((decoder_input, self.attention_context), -1)
    self.attention_hidden, self.attention_cell = self.attention_rnn(
        cell_input, (self.attention_hidden, self.attention_cell))
    self.attention_hidden = F.dropout(
        self.attention_hidden, self.p_attention_dropout, self.training)

    attention_weights_cat = torch.cat(
        (self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)
    self.attention_context, self.attention_weights = self.attention_layer(
        self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat, self.mask)

    self.attention_weights_cum += self.attention_weights
    decoder_input = torch.cat(
        (self.attention_hidden, self.attention_context), -1)
    self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
        decoder_input, (self.decoder_hidden, self.decoder_cell))
    self.decoder_hidden = F.dropout(
        self.decoder_hidden, self.p_decoder_dropout, self.training)

    decoder_hidden_attention_context = torch.cat(
        (self.decoder_hidden, self.attention_context), dim=1)
    decoder_output = self.linear_projection(
        decoder_hidden_attention_context)

    gate_prediction = self.gate_layer(decoder_hidden_attention_context)
    return decoder_output, gate_prediction, self.attention_weights

  def forward(self, memory, decoder_inputs, memory_lengths) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Decoder forward pass for training
    PARAMS
    ------
    memory: Encoder outputs + speaker embeddings
    decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
    memory_lengths: Encoder output lengths for attention masking.

    RETURNS
    -------
    mel_outputs: mel outputs from the decoder
    gate_outputs: gate outputs from the decoder
    alignments: sequence of attention weights from the decoder
    """
    # get_go_frame -> parse_decoder_inputs -> prenet -> initialize_decoder_states -> decode -> parse_decoder_outputs
    decoder_input = self.get_go_frame(memory)
    # [20, 80] -> [1, 20, 80]
    decoder_input = decoder_input.unsqueeze(0)
    decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
    decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
    decoder_inputs = self.prenet(decoder_inputs)

    self.initialize_decoder_states(
        memory, mask=~get_mask_from_lengths(memory_lengths))

    mel_outputs, gate_outputs, alignments = [], [], []
    while len(mel_outputs) < decoder_inputs.size(0) - 1:
      decoder_input = decoder_inputs[len(mel_outputs)]
      mel_output, gate_output, attention_weights = self.decode(
          decoder_input)
      mel_outputs += [mel_output.squeeze(1)]
      gate_outputs += [gate_output.squeeze(1)]
      alignments += [attention_weights]

    return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

  def inference(self, memory, max_decoder_steps: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], bool]:
    """ Decoder inference
    PARAMS
    ------
    memory: Encoder outputs

    RETURNS
    -------
    mel_outputs: mel outputs from the decoder
    gate_outputs: gate outputs from the decoder
    alignments: sequence of attention weights from the decoder
    """
    decoder_input = self.get_go_frame(memory)

    self.initialize_decoder_states(memory, mask=None)
    reached_max_decoder_steps = False

    mel_outputs, gate_outputs, alignments = [], [], []
    while True:
      decoder_input = self.prenet(decoder_input)
      mel_output, gate_output, alignment = self.decode(decoder_input)

      mel_outputs += [mel_output.squeeze(1)]
      gate_outputs += [gate_output]
      alignments += [alignment]

      if torch.sigmoid(gate_output.data) > self.gate_threshold:
        break

      if max_decoder_steps > 0 and len(mel_outputs) == max_decoder_steps:
        logger = getLogger(__name__)
        logger.warning("Reached max decoder steps.")
        reached_max_decoder_steps = True
        break

      decoder_input = mel_output

    decoder_outputs = self.parse_decoder_outputs(
        mel_outputs, gate_outputs, alignments)
    return decoder_outputs, reached_max_decoder_steps


# def get_speaker_weights(hparams: HParams) -> torch.Tensor:
#   weights = get_xavier_weights(hparams.n_speakers + 1, hparams.speakers_embedding_dim)
#   return weights


# def get_symbol_weights(hparams: HParams) -> torch.Tensor:
#   model_weights = get_uniform_weights(hparams.n_symbols, hparams.symbols_embedding_dim)
#   return model_weights


ForwardXIn = Tuple[IntTensor, IntTensor, FloatTensor,
                   IntTensor, Optional[IntTensor], Optional[LongTensor]]


class Tacotron2(nn.Module):
  def __init__(self, hparams: HParams, n_symbols: int, n_stresses: Optional[int], n_speakers: Optional[int]):
    super().__init__()
    self.use_speaker_embedding = hparams.use_speaker_embedding
    self.use_stress_embedding = hparams.use_stress_embedding

    self.mask_padding = hparams.mask_padding
    self.n_mel_channels = hparams.n_mel_channels

    # +1 because of padding
    symbol_emb_weights = get_uniform_weights(
        n_symbols, hparams.symbols_embedding_dim)
    # rename will destroy all previous trained models
    self.symbol_embeddings = weights_to_embedding(symbol_emb_weights)

    if self.use_speaker_embedding:
      assert n_speakers is not None
      speaker_emb_weights = get_xavier_weights(
          n_speakers, hparams.speakers_embedding_dim)
      self.speakers_embeddings = weights_to_embedding(
          speaker_emb_weights)

    self.stress_embedding_dim = None
    if self.use_stress_embedding:
      assert n_stresses is not None
      self.stress_embedding_dim = n_stresses

    self.encoder = Encoder(hparams, self.stress_embedding_dim)
    self.decoder = Decoder(hparams, self.stress_embedding_dim)
    self.postnet = Postnet(hparams)

  def forward(self, inputs: ForwardXIn) -> Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor]:
    symbols, symbol_lengths, mels, output_lengths, speakers, stresses = inputs
    symbol_lengths, output_lengths = symbol_lengths.data, output_lengths.data

    # symbol_inputs: [70, 174] -> [batch_size, maximum count of symbols]

    # shape: [70, 174, 512] -> [batch_size, maximum count of symbols, symbols_emb_dim]
    embedded_inputs: FloatTensor = self.symbol_embeddings(input=symbols)
    assert embedded_inputs.dtype == torch.float32

    if self.use_stress_embedding:
      assert stresses is not None
      # Note: num_classes need to be defined because otherwise the dimension is not always the same since not all batches contain all stresses
      stress_one_hot_tensor: LongTensor = F.one_hot(
          stresses, num_classes=self.stress_embedding_dim)  # _, -, 0, 1, 2
      stress_one_hot_tensor = stress_one_hot_tensor.type(torch.float32)
      assert not stress_one_hot_tensor.requires_grad
      embedded_inputs = torch.cat(
          (embedded_inputs, stress_one_hot_tensor), -1)

    # swap last two dims
    embedded_inputs = embedded_inputs.transpose(1, 2)

    encoder_outputs = self.encoder(
        x=embedded_inputs,
        input_lengths=symbol_lengths
    )

    merged_outputs = encoder_outputs

    if self.use_speaker_embedding:
      assert speakers is not None
      embedded_speakers: FloatTensor = self.speakers_embeddings(
          input=speakers)
      assert embedded_speakers.dtype == torch.float32
      # concatenate symbol and speaker embeddings (-1 means last dimension)
      merged_outputs = torch.cat((merged_outputs, embedded_speakers), -1)

    mel_outputs, gate_outputs, alignments = self.decoder(
        memory=merged_outputs,
        decoder_inputs=mels,
        memory_lengths=symbol_lengths
    )

    mel_outputs_postnet = self.postnet(mel_outputs)
    mel_outputs_postnet = mel_outputs + mel_outputs_postnet

    if self.mask_padding:
      mask = ~get_mask_from_lengths(output_lengths)
      mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
      mask = mask.permute(1, 0, 2)

      mel_outputs.data.masked_fill_(mask, 0.0)
      mel_outputs_postnet.data.masked_fill_(mask, 0.0)
      gate_outputs.data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

    return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

  def inference(self, symbols: IntTensor, stresses: Optional[LongTensor], speakers: Optional[IntTensor], max_decoder_steps: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    embedded_inputs: FloatTensor = self.symbol_embeddings(input=symbols)
    assert embedded_inputs.dtype == torch.float32

    if torch.isnan(embedded_inputs).any():
      # embedding_inputs can be nan if training was not good
      msg = "Symbol embeddings returned nan!"
      logger = getLogger(__name__)
      logger.error(msg)
      raise Exception(msg)

    if self.use_stress_embedding:
      assert stresses is not None
      # Note: num_classes need to be defined because otherwise the dimension is not always the same since not all batches contain all stresses
      stress_embeddings: LongTensor = F.one_hot(
          stresses, num_classes=self.stress_embedding_dim)  # _, -, 0, 1, 2
      stress_embeddings = stress_embeddings.type(torch.float32)
      embedded_inputs = torch.cat(
          (embedded_inputs, stress_embeddings), -1)

    # swap last two dims
    embedded_inputs = embedded_inputs.transpose(1, 2)

    encoder_outputs = self.encoder.inference(embedded_inputs)

    merged_outputs = encoder_outputs

    if self.use_speaker_embedding:
      assert speakers is not None
      embedded_speakers: FloatTensor = self.speakers_embeddings(
          input=speakers)
      assert embedded_speakers.dtype == torch.float32
      # concatenate symbol and speaker embeddings (-1 means last dimension)
      merged_outputs = torch.cat((merged_outputs, embedded_speakers), -1)

    decoder_outputs, reached_max_decoder_steps = self.decoder.inference(
        merged_outputs, max_decoder_steps)
    mel_outputs, gate_outputs, alignments = decoder_outputs

    mel_outputs_postnet = self.postnet(mel_outputs)
    mel_outputs_postnet = mel_outputs + mel_outputs_postnet

    return mel_outputs, mel_outputs_postnet, gate_outputs, alignments, reached_max_decoder_steps
