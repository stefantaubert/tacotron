# tacotron-cli

[![PyPI](https://img.shields.io/pypi/v/tacotron-cli.svg)](https://pypi.python.org/pypi/tacotron-cli)
[![PyPI](https://img.shields.io/pypi/pyversions/tacotron-cli.svg)](https://pypi.python.org/pypi/tacotron-cli)
[![MIT](https://img.shields.io/github/license/stefantaubert/tacotron.svg)](https://github.com/stefantaubert/tacotron/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/wheel/tacotron-cli.svg)](https://pypi.python.org/pypi/tacotron-cli)
[![PyPI](https://img.shields.io/pypi/implementation/tacotron-cli.svg)](https://pypi.python.org/pypi/tacotron-cli)
[![PyPI](https://img.shields.io/github/commits-since/stefantaubert/tacotron/latest/master.svg)](https://pypi.python.org/pypi/tacotron-cli)

Command-line interface (CLI) to train Tacotron 2 using .wav <=> .TextGrid pairs.

## Features

- train phoneme stress separately
- train single-speaker or multi-speaker
- train/synthesize on CPU or GPU
- synthesis of paragraphs
- copy phoneme embeddings from one checkpoint to another

## Installation

```sh
pip install tacotron-cli --user
```

## Usage

```txt
usage: tacotron-cli [-h] [-v] {train,continue-train,validate,synthesize,analyze,add-missing-symbols} ...

Command-line interface (CLI) to train Tacotron 2 using .wav <=> .TextGrid pairs.

positional arguments:
  {train,continue-train,validate,synthesize,analyze,add-missing-symbols}
                              description
    train                     start training
    continue-train            continue training from a checkpoint
    validate                  validate checkpoint(s)
    synthesize                synthesize lines from a file
    analyze                   analyze checkpoint
    add-missing-symbols       copy missing symbols from one checkpoint to another

optional arguments:
  -h, --help                  show this help message and exit
  -v, --version               show program's version number and exit
```

## Dependencies

- `torch`
- `pandas`
- `numpy`
- `librosa`
- `plotly`
- `matplotlib`
- `scikit-image`
- `scikit-learn`
- `scipy`
- `tqdm`
- `ordered_set>=4.1.0`
- `mel-cepstral-distance>=0.0.1`

## Training

The dataset structure need to follow the generic format of [speech-dataset-parser](https://pypi.org/project/speech-dataset-parser/), i.e., each TextGrid need to contain a tier in which all characters/phonemes/symbols are separated into single intervals, e.g., `T|h|i|s| |i|s| |a| |t|e|x|t|.`.

Tips:

- place stress directly to the vowel of the syllable, e.g. `b|ˈo|d|i` instead of `ˈb|o|d|i`
- normalize the text, e.g., numbers should be written out
- substituted space by either `SIL0`, `SIL1` or `SIL2` depending on how long the duration of the pause should take
  - use `SIL0` for no pause
  - use `SIL1` for a short pause, for example after a comma `...|v|i|ˈɛ|n|ʌ|,|SIL1|ˈɔ|s|t|ɹ|i|ʌ|...`
  - use `SIL2` for a longer pause, for example after a sentence: `...|ˈɝ|θ|.|SIL2`
- Note: only phonemes occurring in the TextGrids (on the selected tier) are possible to synthesize

## Synthesis

To prepare a text for synthesis, following things need to be considered.

- each line in the text file will be synthesized as a single file, therefore it is recommended to place each sentence onto a single line
- paragraphs can be separated by a blank line
- each symbol needs can be separated by an separator like `|`, e.g. `s|ˌɪ|ɡ|ɝ|ˈɛ|t`
  - this is useful if the model contains phonemes/symbols that consist of multiple characters, e.g., `ˈɛ`

Example valid sentence: "As the overlying plate lifts up, it also forms mountain ranges." => `ˈæ|z|SIL0|ð|ʌ|SIL0|ˌoʊ|v|ɝ|l|ˈaɪ|ɪ|ŋ|SIL0|p|l|ˈeɪ|t|SIL0|l|ˈɪ|f|t|s|SIL0|ˈʌ|p|,|SIL1|ɪ|t|SIL0|ˈɔ|l|s|oʊ|SIL0|f|ˈɔ|ɹ|m|z|SIL0|m|ˈaʊ|n|t|ʌ|n|SIL0|ɹ|ˈeɪ|n|d͡ʒ|ʌ|z|.|SIL2`

Example invalid sentence: "Digestion is a vital process which involves the breakdown of food into smaller and smaller components, until they can be absorbed and assimilated into the body." => `daɪˈʤɛsʧʌn ɪz ʌ ˈvaɪtʌl ˈpɹɑˌsɛs wɪʧ ɪnˈvɑlvz ðʌ ˈbɹeɪkˌdaʊn ʌv fud ˈɪntu ˈsmɔlɝ ænd ˈsmɔlɝ kʌmˈpoʊnʌnts, ʌnˈtɪl ðeɪ kæn bi ʌbˈzɔɹbd ænd ʌˈsɪmʌˌleɪtɪd ˈɪntu ðʌ ˈbɑdi.`

## Roadmap

- Outsource method to convert audio files to mel-spectrograms before training
- Better logging
- Provide more pre-trained models
- Add audio examples
- Add printing of statistics of a model, e.g., trained symbols, speakers, accents
- Adding tests

## Pretrained Models

- [LJS-IPA-101500](https://tuc.cloud/index.php/s/xxFCDMgEk8dZKbp): Model trained on LJ Speech dataset with IPA transcriptions for 101500 iterations (= 500 epochs)
  - Symbolset: `! " ' ( ) , - . : ; ? SIL0 SIL1 SIL2 [ ] aɪ aʊ b d d͡ʒ eɪ f h i j k l m n oʊ p s t t͡ʃ u v w z æ ð ŋ ɑ ɔ ɔɪ ɛ ɝ ɡ ɪ ɹ ʃ ʊ ʌ ʒ ˈaɪ ˈaʊ ˈeɪ ˈi ˈoʊ ˈu ˈæ ˈɑ ˈɔ ˈɔɪ ˈɛ ˈɝ ˈɪ ˈʊ ˈʌ ˌaɪ ˌaʊ ˌeɪ ˌi ˌoʊ ˌu ˌæ ˌɑ ˌɔ ˌɔɪ ˌɛ ˌɝ ˌɪ ˌʊ ˌʌ θ`

## Audio Example

"The North Wind and the Sun were disputing which was the stronger, when a traveler came along wrapped in a warm cloak." [Listen here](https://tuc.cloud/index.php/s/gzaYDNKinHw6GCz) (headphones recommended)

## Example Synthesis

To reproduce the audio example from above, you can use the following commands:

```sh
# Create example directory
mkdir ~/example

# Download pre-trained Tacotron model checkpoint
wget https://tuc.cloud/index.php/s/xxFCDMgEk8dZKbp/download/LJS-IPA-101500.pt -O ~/example/checkpoint-tacotron.pt

# Download pre-trained Waveglow model checkpoint
wget https://tuc.cloud/index.php/s/yBRaWz5oHrFwigf/download/LJS-v3-580000.pt -O ~/example/checkpoint-waveglow.pt

# Create text containing phonetic transcription of: "The North Wind and the Sun were disputing which was the stronger, when a traveler came along wrapped in a warm cloak."
cat > ~/example/text.txt << EOF
ð|ʌ|SIL0|n|ˈɔ|ɹ|θ|SIL0|w|ˈɪ|n|d|SIL0|ˈæ|n|d|SIL0|ð|ʌ|SIL0|s|ˈʌ|n|SIL0|w|ɝ|SIL0|d|ɪ|s|p|j|ˈu|t|ɪ|ŋ|SIL0|h|w|ˈɪ|t͡ʃ|SIL0|w|ˈɑ|z|SIL0|ð|ʌ|SIL0|s|t|ɹ|ˈɔ|ŋ|ɝ|,|SIL1|h|w|ˈɛ|n|SIL0|ʌ|SIL0|t|ɹ|ˈæ|v|ʌ|l|ɝ|SIL0|k|ˈeɪ|m|SIL0|ʌ|l|ˈɔ|ŋ|SIL0|ɹ|ˈæ|p|t|SIL0|ɪ|n|SIL0|ʌ|SIL0|w|ˈɔ|ɹ|m|SIL0|k|l|ˈoʊ|k|.|SIL2
EOF

# Synthesize text to mel-spectrogram
tacotron-cli synthesize \
  ~/example/checkpoint-tacotron.pt \
  ~/example/text.txt \
  --sep "|"

# Install waveglow-cli for synthesis of mel-spectrograms
pip install waveglow-cli --user

# Synthesize mel-spectrogram to wav
waveglow-cli synthesize \
  ~/example/checkpoint-waveglow.pt \
  ~/example/text -o

# Resulting wav is written to: ~/example/text/1-1.npy.wav
```

## License

MIT License

## Acknowledgments

Model code adapted from [Nvidia](https://github.com/NVIDIA/tacotron2).

Papers:

- [Tacotron: Towards End-to-End Speech Synthesis](https://www.isca-speech.org/archive/interspeech_2017/wang17n_interspeech.html)
- [Natural TTS Synthesis by Conditioning Wavenet on MEL Spectrogram Predictions](https://ieeexplore.ieee.org/document/8461368)

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 416228727 – CRC 1410

## Citation

If you want to cite this repo, you can use this BibTeX-entry:

```bibtex
@misc{tst22,
  author = {Taubert, Stefan},
  title = {tacotron-cli},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/stefantaubert/tacotron}}
}
```
