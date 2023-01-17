# tacotron-cli

[![PyPI](https://img.shields.io/pypi/v/tacotron-cli.svg)](https://pypi.python.org/pypi/tacotron-cli)
[![PyPI](https://img.shields.io/pypi/pyversions/tacotron-cli.svg)](https://pypi.python.org/pypi/tacotron-cli)
[![MIT](https://img.shields.io/github/license/stefantaubert/tacotron.svg)](https://github.com/stefantaubert/tacotron/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/wheel/tacotron-cli.svg)](https://pypi.python.org/pypi/tacotron-cli)
[![PyPI](https://img.shields.io/pypi/implementation/tacotron-cli.svg)](https://pypi.python.org/pypi/tacotron-cli)
[![PyPI](https://img.shields.io/github/commits-since/stefantaubert/tacotron/latest/master.svg)](https://pypi.python.org/pypi/tacotron-cli)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7543638.svg)](https://doi.org/10.5281/zenodo.7543638)

Command-line interface (CLI) to train Tacotron 2 using .wav <=> .TextGrid pairs.

## Features

- train phoneme stress separately (ARPAbet/IPA)
- train phoneme tone separately (IPA)
- train phoneme duration separately (IPA)
- train single/multi-speaker
- train/synthesize on CPU or GPU
- synthesis of paragraphs
- copy embeddings from one checkpoint to another
- train using embeddings or one-hot encodings

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

## Training

The dataset structure need to follow the generic format of [speech-dataset-parser](https://pypi.org/project/speech-dataset-parser/), i.e., each TextGrid need to contain a tier in which all phonemes are separated into single intervals, e.g., `T|h|i|s| |i|s| |a| |t|e|x|t|.`.

Tips:

- place stress directly to the vowel of the syllable, e.g. `b|ˈo|d|i` instead of `ˈb|o|d|i` (body)
- place tone directly to the vowel of the syllable, e.g. `ʈʂʰ|w|a˥˩|n` instead of `ʈʂʰ|w|a|n˥˩` (串)
  - tone-characters which are considered: `˥ ˦ ˧ ˨ ˩`, e.g., `ɑ˥˩`
- duration-characters which are considered: `˘ ˑ ː`, e.g., `ʌː`
- normalize the text, e.g., numbers should be written out
- substituted space by either `SIL0`, `SIL1` or `SIL2` depending on the duration of the pause
  - use `SIL0` for no pause
  - use `SIL1` for a short pause, for example after a comma `...|v|i|ˈɛ|n|ʌ|,|SIL1|ˈɔ|s|t|ɹ|i|ʌ|...`
  - use `SIL2` for a longer pause, for example after a sentence: `...|ˈɝ|θ|.|SIL2`
- Note: only phonemes occurring in the TextGrids (on the selected tier) are possible to synthesize

## Synthesis

To prepare a text for synthesis, following things need to be considered:

- each line in the text file will be synthesized as a single file, therefore it is recommended to place each sentence onto a single line
- paragraphs can be separated by a blank line
- each symbol needs can be separated by an separator like `|`, e.g. `s|ˌɪ|ɡ|ɝ|ˈɛ|t`
  - this is useful if the model contains phonemes/symbols that consist of multiple characters, e.g., `ˈɛ`

Example valid sentence: "As the overlying plate lifts up, it also forms mountain ranges." => `ˈæ|z|SIL0|ð|ʌ|SIL0|ˌoʊ|v|ɝ|l|ˈaɪ|ɪ|ŋ|SIL0|p|l|ˈeɪ|t|SIL0|l|ˈɪ|f|t|s|SIL0|ˈʌ|p|,|SIL1|ɪ|t|SIL0|ˈɔ|l|s|oʊ|SIL0|f|ˈɔ|ɹ|m|z|SIL0|m|ˈaʊ|n|t|ʌ|n|SIL0|ɹ|ˈeɪ|n|d͡ʒ|ʌ|z|.|SIL2`

Example invalid sentence: "Digestion is a vital process which involves the breakdown of food into smaller and smaller components, until they can be absorbed and assimilated into the body." => `daɪˈʤɛsʧʌn ɪz ʌ ˈvaɪtʌl ˈpɹɑˌsɛs wɪʧ ɪnˈvɑlvz ðʌ ˈbɹeɪkˌdaʊn ʌv fud ˈɪntu ˈsmɔlɝ ænd ˈsmɔlɝ kʌmˈpoʊnʌnts, ʌnˈtɪl ðeɪ kæn bi ʌbˈzɔɹbd ænd ʌˈsɪmʌˌleɪtɪd ˈɪntu ðʌ ˈbɑdi.`

## Pretrained Models

- [LJS-IPA-101500](https://tuc.cloud/index.php/s/xxFCDMgEk8dZKbp): Model trained on LJ Speech dataset with IPA transcriptions for 101500 iterations (= 500 epochs) with separated learning of stress
  - Symbolset: `! " ' ( ) , - . : ; ? SIL0 SIL1 SIL2 [ ] aɪ aʊ b d d͡ʒ eɪ f h i j k l m n oʊ p s t t͡ʃ u v w z æ ð ŋ ɑ ɔ ɔɪ ɛ ɝ ɡ ɪ ɹ ʃ ʊ ʌ ʒ ˈaɪ ˈaʊ ˈeɪ ˈi ˈoʊ ˈu ˈæ ˈɑ ˈɔ ˈɔɪ ˈɛ ˈɝ ˈɪ ˈʊ ˈʌ ˌaɪ ˌaʊ ˌeɪ ˌi ˌoʊ ˌu ˌæ ˌɑ ˌɔ ˌɔɪ ˌɛ ˌɝ ˌɪ ˌʊ ˌʌ θ`
- [LJS-IPA-102000-durations](https://tuc.cloud/index.php/s/fPMqsYENEgRkm9J): Model trained on LJ Speech dataset with IPA transcriptions for 102000 iterations (= 500 epochs) with separated learning of stress and phoneme durations
  - Symbolset: `! " ' ( ) , - — . : ; ? SIL0 SIL1 SIL2 SIL3 [ ] aɪ aɪː aɪˑ aɪ˘ aʊ aʊː aʊˑ aʊ˘ b bː bˑ b˘ d dː dˑ d˘ d͡ʒ d͡ʒː d͡ʒˑ d͡ʒ˘ eɪ eɪː eɪˑ eɪ˘ f fː fˑ f˘ h hː hˑ i iː iˑ i˘ j jː jˑ j˘ k kː kˑ k˘ l lː lˑ l˘ m mː mˑ m˘ n nː nˑ n˘ oʊ oʊː oʊˑ oʊ˘ p pː pˑ p˘ s sː sˑ s˘ t tː tˑ t˘ t͡ʃ t͡ʃː t͡ʃˑ t͡ʃ˘ u uː uˑ u˘ v vː vˑ v˘ w wː wˑ w˘ z zː zˑ z˘ æ æː æˑ æ˘ ð ðː ðˑ ð˘ ŋ ŋː ŋˑ ŋ˘ ɑ ɑː ɑˑ ɑ˘ ɔ ɔɪ ɔɪː ɔɪ˘ ɔː ɔˑ ɔ˘ ɛ ɛː ɛˑ ɛ˘ ɝ ɝː ɝˑ ɝ˘ ɡ ɡː ɡˑ ɡ˘ ɪ ɪː ɪˑ ɪ˘ ɹ ɹː ɹˑ ɹ˘ ʃ ʃː ʃˑ ʃ˘ ʊ ʊː ʊˑ ʊ˘ ʌ ʌː ʌˑ ʌ˘ ʒ ʒː ʒˑ ʒ˘ ˈaɪ ˈaɪː ˈaɪˑ ˈaɪ˘ ˈaʊ ˈaʊː ˈaʊˑ ˈaʊ˘ ˈeɪ ˈeɪː ˈeɪˑ ˈeɪ˘ ˈi ˈiː ˈiˑ ˈi˘ ˈoʊ ˈoʊː ˈoʊˑ ˈoʊ˘ ˈu ˈuː ˈuˑ ˈu˘ ˈæ ˈæː ˈæˑ ˈæ˘ ˈɑ ˈɑː ˈɑˑ ˈɑ˘ ˈɔ ˈɔɪ ˈɔɪː ˈɔɪˑ ˈɔɪ˘ ˈɔː ˈɔˑ ˈɔ˘ ˈɛ ˈɛː ˈɛˑ ˈɛ˘ ˈɝ ˈɝː ˈɝˑ ˈɝ˘ ˈɪ ˈɪː ˈɪˑ ˈɪ˘ ˈʊ ˈʊː ˈʊˑ ˈʊ˘ ˈʌ ˈʌː ˈʌˑ ˈʌ˘ ˌaɪ ˌaɪː ˌaɪˑ ˌaɪ˘ ˌaʊ ˌaʊː ˌaʊˑ ˌaʊ˘ ˌeɪ ˌeɪː ˌeɪˑ ˌeɪ˘ ˌi ˌiː ˌiˑ ˌi˘ ˌoʊ ˌoʊː ˌoʊˑ ˌoʊ˘ ˌu ˌuː ˌuˑ ˌu˘ ˌæ ˌæː ˌæˑ ˌæ˘ ˌɑ ˌɑː ˌɑˑ ˌɑ˘ ˌɔ ˌɔɪ ˌɔɪː ˌɔɪˑ ˌɔɪ˘ ˌɔː ˌɔˑ ˌɔ˘ ˌɛ ˌɛː ˌɛˑ ˌɛ˘ ˌɝ ˌɝː ˌɝˑ ˌɝ˘ ˌɪ ˌɪː ˌɪˑ ˌɪ˘ ˌʊ ˌʊː ˌʊˑ ˌʊ˘ ˌʌ ˌʌː ˌʌˑ ˌʌ˘ θ θː θˑ θ˘`

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

## Roadmap

- Outsource method to convert audio files to mel-spectrograms before training
- Better logging
- Provide more pre-trained models
- Add audio examples
- Adding tests

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
- `speech-dataset-parser>=0.0.4`
- `mel-cepstral-distance>=0.0.2`

## License

MIT License

## Acknowledgments

Model code adapted from [Nvidia](https://github.com/NVIDIA/tacotron2).

Papers:

- [Tacotron: Towards End-to-End Speech Synthesis](https://www.isca-speech.org/archive/interspeech_2017/wang17n_interspeech.html)
- [Natural TTS Synthesis by Conditioning Wavenet on MEL Spectrogram Predictions](https://ieeexplore.ieee.org/document/8461368)

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 416228727 – CRC 1410

## Citation

If you want to cite this repo, you can use this BibTeX-entry generated by GitHub (see *About => Cite this repository*).

- Taubert, S. (2023). tacotron-cli (Version 0.0.4) [Computer software]. [https://doi.org/10.5281/zenodo.7543638](https://doi.org/10.5281/zenodo.7543638)

## Cited by

- Taubert, S., Sternkopf, J., Kahl, S., & Eibl, M. (2022). A Comparison of Text Selection Algorithms for Sequence-to-Sequence Neural TTS. 2022 IEEE International Conference on Signal Processing, Communications and Computing (ICSPCC), 1–6. [https://doi.org/10.1109/ICSPCC55723.2022.9984283](https://doi.org/10.1109/ICSPCC55723.2022.9984283)
- Albrecht, S., Tamboli, R., Taubert, S., Eibl, M., Rey, G. D., & Schmied, J. (2022). Towards a Vowel Formant Based Quality Metric for Text-to-Speech Systems: Measuring Monophthong Naturalness. 2022 IEEE 9th International Conference on Computational Intelligence and Virtual Environments for Measurement Systems and Applications (CIVEMSA), 1–6. [https://doi.org/10.1109/CIVEMSA53371.2022.9853712](https://doi.org/10.1109/CIVEMSA53371.2022.9853712)

## Contributing

If you notice an error, please don't hesitate to open an issue.

### Development setup

```sh
# update
sudo apt update
# install Python 3.8, 3.9 & 3.10 for ensuring that tests can be run
sudo apt install python3-pip \
  python3.8 python3.8-dev python3.8-distutils python3.8-venv \
  python3.9 python3.9-dev python3.9-distutils python3.9-venv \
  python3.10 python3.10-dev python3.10-distutils python3.10-venv
# install pipenv for creation of virtual environments
python3.8 -m pip install pipenv --user

# check out repo
git clone https://github.com/stefantaubert/tacotron.git
cd tacotron
# create virtual environment
python3.8 -m pipenv install --dev
```

## Running the tests

```sh
# first install the tool like in "Development setup"
# then, navigate into the directory of the repo (if not already done)
cd tacotron
# activate environment
python3.8 -m pipenv shell
# run tests
tox
```

Final lines of test result output:

```log
  py38: commands succeeded
  py39: commands succeeded
  py310: commands succeeded
  congratulations :)
```

## Changelog

- v0.0.4 (2023-01-17)
  - Added:
    - Added `--n_jobs` argument for `train` and `continue-train`
    - Added support of more diphthongs for stress detection
    - Added logging of mel-spectrogram duration in synthesis
    - Added logging of PEN in `validate` on non-fast validation
    - Added logging of device in training
    - Added returning of an exit code
  - Bugfixes:
    - Fixed evaluation of mapping before training
    - Fixed `validate` with custom file names
- v0.0.3 (2022-10-19)
  - Support for separate learning of phoneme durations
  - Support for learning via one-hot encoding or embedding
  - Bugfixes
- v0.0.2 (2022-09-28)
  - Added support for separate learning of tone
  - Added plotting of speaker weights in `analyze`
  - Bugfixes
- v0.0.1 (2022-06-08)
  - Initial release
