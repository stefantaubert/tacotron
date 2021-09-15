# Remote Dependencies

- text-utils
  - pronunciation_dict_parser
  - g2p_en
  - sentence2pronunciation
- text-selection
- audio_utils
- image_utils
- speech-dataset-preprocessing
  - speech-dataset-parser
  - text-utils
  - audio-utils
  - image-utils
- tts-preparation
  - text-utils
  - speech-dataset-preprocessing
  - accent-analyser
  - text-selection
  - sentence2pronunciation
- mcd

## Pipfile

### Local

```Pipfile
text-utils = {editable = true, path = "./../text-utils"}
text-selection = {editable = true, path = "./../text-selection"}
audio-utils = {editable = true, path = "./../audio-utils"}
image-utils = {editable = true, path = "./../image-utils"}
speech-dataset-preprocessing = {editable = true, path = "./../speech-dataset-preprocessing"}
tts-preparation = {editable = true, path = "./../tts-preparation"}
mcd = {editable = true, path = "./../mel_cepstral_distance"}

pronunciation_dict_parser = {editable = true, path = "./../pronunciation_dict_parser"}
g2p_en = {editable = true, path = "./../g2p"}
sentence2pronunciation = {editable = true, path = "./../sentence2pronunciation"}
speech-dataset-parser = {editable = true, path = "./../speech-dataset-parser"}
accent-analyser = {editable = true, path = "./../accent-analyser"}
```

### Remote

```Pipfile
text_utils = {editable = true, ref = "master", git = "https://github.com/stefantaubert/text-utils.git"}
text_selection = {editable = true, ref = "master", git = "https://github.com/stefantaubert/text-selection.git"}
audio_utils = {editable = true, ref = "master", git = "https://github.com/stefantaubert/audio-utils.git"}
image_utils = {editable = true, ref = "master", git = "https://github.com/stefantaubert/image-utils.git"}
speech-dataset-preprocessing = {editable = true, ref = "master", git = "https://github.com/stefantaubert/speech-dataset-preprocessing.git"}
tts-preparation = {editable = true, ref = "master", git = "https://github.com/stefantaubert/tts-preparation.git"}
mcd = {editable = true, ref = "main", git = "https://github.com/jasminsternkopf/mel_cepstral_distance.git"}
```

## setup.cfg

```cfg
text_utils@git+https://github.com/stefantaubert/text-utils.git@master
text_selection@git+https://github.com/stefantaubert/text-selection.git@master
audio_utils@git+https://github.com/stefantaubert/audio-utils.git@master
image_utils@git+https://github.com/stefantaubert/image-utils.git@master
speech_dataset_preprocessing@git+https://github.com/stefantaubert/speech-dataset-preprocessing.git@master
tts_preparation@git+https://github.com/stefantaubert/tts_preparation.git@master
mcd@git+https://github.com/jasminsternkopf/mel_cepstral_distance.git@main
```
