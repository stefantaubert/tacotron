# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.5] - 2024-01-25

### Added

- `synthesize-grids`
- `create-mels`
- Support for Python 3.8-3.11

### Changed

- Logging method

## [0.0.4] - 2023-01-17

### Added

- `--n_jobs` argument for `train` and `continue-train`
- support of more diphthongs for stress detection
- logging of mel-spectrogram duration in synthesis
- logging of PEN in `validate` on non-fast validation
- logging of device in training
- returning of an exit code

### Fixed

- evaluation of mapping before training
- `validate` with custom file names

## [0.0.3] - 2022-10-19

### Added

- Support for separate learning of phoneme durations
- Support for learning via one-hot encoding or embedding

### Fixed

- Several bugfixes

## [0.0.2] - 2022-09-28

### Added

- support for separate learning of tone
- plotting of speaker weights in `analyze`

### Fixed

- Several bugfixes

## [0.0.1] - 2022-06-08

- Initial release

[unreleased]: https://github.com/stefantaubert/tacotron/compare/v0.0.5...HEAD
[0.0.5]: https://github.com/stefantaubert/tacotron/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/stefantaubert/tacotron/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/stefantaubert/tacotron/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/stefantaubert/tacotron/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/stefantaubert/tacotron/releases/tag/v0.0.1
