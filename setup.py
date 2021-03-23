from setuptools import setup

setup(
    dependency_links=[
        "git+https://github.com/stefantaubert/audio-utils.git@e3c9398aeebe445a55e54d8cf55f286173d171c0#egg=audio-utils",
        "git+https://github.com/stefantaubert/cmudict-parser.git@5f7c38d98dcae0a462ec7dedb5f4a3b49310bfaf#egg=cmudict-parser",
        "git+https://github.com/stefantaubert/image-utils.git@64e6e1f0cae87c1b5408c24f7d4803c500bc1373#egg=image-utils",
        "git+https://github.com/stefantaubert/speech-dataset-parser.git@26aeb590c9e1653a38311bfe1a17c610e181cae3#egg=speech-dataset-parser",
        "git+https://github.com/stefantaubert/text-utils.git@7962aada2ab6fac7f74f747bbb9c9b31420abeeb#egg=text-utils",
    ],
    name="tacotron",
    version="1.0.0",
    url="https://github.com/stefantaubert/tacotron.git",
    author="Stefan Taubert",
    author_email="stefan.taubert@posteo.de",
    description="tacotron",
    packages=["tacotron"],
    install_requires=[
        "pandas",
        "matplotlib",
        "tqdm",
        "numpy",
        "Unidecode",
        "torch<=1.7.1",
    ],
)
