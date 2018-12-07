# LipReading

Main repository for LipReading with Deep Neural Networks

## Introduction

The goal is to implement LipReading: Similar to how end-to-end Speech
Recognition systems work, mapping high-fidelity speech audio to sensible
characters and word level outputs, we will do the same for "speech visuals". 
In particular, we will take video frame input, extract the relevant mouth/chin
signals as input to map to characters and words.

## Overview

- [TODO](#**todo**)
- [Architecture](#architecture): High level pipeline
  - [Vision Pipeline](#vision-pipeline)
  - [NLP Pipeline](#nlp-pipeline)
  - [Datasets](#datasets)
- [Setup](#setup): Quick setup and installation instructions
  - [SpaCy Setup](#spacy-setup): Setup for NLP utilities.
  - [Data Directories Structure](#data-directories-structure): How data files are organized
  - [Collecting Data](#collecting-data): See [README_DATA_COLLECTION.md](./README_DATA_COLLECTION.md)
  - [Getting Started](#getting-started): Finally get started on running things
    - [Tutorial on Configuration files](#configuration): Tutorial on how to run executables via a config file
    - [Download Data](#download-data): Collect raw data from Youtube.
    - [Generate Dataview](#generate-dataview): Generate dataview from raw data.
  - [Train Model](#train-model): :train: Train :train:
    - [Examples](#examples): Example initial configurations to experiment.
  - [Tensorboard Visualization](#tensorboard-visualization)
- [Other Resources](#other-resources): Collection of reading material, and
  projects
  
## **TODO** 

A high level overview of some TODO items. For more project details please see the 
Github [project](https://github.com/joseph-zhong/LipReading/projects/2)

- [x] Download Data (926 videos)
- [x] Build Vision Pipeline (1 week) [in review](https://github.com/joseph-zhong/LipReading/projects/2#card-14669202)
- [ ] Build NLP Pipeline (1 week) [wip](https://github.com/joseph-zhong/LipReading/projects/2#card-14669211)
- [ ] Build Loss Fn and Training Pipeline (2 weeks) [wip](https://github.com/joseph-zhong/LipReading/projects/2#card-14669251)
- [ ] Train :train: and Ship :ship: [wip](https://github.com/joseph-zhong/LipReading/projects/2#card-14669014)
  
## Architecture

There are two primary interconnected pipelines: a "vision" pipeline for extracting
the face and lip features from video frames, along with a "nlp-inspired"
pipeline for temporally correlating the sequential lip features into the final
output.

Here's a quick dive into tensor dimensionalities

### Vision Pipeline

```javascript
Video -> Frames       -> Face Bounding Box Detection      -> Face Landmarking    
Repr. -> (n, y, x, c) -> (n, (box=1, y_i, x_i, w_i, h_i)) -> (n, (idx=68, y, x))   
```

### NLP Pipeline

```javascript
 -> Letters  ->  Words    -> Language Model 
 -> (chars,) ->  (words,) -> (sentences,)
```

### Datasets

- `all`: 926 videos (projected, not generated yet)
- `large`: 464 videos (failed at 35/464)
- `medium`: 104 videos (currently at 37/104)
- `small`: 23 videos 
- `micro`: 6 videos
- `nano`: 1 video

## Setup

0. Clone this repository and install the requirements. We will be using `python3`.
 
Please make sure you run python scripts, setup your `PYTHONPATH` at `./`, as well as a workspace env variable.

```bash
git clone git@github.com:joseph-zhong/LipReading.git 
# (optional, setup venv) cd LipReading; python3  -m venv .
```

1. Once the repository is cloned, the last step for setup is to setup the repository's `PYTHONPATH` and workspace environment variable to take advantage of standardized directory utilities in [`./src/utils/utility.py`](src/utils/utility.py)

Copy the following into your `~/.bashrc`

```bash
export PYTHONPATH="$PYTHONPATH:/path/to/LipReading/" 
export LIP_READING_WS_PATH="/path/to/LipReading/"
```

2. Install the simple `requirements.txt` with `PyTorch` with CTCLoss, `SpaCy`, and others.

On MacOS for CPU capabilities only.

```bash
pip3 install -r requirements.macos.txt
```

On Ubuntu, for GPU support

```bash
pip3 install -r requirements.ubuntu.txt
```

### SpaCy Setup

We need to install a pre-built English model for some capabilities

```bash
python3 -m spacy download en
```

### Data Directories Structure

This allows us to have a simple standardized directory structure for all our datasets, raw data, model weights, logs, etc.

```text
./data/
  --/datasets (numpy dataset files for dataloaders to load)
  --/raw      (raw caption/video files extracted from online sources)
  --/weights  (model weights, both for training/checkpointing/running)
  --/tb       (Tensorboard logging)
  --/...
```

See [`./src/utils/utility.py`](src/utils/utility.py) for more.

## Getting Started

Now that the dependencies are all setup, we can finally do stuff!

### Configuration

Each of our "standard" scripts in `./src/scripts` (i.e. not `./src/scripts/misc`) take the standard `argsparse`-style 
arguments. For each of the "standard" scripts, you will be able to pass `--help` to see the expected arguments.
To maintain reproducibility, cmdline arguments can be written in a raw text file with one argument per line.

e.g. for `./config/gen_dataview/nano`

```text
--inp=StephenColbert/nano 
``` 

Represent the arguments to pass to `./src/scripts/generate_dataview.py`, automatically passable via 

```bash
./src/scripts/generate_dataview.py $(cat ./config/gen_dataview/nano)
```

## Train Model

3. Train Model

```bash
./src/scripts/train.py
```

### Examples

#### Training on Micro

```bash
./src/scripts/train_model.py $(cat ./config/train/micro)
```

## Tensorboard Visualization

See [README_TENSORBOARD.md](README_TENSORBOARD.md)

## Other Resources

This is a collection of external links, papers, projects, and otherwise
potentially helpful starting points for the project.

- [Other Projects](#other-projects)
- [Other Academic Papers](#other-academic-papers)
- [Aacdemic Datsets](#academic-datasets)

### Other Projects

- Lip Reading - Cross Audio-Visual Recognition using 3D Convolutional Neural
  Networks (Jul. 2017, West Virginia University)
  - Github: https://github.com/joseph-zhong/lip-reading-deeplearning#demo
  - Demo:
    https://codeocean.com/2017/07/14/3d-convolutional-neural-networks-for-audio-visual-recognition/code 
  - Paper: 
    - (IEEE) https://ieeexplore.ieiee.org/stamp/stamp.jsp?tp=&arnumber=8063416 
    - (ArXiv) https://arxiv.org/pdf/1706.05739.pdf
- Lip reading using CNN and LSTM (2017, Stanford)
  - http://cs231n.stanford.edu/reports/2016/pdfs/217_Report.pdf
  - Runtime: https://github.com/adwin5/CNN-for-visual-speech-recognition
  - Training:
    https://github.com/adwin5/lipreading-by-convolutional-neural-network-keras
- LipNet (Dec. 2016, DeepMind)
  - Paper: https://arxiv.org/abs/1611.01599
  - Original Repo: https://github.com/bshillingford/LipNet
  - Working Keras Implementation: https://github.com/rizkiarm/LipNet

### Other Academic Papers

- Deep Audio-Visual Speech Recognition (Sept. 2018, DeepMind)
  - https://arxiv.org/pdf/1809.02108.pdf
- Lip Reading Sentences in the Wild (Jan. 2017, Deepmind)
  - https://arxiv.org/pdf/1611.05358.pdf
  - CNN + LSTM encoder, attentive LSTM decoder
- LARGE-SCALE VISUAL SPEECH RECOGNITION (Oct. 2018, DeepMind)
  - https://arxiv.org/pdf/1807.05162.pdf
- Lip Reading in Profile (2017, Oxford)
  - http://www.robots.ox.ac.uk/~vgg/publications/2017/Chung17a/chung17a.pdf
- JOINT CTC-ATTENTION BASED END-TO-END SPEECH RECOGNITION USING MULTI-TASK LEARNING (Jan. 2017, CMU)
  - https://arxiv.org/pdf/1609.06773.pdf
  - Joint CTC + attention model
  - [Unofficial implementation](https://github.com/hirofumi0810/tensorflow_end2end_speech_recognition)
- A Comparison of Sequence-to-Sequence Models for Speech Recognition (2017, Google & Nvidia)
  - https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0233.PDF
  - CTC vs. attention vs. RNN-transducer vs. RNN-transducer w/ attention
- EXPLORING NEURAL TRANSDUCERS FOR END-TO-END SPEECH RECOGNITION (July 2017, Baidu)
  - https://arxiv.org/pdf/1707.07413.pdf
  - CTC vs. attention vs. RNN-transducer


### Academic Datasets

- Lip Reading Datasets (Oxford)
  - http://www.robots.ox.ac.uk/~vgg/data/lip_reading/
