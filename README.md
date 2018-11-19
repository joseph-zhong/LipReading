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
- [Setup](#setup): Quick setup and installation instructions
  - [Data Directories Structure](#data-directories-structure): How data files are organized
  - [Collecting Data](#collecting-data): Overview on dependencies for collecting Data
      - [ffmpeg](#ffmpeg-installation)
      - [Pycaption](#pycaption-caption-reading)
      - [dlib](#dlib-setup)
      - [PRNet](#prnet-setup)
  - [Getting Started](#getting-started): Finally get started on running things
    - [Tutorial on Configuration files](#configuration): Tutorial on how to run executables via a config file
    - [Download Data](#download-data): Collect raw data from Youtube.
    - [Generate Dataview](#generate-dataview): Generate dataview from raw data.
    - [Train Model](#train-model): :train: Train :train:
      - [Examples](#examples): Example initial configurations to experiment.
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

## Setup

0. Clone this repository and install the requirements. We will be using `python3`.
 
Please make sure you run python scripts, setup your `PYTHONPATH` at `./`, as well as a workspace env variable.

```bash
git clone --recurse-submodules -j8 git@github.com:joseph-zhong/LipReading.git 
# (optional, setup venv) cd LipReading; python3  -m venv .
```

1. Once the repository is cloned, the last step for setup is to setup the repository's `PYTHONPATH` and workspace environment variable to take advantage of standardized directory utilities in [`./src/utils/utility.py`](src/utils/utility.py)

Copy the following into your `~/.bashrc`

```bash
export PYTHONPATH="$PYTHONPATH:/path/to/LipReading/" 
export LIP_READING_WS_PATH="/path/to/LipReading/"
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

### Collecting Data

To collect raw data and run the dataset generation pipeline the requirements are as follows:

- `ffmpeg`
- `pycaption`
- `dlib`
- `pytorch` and `tensorflow1.4+`

#### ffmpeg Installation

- `ffmpeg`
Can easily be installed on MacOS via:
```bash
brew install ffmpeg
ffmpeg -version
```

or on Ubuntu 16.04/18.04
```bash
sudo add-apt-repository ppa:jonathonf/ffmpeg-3
sudo apt-get update
sudo apt-get install ffmpeg libav-tools x264 x265
ffmpeg -version
```

#### Python Dependencies

This is primarily dependencies with `tensorflow` vs `tensorflow-gpu`, `scipy` and `imageio`.

On MacOS, try the following:

```bash
pip3 install -r requirements.macos.txt
```

On Ubuntu (with CUDA), instead we will use `tensorflow-gpu`

```bash
pip3 install -r requirements.ubuntu.txt
```

##### pycaption: Caption Reading 

Install `pycaption`, an open-source Python-based subtitle reader/writer tool. See the [Github](https://github.com/pbs/pycaption)
This will allow us to read and process the `webVTT` caption files.

```bash
pip3 install git+https://github.com/pbs/pycaption.git
```

Basic usage will be something along the lines of 

```python
import pycaption
reader = pycaption.WebVTTReader()
fname = 'data/raw/StephenColbert/ZfHPLpXKfWg.en.vtt'
with open(fname) as fin:
  captions_raw = fin.read()
  assert reader.detect(captions_raw), "Malformed file: '{}'".format(fname)
  caption_set = reader.read(captions_raw)
  assert not caption_set.is_empty(), "Empty VTT file: '{}'".format(fname)
  assert 'en-US' in caption_set.get_languages() # we'll need to check what other possibilities there are.
  captions = caption_set.get_captions(lang='en-US') 
  print("Number of captions:", len(captions))
  for cap in captions:
    print(cap)
```

Produces the following output:

```text
'00:00:00.467 --> 00:00:02.836\n>> Stephen: NOW, WHILE HE WAS\nIN CALIFORNIA, HE ADDRESSED SOME'
'00:00:02.836 --> 00:00:07.340\nMARINES AND HE ROLLED OUT A NEW\nPLAN FOR PROTECTING THE PLANET.'
'00:00:07.340 --> 00:00:13.013\n>> MY NEW NATIONAL STRATEGY FOR\nSPACE RECOGNIZES THAT SPACE IS A'
'00:00:13.013 --> 00:00:18.318\nWAR-FIGHTING DOMAIN JUST LIKE\nTHE LAND, AIR, AND SEA.'
'00:00:18.318 --> 00:00:23.690\nWE MAY EVEN HAVE A "SPACE\nFORCE," DEVELOP ANOTHER ONE.'
"00:00:23.690 --> 00:00:26.960\nSPACE FORCE-- WE HAVE THE AIR\nFORCE, WE'LL HAVE THE SPACE"
'00:00:26.960 --> 00:00:28.361\nFORCE.'
...
"00:01:59.586 --> 00:02:01.020\n( APPLAUSE )\nYOU WOULDN'T BE THINKING ABOUT"
'00:02:01.020 --> 00:02:01.354\nIT.'
'00:02:01.354 --> 00:02:03.223\n>> Stephen: YES, IF HILLARY\nCLINTON WERE PRESIDENT, WE'
"00:02:03.223 --> 00:02:06.159\nWOULDN'T BE SO URGENTLY TRYING\nTO FLEE THE PLANET."
'00:02:06.159 --> 00:02:11.998\n( LAUGHTER )\n( CHEERS AND APPLAUSE )'
'00:02:11.998 --> 00:02:13.433\nSEEMS NICE.'
'00:02:13.433 --> 00:02:14.434\nSEEMS NICE.'
```

#### `dlib` and `PRNet`: Real-time Face Detection and 3D Landmarking

`dlib` is a popular C++ based open-source computer vision library that has become a standardized tool in 
computer vision research with many useful utilities such as real-time 2D face-detection. 2009 
[Paper](http://www.jmlr.org/papers/volume10/king09a/king09a.pdf) and Github 
[project](https://github.com/davisking/dlib) 

`PRNet` is an amazingly powerful **real-time**, and **3D** face-reconstruction algorithm that came out of 
ECCV2018. It's truly incredible and enables robust, accurate, and most impressively, real-time 3D-face 
landmarking, all in simple Tensorflow and Python code. 
Thankfully the authors released the source code with an MIT License. Check out their ECCV2018 [Paper]
(http://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Feng_Joint_3D_Face_ECCV_2018_paper.pdf) and Github 
[project](https://github.com/YadiraF/PRNet).   

Each of these projects will require some pre-trained weights for their respective models (face-detection, 3D 
face-reconstruction).

##### dlib Setup

This can be a little bit tricky since it's a C++ library relying on `cmake` and `CUDA`, three things that can be
quite painful to get working together as a team but when they do the results are great.

###### GPU Setup

If you're using a machine with CUDA available, you should follow these instructions

Make sure CUDA and CuDNN are installed, this can be quite time-consuming.

```bash
# Installing CUDA 9.0 and cuDNN7.0.5 for CUDA 9.0 for Tensorflow 1.9.0 support
# For more information see: https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=debnetworki

# Installing CUDA 9.0.
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-9-0

# Installing cuDNN7.0.5.
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.0.5.15-1+cuda9.0_amd64.deb

# Test cuDNN7.0.5 installation correctness.
cp -ur /usr/src/cudnn_samples_v7/ tests
cd tests/cudnn_samples_v7/mnistCUDNN
make clean && make && ./mnistCUDNN
```

If cuDNN is properly installed and running on your Linux system, you will see a message similar to the following:
```text
Test passed!
```

###### `dlib` Python Setup

If you're not using the GPU, you can simply run
```bash
pip3 install dlib
```

Othewrwise you will need to compile `dlib` using the following

```bash
git clone https://github.com/davisking/dlib.git
cd dlib
python3 setup.py install \
  --yes USE_AVX_INSTRUCTIONS \
  --yes DLIB_USE_CUDA \
  --set CMAKE_PREFIX_PATH=/usr/local/cuda \
  --set CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/bin/ --clean
```

**Important:** make sure you see the following output
```text
...
-- Found CUDA: /usr/local/cuda (found suitable version "9.0", minimum required is "7.5") 
-- Looking for cuDNN install...
-- Found cuDNN: /usr/lib/x86_64-linux-gnu/libcudnn.so
-- Building a CUDA test project to see if your compiler is compatible with CUDA...
-- Checking if you have the right version of cuDNN installed.
-- Try OpenMP C flag = [-fopenmp]
-- Performing Test OpenMP_FLAG_DETECTED
-- Performing Test OpenMP_FLAG_DETECTED - Success
-- Try OpenMP CXX flag = [-fopenmp]
-- Performing Test OpenMP_FLAG_DETECTED
-- Performing Test OpenMP_FLAG_DETECTED - Success
-- Found OpenMP: -fopenmp  
-- Enabling CUDA support for dlib.  DLIB WILL USE CUDA
...
```

This last line is critical. Otherwise either CUDA or CuDNN is not installed correctly.

If you're using a virtualenv, you will need to copy the installed `.so` to your `virtualenv` 
```bash
cd build
cp -r lib.linux-x86_64-3.5/ </path/to/vituralenv/.../lib/python3.x/site-packages/>
```

We will need a few trained models for face detection, 2d landmarking, etc, conveniently provided by Davis King.

```bash
mkdir -p ./data/weights
cd ./data/weights
git clone https://github.com/davisking/dlib-models dlib
cd dlib
bzip2 -dk *.bz2
```

###### Testing `dlib`

Now let us verify the installation, run the following: We should expect inference time per image to be on the 
order of less than 0.3s per image. It will be over 10x slower on CPU (5-10s/inference).  

```python
import sys
import dlib
import time

if len(sys.argv) < 3:
    print(
        "Call this program like this:\n"
        "   ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg\n"
        "You can get the mmod_human_face_detector.dat file from:\n"
        "    http://dlib.net/files/mmod_human_face_detector.dat.bz2")
    exit()

cnn_face_detector = dlib.cnn_face_detection_model_v1(sys.argv[1])
for f in sys.argv[2:]:
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    ts = time.time()
    dets = cnn_face_detector(img, 1)
    print("Time elapsed: ", time.time() - ts)
```

##### PRNet Setup

Download the PRNet weights from [GDrive](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view?usp=sharing)
and place them into `./src/models/extern/prnet/Data/net-data`.

If you've installed the package requirements above, then test and make sure that PRNet will run

```bash
cd ./src/models/extern/prnet
python run_basics.py 
```

From here you can check `./extern/PRNet/TestImages/AFLW2000_results` for output.
You should see `*.obj` 3D face model files which you can open in Preview on MacOS, while the `*.txt` contain 
the `xyz` 3D coordinates of the 68 face landmarks in pixel space for the face.

You can see further examples of how to use this algorithm in [`./extern/PRNet/demo.py`](extern/PRNet/demo.py)

Once the api is verified, move the contents of the model and `uv` map weights to `./data/weights/prnet` 

## Getting Started

Now that the dependencies are all setup, we can finally do stuff!

### Configuration

Each of our "standard" scripts in `./src/scripts` (i.e. not `./src/scripts/misc`) take the standard `argsparse`-style 
arguments. For each of the "standard" scripts, you will be able to pass `--help` to see the expected arguments.
To maintain reproducibility, cmdline arguments can be written in a raw text file with one argument per line.

e.g. for `./config/gen_dataview/nano`

```text
--inp=StephenColbert/nano 
--outp_dir=StephenColbert/nano 
``` 

Represent the arguments to pass to `./src/scripts/generate_dataview.py`, automatically passable via 

```bash
./src/scripts/generate_dataview.py $(cat ./config/gen_dataview/nano)
```

### Download Data

1. Download the Data

This will pull the Stephen Colbert Youtube monologue videos (`.mp4`), audio (`.wav`) and captions (`.vtt`) into 
`./data/raw`.  

```bash
./src/scripts/misc/collect_data.py
```

### Generate Dataview

2. Generate the Dataview

```bash
./src/scripts/generate_dataview.py $(cat ./config/gen_dataview/nano)
```

This will correlate each of data examples as a dense matrix as follows for each corresponding video and caption 
pair.
Thus, during cross-validation or testing time, we will be able to separate the train/test samples based on 
either by unique videos (first 900 vs last 100), or perhaps just segments of videos (first `k` samples vs last 
`n-k` samples of each video).

A dataview is a dense table of data input and label pairs.
For our purposes, we will generate a table for each video-caption pair as follows:

| idx  |  start, end  |  face_lmk_seq         | face_vtx_seq         | caption text |
| ---- | ------------ | --------------------- | -------------------- | ------------ |
| `i`  | `(s_i, e_i)` | `(frames, lmks, yxz)` | `(frames, vtx, xyz)` |  `"str...."` |

Note, the face landmarks are landmarks with coordinates relative to the face frame, which are take from the raw
landmarks which are coordinates relative to the full frame.

There are 68 canonical face landmarks, and 45128 total face vertices in the point cloud.

### Train Model

3. Train Model

```bash
./src/scripts/train_model.py
```

#### Examples

##### Training on Micro

```bash
./src/scripts/train_model.py $(cat ./config/train/micro)
```

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
