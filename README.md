# LipReading

Main repository for LipReading with Deep Neural Networks

## Introduction

The goal is to implement LipReading: Similar to how end-to-end Speech
Recognition systems work, mapping high-fidelity speech audio to sensible
characters and word level outputs, we will do the same for "speech visuals". 
In particular, we will take video frame input, extract the relevant mouth/chin
signals as input to map to characters and words.

## Overview

- [Other Resources](#other-resources): Collection of reading material, and
  projects
- [Get Started](#get-started): Quick setup

## Architecture

There are two primary interconnected pipelines: a "vision" pipeline for extracting
the face and lip features from video frames, along with a "nlp-inspired"
pipeline for temporally correlating the sequential lip features into the final
output.

### Pipeline Representations: A Dive into Tensor Dimensionalities

#### Vision Pipeline

```javascript
Video -> Frames       -> Face Bounding Box Detection      -> Face Landmarking    
Repr. -> (n, y, x, c) -> (n, (box=1, y_i, x_i, w_i, h_i)) -> (n, (idx=68, y, x))   
```

#### NLP Pipeline

```javascript
 -> Letters  ->  Words    -> Language Model 
 -> (chars,) ->  (words,) -> (sentences,)
```

## Get Started

0. Clone this repository and install the requirements. We will be using `python3`.
 
Please make sure you run python scripts, setup your `PYTHONPATH` at `./`, as well as a workspace env variable.

```bash
git clone --recurse-submodules -j8 git@github.com:joseph-zhong/LipReading.git 
# (optional, setup venv) cd LipReading; python3  -m venv .
cd LipReading 
pip3 install -r requirements.txt
pwd
# Copy the following into your `~/.bashrc`
# export PYTHONPATH="$PYTHONPATH:/path/to/LipReading/" 
# export LIP_READING_WS_PATH="/path/to/LipReading/"
```

### External Requirements

#### `pycaption`: Caption Reading 

Install `pycaption`, an open-source Python-based subtitle reader/writer tool. See the [Github](https://github.com/pbs/pycaption)
This will allow us to read and process the `webVTT` caption files.

```bash
pip3 install git+https://github.com/pbs/pycaption.git
```

Basic usage will be something along the lines of 

```python3
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

##### `dlib` Setup

We will need a few trained models for face detection, 2d landmarking, etc, conveniently provided by Davis King.

```bash
mkdir -p ./data/weights
cd ./data/weights
git clone https://github.com/davisking/dlib-models dlib
cd dlib
bzip2 -dk *.bz2
```

##### `PRNet` Setup

Download the PRNet weights from [GDrive](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view?usp=sharing)
and place them into `./extern/PRNet/Data/net-data`.

If you've installed the package requirements above, then test and make sure that PRNet will run

```bash
cd ./src/models/extern/prnet
python run_basics.py 
```

From here you can check `./extern/PRNet/TestImages/AFLW2000_results` for output.
You should see `*.obj` 3D face model files which you can open in Preview on MacOS, while the `*.txt` contain 
the `xyz` 3D coordinates of the 68 face landmarks in pixel space for the face.

You can see further examples of how to use this algorithm in [`./extern/PRNet/demo.py`](extern/PRNet/demo.py)

1. Download the Data

This will pull the Stephen Colbert Youtube monologue videos (`.mp4`), audio (`.wav`) and captions (`.vtt`) into 
`./data/raw`.  

```bash
./src/scripts/collect_data.py
```

2. Generate the Dataview

```bash
./src/scripts/generate_dataview.py
```

This will correlate each of data examples as a dense matrix as follows for each corresponding video and caption 
pair. 
Thus, during cross-validation or testing time, we will be able to separate the train/test samples based on 
either by unique videos (first 900 vs last 100), or perhaps just segments of videos (first `k` samples vs last 
`n-k` samples of each video).

| idx  | video_fname | captions_fname |  start, end  | landmark tensor       | caption text         |
| ---- | ----------- | -------------- | ------------ | --------------------- | -------------------- |
| `i`  | `...mp4`    | `...vtt`       | `(s_i, e_i)` | `(frames, lmks, yxz)` | `"str0, str1, ...,"` |




## **TODO** 

- [x] Download Data (943 videos)
- [ ] Build Vision Pipeline (1 week)
- [ ] Build NLP Pipeline (1 week)
- [ ] Build Loss Fn and Training Pipeline (2 weeks)
- [ ] Train :train: and Ship :ship: 

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
- LARGE-SCALE VISUAL SPEECH RECOGNITION (Oct. 2018, DeepMind)
  - https://arxiv.org/pdf/1807.05162.pdf
- Lip Reading in Profile (2017, Oxford)
  - http://www.robots.ox.ac.uk/~vgg/publications/2017/Chung17a/chung17a.pdf

### Academic Datasets

- Lip Reading Datasets (Oxford)
  - http://www.robots.ox.ac.uk/~vgg/data/lip_reading/



