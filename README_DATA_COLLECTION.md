# Collecting Data

## Introduction

To collect raw data and run the dataset generation pipeline the requirements are as follows:

- `ffmpeg`: Video decoding
- `pycaption`: VTT File Parsing
- `dlib`: Face Detection
- `tensorflow1.4+`: Running PRNet 3D Face Alignment

## Setup

### ffmpeg Installation

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

### Python Dependencies

This is primarily dependencies with `tensorflow` vs `tensorflow-gpu`, `scipy` and `imageio`.

On MacOS, try the following:

```bash
pip3 install -r requirements.macos.txt
```

On Ubuntu (with CUDA), instead we will use `tensorflow-gpu`

```bash
pip3 install -r requirements.ubuntu.txt
```

#### pycaption: Caption Reading 

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

##### `dlib` Python Setup

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

##### Testing `dlib`

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

#### PRNet Setup

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
