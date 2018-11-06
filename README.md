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



