# Tensorboard Visualization

Tensorboard is a tool that allows you to deeply and interactively visualize your data, outputs, learning metrics and even 
the model itself! 

![Tensorboard Example](https://github.com/lanpa/tensorboardX/raw/master/screenshots/Demo.gif)

## Setup 

First install `tensorboardX`, the Tensorboard clone for PyTorch originally from Tensorflow.

```bash
pip3 install tensorboardX
```

### Quick Start

When running the training script, there will be a directory indicating where the tensorboard logs are being written to.

```text
$ ./src/scripts/train.py
...
Initializing model

[2018-12-06 17:02:28  INFO train.py train:161] Writing Tensorboard logs to '/Users/josephz/GoogleDrive/Work/personal/experiments/py/ml/lipreading/LipReading/data/weights/StephenColbert/2'
Try visualizing by running the following:
	tensorboard --logdir='/Users/josephz/GoogleDrive/Work/personal/experiments/py/ml/lipreading/LipReading/data/weights/StephenColbert/2
Then open the following URL in your local browser. 
	If you're running on a remote machine see `README_TENSORBOARD.md` for help...
...
```

On a local machine, you can start the tensorboard server and directly visualize the logs 

```bash
tensorboard --logdir='/Users/josephz/GoogleDrive/Work/personal/experiments/py/ml/lipreading/LipReading/data/weights/StephenColbert/2
``` 

Note that this is only for one training session and to compare different training sessions with logs saved to `
./data/weights/StephenColbert/...`, you can visualize each of them, recursively via

```bash
tensorboard --logdir='/Users/josephz/GoogleDrive/Work/personal/experiments/py/ml/lipreading/LipReading/data/weights/StephenColbert
```

### Remote Server and Local Visualization

If you are training on a remote server, you can still visualize the results via `ssh` port forwarding.

For example, you are training on a remote server, `ai2`, you can instead ssh into that machine with the following, and 
gain access to the tensorboard server through `http`

```bash
ssh -L localhost:8888:localhost:8888 ai2
tensorboard --logdir='/Users/josephz/GoogleDrive/Work/personal/experiments/py/ml/lipreading/LipReading/data/weights/StephenColbert --port=8888
```

Now open a local Chrome browser and go to `localhost:8888`!
 