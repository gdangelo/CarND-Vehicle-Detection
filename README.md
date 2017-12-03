# CarND-Vehicle-Detection

> Vehicle Detection and Tracking Project for Self-Driving Car ND

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![Video output](https://user-images.githubusercontent.com/4352286/33525108-0200d7cc-d829-11e7-819d-fc0de2a9b233.gif)

The goal of this project is to write a software pipeline to detect vehicles in a video (start with the [test_video.mp4](https://github.com/gdangelo/CarND-Vehicle-Detection/blob/master/test_videos/test_video.mp4) and later implement on full [project_video.mp4](https://github.com/gdangelo/CarND-Vehicle-Detection/blob/master/test_videos/project_video.mp4)). More info in the [writeup](https://github.com/gdangelo/CarND-Vehicle-Detection/blob/master/writeup.md).


## The Project
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

Some example images for testing the pipeline on single frames are located in the [`test_images`](https://github.com/gdangelo/CarND-Vehicle-Detection/tree/master/test_images) folder.  Examples of the output from each stage of the pipeline is located in the folder called [`ouput_images`](https://github.com/gdangelo/CarND-Vehicle-Detection/tree/master/output_images). 

## Overview
Starting to work on this project consists of the following steps:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer
2. Create a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html) using this project
3. Each time you wish to work, activate your `conda` environment
4. Launch `python vehicle_detection.py`

---

## Installation

**Download** the latest version of `miniconda` that matches your system.

**NOTE**: There have been reports of issues creating an environment using miniconda `v4.3.13`. If it gives you issues try versions `4.3.11` or `4.2.12` from [here](https://repo.continuum.io/miniconda/).

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

**Clone** the project:

```sh
git clone https://github.com/gdangelo/CarND-Vehicle-Detection.git
cd CarND-Vehicle-Detection
```

**Download the datasets** as described in the `Project` section above, and unzip it under the `vehicles` and `non-vehicles` folders.

**Setup** your `carnd` environment. 

If you are on Windows, **rename** `meta_windows_patch.yml` to `meta.yml`

**Create** carnd.  Running this command will create a new `conda` environment that is provisioned with all libraries you need to be successful in this program.
```
conda env create -f environment.yml
```

*Note*: Some Mac users have reported issues installing TensorFlow using this method. The cause is unknown but seems to be related to `pip`. For the time being, we recommend opening environment.yml in a text editor and swapping
```yaml
    - tensorflow==0.12.1
```
with
```yaml
    - https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl
```

**Verify** that the carnd environment was created in your environments:

```sh
conda info --envs
```

**Cleanup** downloaded libraries (remove tarballs, zip files, etc):

```sh
conda clean -tp
```

### Uninstalling 

To uninstall the environment:

```sh
conda env remove -n carnd
```

---

## Usage

Now that you have created an environment, in order to use it, you will need to activate the environment. This must be done **each** time you begin a new working session i.e. open a new terminal window. 

**Activate** the `carnd` environment:

### OS X and Linux
```sh
$ source activate carnd
```
### Windows
Depending on shell either:
```sh
$ source activate carnd
```
or

```sh
$ activate carnd
```

Now all of the `carnd` libraries are available to you. 

**Launch** the main python file: 

```sh
$ python vehicle_detection.py
```

That's it.

---

## Questions or Feedback

> Contact me anytime for anything about my projects or machine learning in general. I'd be happy to help you :wink:

* Twitter: [@gdangel0](https://twitter.com/gdangel0)
* Linkedin: [Grégory D'Angelo](https://www.linkedin.com/in/gregorydangelo)
* Email: [gregory@gdangelo.fr](mailto:gregory@gdangelo.fr)


