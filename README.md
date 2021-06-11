# Fingerprint Image Quality


## MiDeCon: Minutia Detection Confidence for Unsupervised and Accurate Minutia and Fingerprint Quality Assessment

The code for MiDeCon is currently cleaned and will be online end of June.

IEEE International Joint Conference on Biometrics (IJCB) 2021

* [Research Paper](https://arxiv.org/abs/2106.05601)
* [Implementation - MiDeCon on FineNet](to follow)


## Table of Contents 

<img src="Concept.png" width="500" align="right">

- [Abstract](#abstract)
- [Key Points](#key-points)
- [Results](#results)
- [Installation](#installation)
- [Citing](#citing)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Abstract


The most determinant factor to achieve high accuracies in fingerprint recognition systems is the quality of its samples. Previous works mainly proposed supervised solutions based on image properties that neglects the minutiae extraction process, despite that most fingerprint recognition techniques are based on this extracted information. Consequently, a fingerprint image might be assigned as high quality even if the utilized minutia extractor produces unreliable information for recognition. In this work, we propose a novel concept of assessing minutia and fingerprint quality based on minutia detection confidence (MiDeCon). MiDeCon can be applied to an arbitrary deep learning based minutia extractor and does not require quality labels for learning. Moreover, the training stage of MiDeCon can be completely avoided if a pre-trained minutiae extraction neural network is available. We propose using the detection reliability of the extracted minutia as its quality indicator. By combining the highest minutia qualities, MeDiCon accurately determines the quality of a full fingerprint. Experiments are done on the publicly available databases of the FVC 2006 and compared against NIST’s widely-used fingerprint image quality software NFIQ1 and NFIQ2. The results demonstrate a significantly stronger quality assessment performance of the proposed MiDeCon-qualities as related works on both, minutia- and fingerprint-level. 

## Key Points
In contrast to previous works, the proposed approach:

- **Does not require quality labels for training** - Previous works often rely on error-prone labelling mechanisms without a clear definition of quality. Our approach avoids the use of inaccurate quality labels by using the minutia detection confidence as a quality estimate. Moreover, the training state can be completely avoided if pre-trained minutiae extraction neural network trained with dropout is available. 

- **Considers difficulties in the minutiae extraction** - Previous works estimates the quality of a fingerprint based on the properties of the image neglecting the minutiae extraction process. However, the extraction process might face difficulties that are not considered in the image properties and thus, produce unreliable minutia information. Our solution defines quality through the prediction confidence of the extractor and thus, considers this problem. 

- **Produces continuous quality values** - While previous works often categorize the quality outputs in discrete categories (e.g. {good, bad, ugly}; {1,2,3,4,5}),, our approach produces continuous quality values that allow more fine-grained and flexible enrolment and matching processes. 

- **Includes quality assessment of single minutiae** - Unlike previous works, our solution assesses the quality of full fingerprints as well as the quality of single minutiae. This is specifically useful in forensic scenarios where forensic examiners aim to find reliable minutiae suitable for identification.

For more details, please take a look at the paper.

## Results

### Quality Assessment of Single Minutiae

Evaluating minutia quality assessment - only a certain number of the highest quality minutiae are used for recognition.
The recognition performance is reported in FNMR@![\Large 10^{-2}](https://latex.codecogs.com/gif.latex?\inline&space;10^{-2})FMR on the Bozorth3 and the MCC matcher. Each DB was captured with a different sensor. Our proposed methodology based on minutia detection confidence shows lower recognition errors than related works in all cases, except on the synthetic data (DB4). This demonstrates a strong quality estimation performance for single minutiae.

<img src="Table1.png" width="800" > 


### Quality Assessment of Full Fingerprints

Fingerprint quality assessment on the MCC matcher. Each row represents the recognition error at a different FMR
(![\Large 10^{-1}](https://latex.codecogs.com/gif.latex?\inline&space;10^{-1}), ![\Large 10^{-2}](https://latex.codecogs.com/gif.latex?\inline&space;10^{-2}), and ![\Large 10^{-3}](https://latex.codecogs.com/gif.latex?\inline&space;10^{-3})). Especially on the real-world sensor data, the proposed approach outperforms the widely-used NFIQ and NFIQ2 baselines. This holds true for all investigated sensor-types.

<img src="Figure3.png" width="800" > 

## Installation
TODO - Andre

We recommend Anaconda to install the required packages.
This can be done by creating an virtual environment via

```shell
conda env create -f environment.yml
```

or by manually installing the following packages.


```shell
conda create -n serfiq python=3.6.9
conda install cudatoolkit
conda install cudnn
conda install tensorflow=1.14.0
conda install mxnet
conda install mxnet-gpu
conda install tqdm
conda install -c conda-forge opencv
conda install -c anaconda scikit-learn
conda install -c conda-forge scikit-image
conda install keras=2.2.4
```

After the required packages have been installed, also download the [Insightface codebase at the needed git point in the repository history](https://github.com/deepinsight/insightface/tree/60bb5829b1d76bfcec7930ce61c41dde26413279) to a location of your choice and extract the archive if necessary.

We will refer to this location as _$Insightface_ in the following. 

The path to the Insightface repository must be passed to the [InsightFace class in face_image_quality.py](https://github.com/pterhoer/FaceImageQuality/blob/b59b2ec3c58429ee867dee25a4d8165b9c65d304/face_image_quality.py#L25). To avoid any problems, absolute paths can be used. Our InsightFace class automatically imports the required dependencies from the Insightface repository.
```
insightface = InsightFace(insightface_path = $Insightface) # Repository-path as parameter
```
[Please be aware to change the location in our example code according to your setup](https://github.com/pterhoer/FaceImageQuality/blob/b59b2ec3c58429ee867dee25a4d8165b9c65d304/serfiq_example.py#L9).

A pre-trained Arcface model is also required. We recommend using the "_LResNet100E-IR,ArcFace@ms1m-refine-v2_" model. [This can be downloaded from the Insightface Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo#31-lresnet100e-irarcfacems1m-refine-v2).

Extract the downloaded _model-0000.params_ and _model-symbol.json_ to the following location on your computer:
```
$Insightface/models/
```

After following these steps you can activate your environment (default: _conda activate serfiq_) and run the [example code](serfiq_example.py).

The implementation for SER-FIQ based on ArcFace can be found here: [Implementation](face_image_quality.py). <br/>
In the [Paper](https://arxiv.org/abs/2003.09373), this is refered to _SER-FIQ (same model) based on ArcFace_. <br/>






## Citing

If you use this code, please cite the following paper.


```
@misc{terhoerst2021midecon,
      title={{MiDeCon}: Unsupervised and Accurate Fingerprint and Minutia Quality Assessment based on Minutia Detection Confidence}, 
      author={Philipp Terh{\"{o}}rst and Andre Boller and Naser Damer and Florian Kirchbuchner and Arjan Kuijper},
      year={2021},
      eprint={2106.05601},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

If you make use of our MiDeCon implementation based on FineNet, please additionally cite the original ![MinutiaeNet paper](https://github.com/luannd/MinutiaeNet).

## Acknowledgement

This research work has been funded by the German Federal Ministry of Education and Research and the Hessen State Ministry for Higher Education, Research and the Arts within their joint support of the National Research Center for Applied Cybersecurity ATHENE. 

## License 

This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.
Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
