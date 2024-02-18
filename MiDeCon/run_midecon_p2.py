import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import imageio.v2 as imageio
import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from keras import backend as K
from keras.layers.core import Dropout
from keras.models import Model
from keras.optimizers import Adam

# Suppress tensorflow warnings for now
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from CoarseNet.CoarseNet_utils import get_maximum_img_size_and_names
from CoarseNet.MinutiaeNet_utils import init_log
from FineNet.FineNet_model import FineNetmodel


def npdeviation(predictions): 
    return np.std(predictions)
    
def npmean(predictions):
    return np.mean(predictions)
    
def convert_array(array):
    for index, item in enumerate(array):
        array[index] = float(item)
    return array
    
def save_result(path, measurements):
    with open(path, 'w') as file:
        json.dump(measurements, file)

    
def mnt_reader(file_name):                      
    f = open(file_name)
    minutiae = []
    for i, line in enumerate(f):
        if i <= 1:
            continue
        w, h, o, _ = [float(x) for x in line.split()]
        w, h = int(w), int(h)
        minutiae.append([w, h, o])
    f.close()
    return minutiae


def mean_best(data, best_x):
    data = sorted(data.items(), key=lambda item: float(item[1]), reverse=True)
    av_value = 0.
    i = 0
    while i < best_x and i < len(data):
        av_value += float(data[i][1])
        i += 1
    av_value = av_value / best_x
    return av_value


def compute_minutiae_quality(
    minu_conf_dir: Path,
    output_dir: Path,
    alpha: float,
) -> None:
    files_in_dir = os.listdir(minu_conf_dir)
    for file_name in files_in_dir:
        #print(file_name)
        with open(minu_conf_dir / file_name, 'r') as file:
            data = json.load(file)
            measurements = {}
            for key, value in data.items():
                predictions = convert_array(value)

                deviation = npdeviation(predictions)
                mean = npmean(predictions)
                
                measurements[key] = (((1-alpha)*mean) + (alpha*deviation))
            save_result(output_dir / file_name, measurements)


def write_minutiae_template(
    minu_dir: Path,
    minu_conf_outdir: Path,
    output_dir: Path,
    ):
    files_in_dir = os.listdir(minu_conf_outdir)
    for file_name in files_in_dir:
        name = file_name.split(".")[0]
        
        with open(minu_conf_outdir / file_name, 'r') as file:
            #load minutiae qualities
            data = json.load(file)
            
            qualityminu = []
            for key, value in data.items():
                qualityminu.append((int(key), float(value)))
                
            #sort minutiae qualities in descending order
            qualityminu.sort(key=lambda tup: tup[1], reverse=True)
            
            #get minutiae from template
            minutiae = mnt_reader(minu_dir / (name + ".mnt"))
            
            
            #create template with new quality scores
            template = open(output_dir / (name + ".txt"), "w")
            c = 0
            while (c < len(qualityminu)):
                key = qualityminu[c][0]
                value = qualityminu[c][1]

                #write template file
                template.write(str(minutiae[key][0]) + " " + str(minutiae[key][1]) + " " + str(minutiae[key][2]) + " " + str(value) + " \n")
                c += 1


def compute_fingerprint_quality(
        minu_conf_outdir: Path,
        output_dir: Path,
    ):
    img_list = []
    files_in_dir = os.listdir(minu_conf_outdir)
    for file_name in files_in_dir:
        name = file_name.split(".")[0]
        with open(minu_conf_outdir / file_name, 'r') as file:
            data = json.load(file)
            if not len(data.keys()) == 0:
                #compute the quality value for a fingerprint
                value = mean_best(data, 25)
            else: value = -1.                
            
            #append img to list
            img_list.append([name, value])
                    
    #sort quality values in descending order
    img_list.sort(key=lambda tup: tup[1], reverse=True)
    print(len(img_list))

    imgs = open(output_dir / "imglist.txt", "w")
    for elem in img_list:
        name, value = elem
        imgs.write(name+","+str(value)+"\n")


def main(
    minu_dir: Path,
    minu_conf_dir: Path,
    output_dir: Path,
    alpha: float = 0.5,
):
    assert minu_dir.exists()
    assert minu_conf_dir.exists()
    assert output_dir.exists()
    
    output_dir1 = output_dir / "minutiae_quality"
    output_dir1.mkdir()
    # output_dir2 = output_dir / "minutiae_template"
    # output_dir2.mkdir()
    output_dir3 = output_dir / "fingerprint_quality"
    output_dir3.mkdir()

    compute_minutiae_quality(minu_conf_dir, output_dir1, alpha)
    # write_minutiae_template(minu_dir, output_dir1, output_dir2)
    compute_fingerprint_quality(output_dir1, output_dir3)


if __name__ == "__main__":
    from jsonargparse import ActionConfigFile, ArgumentParser
    # jsonargparse Path_* types check for existence and access rights
    from jsonargparse.typing import Path_drw

    parser = ArgumentParser(parser_mode="omegaconf", description="Feature Extraction")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--minu_dir", type=Path, required=True)
    parser.add_argument("--minu_conf_dir", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--experiment_name", type=str, default=Path(__file__).stem)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    cfg = vars(args)

    # Adjust config
    cfg.pop("config")
    assert cfg["results_dir"].exists()
    cfg_out_dir = Path(
        cfg["results_dir"] / cfg["experiment_name"] / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    cfg_out_dir.mkdir(parents=True, exist_ok=False)
    parser.save(
        cfg=cfg, path=str(cfg_out_dir / "config.yaml"), overwrite=True
    )  # Save config to output_dir
    cfg.pop("results_dir")
    cfg.pop("experiment_name")
    cfg["output_dir"] = cfg_out_dir
    # Write log file to output_dir
    init_log(str(cfg["output_dir"]))

    main(**cfg)
