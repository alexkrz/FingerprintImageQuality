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


# returns minutia patch
def getpatch(x, y, patch_minu_radio, img_size, original_image):
    try:
        x_patch_minu_radio = patch_minu_radio
        y_patch_minu_radio = patch_minu_radio
        # Extract patch from image
        x_begin = x - patch_minu_radio
        y_begin = y - patch_minu_radio

        # check if begin out of image
        if x_begin < 0:
            x_patch_minu_radio += (-1) * x_begin
            x_begin = 0
        if y_begin < 0:
            y_patch_minu_radio += (-1) * y_begin
            y_begin = 0
        # check if end out of image
        x_end = x_begin + 2 * x_patch_minu_radio
        y_end = y_begin + 2 * y_patch_minu_radio
        if x_end > img_size[0]:
            offset = x_end - img_size[0]
            x_begin -= offset
            x_end = img_size[0]
        if y_end > img_size[1]:
            offset = y_end - img_size[1]
            y_begin -= offset
            y_end = img_size[1]

        # create patch
        patch_minu = original_image[x_begin:x_end, y_begin:y_end]
        return patch_minu
    except:
        return np.array(None)


def generate_minu_preds(
    deploy_set: str,
    minupath: str,
    output_dir: str,
    extens: str,
) -> None:
    # Dummy values in case of failure
    minus_arr = []
    for nn in range(100):
        minus_arr.append(str(-1))

    # Load FineNet to verify
    model = FineNetmodel(num_classes=2, pretrained_path=None, input_shape=(224, 224, 3))

    dense = model.layers[-1]
    model_out = Model(model.input, model.layers[-2].output)
    model_out.summary()
    x = model_out.output
    dropout = Dropout(rate=0.3)(x, training=True)
    prediction = dense(dropout)
    model_FineNet = Model(inputs=model.input, outputs=prediction)

    model_FineNet.summary()

    # Load pre-trained FineNet weights
    FineNet_path = "output_FineNet/FineNet_dropout/FineNet__dropout__model.h5"
    model_FineNet.load_weights(FineNet_path)
    print("Pretrained FineNet loaded.")

    model_FineNet.compile(
        loss="categorical_crossentropy", optimizer=Adam(lr=0), metrics=["accuracy"]
    )

    print("Set", deploy_set)
    set_name = deploy_set.split("/")[-2]

    os.mkdir(output_dir + "/" + set_name + "/")

    # Read image and GT
    img_name, folder_name, img_size = get_maximum_img_size_and_names(deploy_set)

    logging.info('Predicting "%s":' % (set_name))

    for i in range(0, len(img_name)):
        logging.info('"%s" %d / %d: %s' % (set_name, i + 1, len(img_name), img_name[i]))

        image = imageio.imread(deploy_set + "/" + img_name[i] + extens, pilmode="L")  # / 255.0

        img_size = image.shape
        img_size = np.array(img_size, dtype=np.int32) // 8 * 8
        image = image[: img_size[0], : img_size[1]]

        original_image = image.copy()

        ########read minutiae from file########
        minufile = open("%s/%s.mnt" % (minupath, img_name[i]), "r")
        minu_list = []
        for line, content in enumerate(minufile):
            if line > 1:
                x, y, _, _ = [float(x) for x in content.split()]
                minu_list.append([int(x), int(y)])

        preds = {}

        # ======= Verify using FineNet ============
        patch_minu_radio = 28
        for idx, minu in enumerate(minu_list):
            print((minu[0], minu[1]))
            minu_prediction = []
            try:
                # Extract patch from image
                patch_minu = getpatch(
                    int(minu[1]), int(minu[0]), patch_minu_radio, img_size, original_image
                )
                patch_minu = cv2.resize(
                    patch_minu, dsize=(224, 224), interpolation=cv2.INTER_NEAREST
                )

                ret = np.empty((patch_minu.shape[0], patch_minu.shape[1], 3), dtype=np.uint8)
                ret[:, :, 0] = patch_minu
                ret[:, :, 1] = patch_minu
                ret[:, :, 2] = patch_minu
                patch_minu = ret
                patch_minu = np.expand_dims(patch_minu, axis=0)

                # predict 100 times on each minutia
                for n in range(100):
                    [isMinutiaeProb] = model_FineNet.predict(patch_minu)  # XXX: Key approach
                    isMinutiaeProb = isMinutiaeProb[0]

                    minu_prediction.append(str(isMinutiaeProb))

            except KeyboardInterrupt:
                raise
            except:
                minu_prediction = minus_arr

            preds[str(idx)] = minu_prediction

        with open("%s/%s/%s.json" % (output_dir, set_name, img_name[i]), "w") as file:
            json.dump(preds, file)


def main(
    data_dir: Path,
    minu_dir: Path,
    output_dir: Path,
    cuda_visible_devices: Optional[list] = None,
    file_ext: str = ".bmp",
):
    assert data_dir.exists()
    assert minu_dir.exists()
    assert output_dir.exists()
    if isinstance(cuda_visible_devices, list):
        if len(cuda_visible_devices) == 1:
            gpu_devices = str(cuda_visible_devices[0])
        else:
            gpu_devices = str(cuda_visible_devices)
    else:
        gpu_devices = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

    tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    generate_minu_preds(
        deploy_set=str(data_dir),
        minupath=str(minu_dir),
        output_dir=str(output_dir),
        extens=file_ext,
    )


if __name__ == "__main__":
    from jsonargparse import ActionConfigFile, ArgumentParser
    # jsonargparse Path_* types check for existence and access rights
    from jsonargparse.typing import Path_drw

    parser = ArgumentParser(parser_mode="omegaconf", description="Feature Extraction")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--minu_dir", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--file_ext", type=str, default=".bmp")
    parser.add_argument("--experiment_name", type=str, default=Path(__file__).stem)
    parser.add_argument("--cuda_visible_devices", type=Optional[list], default=None)
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
