import os
from pathlib import Path
from typing import Optional
from datetime import datetime

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers.core import Dropout
from keras.optimizers import Adam

# Suppress tensorflow warnings for now
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from CoarseNet.MinutiaeNet_utils import init_log
from FineNet.FineNet_model import FineNetmodel


def generate_minu_preds():
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


def main(
    data_dir: Path,
    minu_dir: Path,
    output_dir: Path,
    cuda_visible_devices: Optional[list] = None,
    file_ext: str = ".bmp",
):
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

    generate_minu_preds()


if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(parser_mode="omegaconf", description="Feature Extraction")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--minu_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--file_ext", type=str, default=".bmp")
    parser.add_argument("--experiment_name", type=str, default="run_midecon")
    parser.add_argument("--cuda_visible_devices", type=Optional[list], default=None)
    args = parser.parse_args()
    cfg = vars(args)

    # Adjust config
    cfg.pop("config")
    data_dir = Path(cfg["data_dir"])
    minu_dir = Path(cfg["minu_dir"])
    results_dir = Path(cfg["results_dir"])
    assert data_dir.exists()
    assert minu_dir.exists()
    assert results_dir.exists()
    output_dir = Path(
        results_dir / cfg["experiment_name"] / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    # Save config to output_dir
    parser.save(cfg=cfg, path=str(output_dir / "config.yaml"), overwrite=True)
    # Write log file to output_dir
    init_log(str(output_dir))

    main()
