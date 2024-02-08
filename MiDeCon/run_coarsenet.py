import os
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from keras import backend as K

# Suppress tensorflow warnings for now
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from CoarseNet.CoarseNet_model import inference
from CoarseNet.MinutiaeNet_utils import init_log


def get_available_gpus():
    local_device_protos = tf.config.experimental.list_physical_devices("GPU")
    return [x.name for x in local_device_protos]


def main(
    data_dir: Path,
    output_dir: Path,
    cuda_visible_devices: Optional[list] = None,
    file_ext: str = ".bmp",
) -> None:
    assert data_dir.exists()
    assert output_dir.exists()
    if isinstance(cuda_visible_devices, list):
        if len(cuda_visible_devices) == 1:
            gpu_devices = str(cuda_visible_devices[0])
        else:
            gpu_devices = str(cuda_visible_devices)
    else:
        gpu_devices = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    # print(os.environ["CUDA_VISIBLE_DEVICES"])

    tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    pretrain_dir = "Models/CoarseNet.h5"
    FineNet_dir = "output_FineNet/FineNet_dropout/FineNet__dropout__model.h5"

    inference(
        deploy_set=str(data_dir),
        output_dir=str(output_dir),
        model_path=pretrain_dir,
        FineNet_path=FineNet_dir,
        file_ext=file_ext,
        isHavingFineNet=False,
    )


if __name__ == "__main__":
    from jsonargparse import ActionConfigFile, ArgumentParser

    parser = ArgumentParser(parser_mode="omegaconf", description="Feature Extraction")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--file_ext", type=str, default=".bmp")
    parser.add_argument("--experiment_name", type=str, default="run_coarsenet")
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
    # Save config to output_dir
    parser.save(cfg=cfg, path=str(cfg_out_dir / "config.yaml"), overwrite=True)
    # Write log file to output_dir
    cfg.pop("results_dir")
    cfg.pop("experiment_name")
    cfg["output_dir"] = cfg_out_dir
    # Write log file to output_dir
    init_log(str(cfg["output_dir"]))

    main(**cfg)
