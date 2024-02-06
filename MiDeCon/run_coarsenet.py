import os
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

os.environ["KERAS_BACKEND"] = "tensorflow"

from datetime import datetime

import tensorflow as tf
from keras import backend as K

# Suppress tensorflow warnings for now
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from CoarseNet.CoarseNet_model import inference
from CoarseNet.MinutiaeNet_utils import init_log


def get_available_gpus():
    local_device_protos = tf.config.experimental.list_physical_devices("GPU")
    return [x.name for x in local_device_protos]


@dataclass
class Config:
    data_dir: str = ""
    results_dir: str = ""
    experiment_name: str = "Unknown"
    file_ext: str = ".bmp"
    cuda_visible_devices: list = field(default_factory=list)


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(version_base="1.3", config_path="configs")
def main(cfg: DictConfig) -> None:
    # print(cfg)
    data_dir = Path(cfg["data_dir"])
    results_dir = Path(cfg["results_dir"])
    assert data_dir.exists()
    assert results_dir.exists()
    gpu_list = cfg["cuda_visible_devices"]
    if len(gpu_list) == 1:
        gpu_devices = str(gpu_list[0])
    else:
        gpu_devices = str(gpu_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    # print(os.environ["CUDA_VISIBLE_DEVICES"])

    tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    pretrain_dir = "Models/CoarseNet.h5"
    FineNet_dir = "output_FineNet/FineNet_dropout/FineNet__dropout__model.h5"

    output_dir = results_dir / cfg["experiment_name"] / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    logging = init_log(output_dir)
    inference(
        deploy_set=str(data_dir),
        output_dir=str(output_dir),
        model_path=pretrain_dir,
        FineNet_path=FineNet_dir,
        file_ext=cfg["file_ext"],
        isHavingFineNet=False,
    )


if __name__ == "__main__":
    main()
