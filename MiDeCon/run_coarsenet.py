import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

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


@dataclass
class Config:
    data_dir: str = MISSING
    file_ext: str = ".bmp"
    cuda_visible_devices: Optional[list] = None


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(version_base=None, config_path="configs", config_name="casia-coarsenet")
def main(cfg: DictConfig) -> None:
    # print(cfg)
    data_dir = Path(cfg["data_dir"])
    assert data_dir.exists()
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    gpu_list = cfg["cuda_visible_devices"]
    if isinstance(gpu_list, (list, ListConfig)):
        if len(gpu_list) == 1:
            gpu_devices = str(gpu_list[0])
        else:
            gpu_devices = str(gpu_list)
    else:
        gpu_devices = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    # print(os.environ["CUDA_VISIBLE_DEVICES"])

    print(f"Hydra output directory  : {output_dir}")

    tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    pretrain_dir = "Models/CoarseNet.h5"
    FineNet_dir = "output_FineNet/FineNet_dropout/FineNet__dropout__model.h5"

    # output_dir = results_dir / cfg["experiment_name"] / datetime.now().strftime("%Y%m%d-%H%M%S")
    # output_dir.mkdir(parents=True, exist_ok=False)

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
