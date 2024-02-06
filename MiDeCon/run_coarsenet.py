import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs")
def main(cfg : DictConfig) -> None:
    print(cfg["data_dir"])
    

if __name__ == "__main__":
    main()
