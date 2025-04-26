import hydra
from omegaconf.dictconfig import DictConfig

from src import run_train
from src.dataset import TGLFData
from src.model import MoE
from src.trainer import MoETrainer

@hydra.main(version_base=None, config_path="./", config_name="configs")
def main(cfg: DictConfig):
    """
    Main function to run the training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing training parameters.
    """
    # Disable Hydra logging to file
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module(config_module=None)

    # start training
    print("starting run train")
    run_train(cfg, MoE, TGLFData, MoETrainer)

if __name__ == "__main__":
    main()
