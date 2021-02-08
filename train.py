import hydra
from hydra.core.config_store import ConfigStore
from hydra_configs.pytorch_lightning.callbacks import ModelCheckpointConf

from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, AdamConfig, SGDConfig, BiDirectionalConfig, \
    UniDirectionalConfig, GCSCheckpointConfig, ConvolutionConfig
from deepspeech_pytorch.training import train

from omegaconf import OmegaConf

cs = ConfigStore.instance()
cs.store(name="config", node=DeepSpeechConfig)
cs.store(group="optim", name="sgd", node=SGDConfig)
cs.store(group="optim", name="adam", node=AdamConfig)
cs.store(group="checkpoint", name="file", node=ModelCheckpointConf)
cs.store(group="checkpoint", name="gcs", node=GCSCheckpointConfig)
cs.store(group="model", name="convolution", node=ConvolutionConfig)
cs.store(group="model", name="bidirectional", node=BiDirectionalConfig)
cs.store(group="model", name="unidirectional", node=UniDirectionalConfig)


@hydra.main(config_name="config")
def hydra_main(cfg: DeepSpeechConfig):
    train(cfg=cfg)


@hydra.main(config_name="config")
def hydra_main_debug(cfg: DeepSpeechConfig):
    update_conf = OmegaConf.load('/Users/a18249761/faang_prep/Smarsh/deepspeech.pytorch/configs/an4_convolution.yaml')
    merged_conf = OmegaConf.merge(cfg, update_conf)
    train(cfg=merged_conf)


if __name__ == '__main__':
    # hydra_main_debug()
    hydra_main()
