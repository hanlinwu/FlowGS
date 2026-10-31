import os
import time

import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True)
        OmegaConf.save(self.config, os.path.join(self.cfgdir, f"{self.now}-project.yaml"))
        OmegaConf.save(
            OmegaConf.create({"lightning": self.lightning_config}),
            os.path.join(self.cfgdir, f"{self.now}-lightning.yaml"),
        )


class ImageLogger(Callback):
    def __init__(self, batch_frequency=1000, max_images=4, clamp=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.clamp = clamp

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if not hasattr(pl_module, "log_images") or batch_idx % self.batch_freq != 0:
            return
        was_training = pl_module.training
        if was_training:
            pl_module.eval()
        with torch.no_grad():
            images = pl_module.log_images(batch, split=split)
        for key, value in images.items():
            if not isinstance(value, torch.Tensor):
                continue
            value = value[: self.max_images].detach().cpu()
            if self.clamp:
                value = value.clamp(-1.0, 1.0)
            grid = torch.cat([image for image in value], dim=-1)
            if hasattr(pl_module.logger, "experiment"):
                pl_module.logger.experiment.add_image(
                    f"{split}/{key}", grid, global_step=pl_module.global_step
                )
        if was_training:
            pl_module.train()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.log_img(pl_module, batch, batch_idx, split="train")


class CUDACallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if not torch.cuda.is_available():
            return
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        max_memory = torch.cuda.max_memory_allocated() / 2**20
        epoch_time = time.time() - getattr(self, "start_time", time.time())
        print(f"Epoch time: {epoch_time:.2f}s | peak memory: {max_memory:.2f} MiB")
