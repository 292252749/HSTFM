import inspect
import os
import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics.functional.image as MF
from torch.optim.lr_scheduler import StepLR
from torchmetrics.functional.regression import r2_score

from LibSTFv1.loss.l1_loss import l1_loss
from LibSTFv1.metric.cross_correlation import cross_correlation
from LibSTFv1.util.misc import regularize_inputs
from sorcery import dict_of


class BaseModel(pl.LightningModule):
    def __init__(self, epochs, bands, rgb_c, dataname):
        super().__init__()
        self.automatic_optimization = False
        self.rgb_c = rgb_c
        self.model = None
        self.loss = l1_loss
        self.dataname = dataname
        self.bands = bands
        self.spatial_ratio = 16.0
        self.save_hyperparameters()
        self.reset_metrics("val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return [opt], [StepLR(opt, step_size=100, gamma=0.8)]

    def forward(self, LR_t1, LR_t2, HR_t1, HR_t2=None):
        pred = self.model(LR_t1, LR_t2, HR_t1)
        return dict_of(pred)

    def _predict_batch(self, batch):
        lr_t1 = batch["LR_t1"]
        lr_t2 = batch["LR_t2"]
        hr_t1 = batch["HR_t1"]
        forward_params = inspect.signature(self.forward).parameters
        if "mask" in forward_params:
            return self.forward(lr_t1, lr_t2, hr_t1, batch["mask"])["pred"]
        return self.forward(lr_t1, lr_t2, hr_t1)["pred"]

    def training_step(self, batch, batch_idx):
        lr_t1, lr_t2, hr_t1, gt = batch["LR_t1"], batch["LR_t2"], batch["HR_t1"], batch["HR_t2"]
        out = self.forward(lr_t1, lr_t2, hr_t1, gt)
        pred = out["pred"]
        opt = self.optimizers()
        opt.zero_grad()
        total_loss, log_dict = self.loss(gt, pred)
        self.manual_backward(total_loss)
        opt.step()
        log_dict["lr"] = opt.param_groups[0]["lr"]
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        self.lr_schedulers().step()

    def validation_step(self, batch, batch_idx):
        pred = self._predict_batch(batch)
        pred, gt = regularize_inputs(pred, batch["HR_t2"])
        self.record_full_reference_metrics(pred, gt, split="val")

    def on_validation_epoch_start(self):
        self.reset_metrics("val")

    def on_validation_epoch_end(self):
        self.write_metric_summary("val", filename="val_metrics.csv")

    def test_step(self, batch, batch_idx):
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        pred = self._predict_batch(batch)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        pred, gt = regularize_inputs(pred, batch["HR_t2"])
        self.record_full_reference_metrics(pred, gt, split="test")
        self.record_metrics("Time", torch.tensor(elapsed, device=pred.device), split="test")

    def on_test_epoch_start(self):
        self.reset_metrics("test")

    def on_test_epoch_end(self):
        self.write_metric_summary("test", filename="test_metrics.csv")

    def record_full_reference_metrics(self, pred, gt, split="val"):
        data_range = (0.0, 1.0)
        self.record_metrics("MAE", F.l1_loss(pred, gt), split)
        self.record_metrics("RMSE", torch.sqrt(F.mse_loss(pred, gt)), split)
        self._record_metric_call("SSIM", lambda: MF.structural_similarity_index_measure(pred, gt, data_range=data_range), split)
        self._record_metric_call("ERGAS", lambda: MF.error_relative_global_dimensionless_synthesis(pred, gt) / float(self.spatial_ratio), split)
        self._record_metric_call("SAM", lambda: MF.spectral_angle_mapper(pred, gt), split)
        self._record_metric_call("RASE", lambda: MF.relative_average_spectral_error(pred, gt), split)
        self._record_metric_call("PSNR", lambda: MF.peak_signal_noise_ratio(pred, gt, data_range=data_range), split)
        self._record_metric_call("UQI", lambda: MF.universal_image_quality_index(pred, gt), split)
        self._record_metric_call("SCC", lambda: MF.spatial_correlation_coefficient(pred, gt), split)
        self._record_metric_call("CC", lambda: cross_correlation(pred, gt), split)
        self._record_metric_call("R2", lambda: r2_score(pred.reshape(-1), gt.reshape(-1)), split)

    def _record_metric_call(self, name, fn, split):
        try:
            self.record_metrics(name, fn(), split)
        except Exception as exc:
            if self.trainer is None or self.trainer.is_global_zero:
                print(f"Warning: skip metric {split}/{name}: {exc}")

    def reset_metrics(self, split="val"):
        names = ["MAE", "SCC", "SAM", "RMSE", "ERGAS", "PSNR", "SSIM", "RASE", "UQI", "CC", "R2", "Time"]
        self.eval_metrics = [f"{split}/{name}" for name in names]
        self.metrics_all = {metric: [] for metric in self.eval_metrics}

    def record_metrics(self, name, value, split="val"):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=self.device)
        if not torch.isfinite(value).all():
            return
        key = f"{split}/{name}"
        self.metrics_all.setdefault(key, []).append(float(value.detach().mean().cpu()))

    def write_metric_summary(self, split, filename):
        result = {"method": self.__class__.__name__, "epoch": self.current_epoch, "split": split}
        log_values = {}
        for key, values in self.metrics_all.items():
            if not values:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values))
            result[f"{key}_mean"] = mean
            result[f"{key}_std"] = std
            log_values[f"{key}_mean"] = mean
        if log_values:
            self.log_dict(log_values, prog_bar=False, logger=True, sync_dist=True)
        if self.logger is not None and (self.trainer is None or self.trainer.is_global_zero):
            log_dir = getattr(self.trainer, "default_root_dir", None) if self.trainer is not None else None
            if not log_dir:
                logger_obj = self.logger
                if hasattr(logger_obj, "loggers") and logger_obj.loggers:
                    logger_obj = logger_obj.loggers[0]
                log_dir = getattr(logger_obj, "log_dir", None) or getattr(logger_obj, "save_dir", ".")
            os.makedirs(log_dir, exist_ok=True)
            csv_path = os.path.join(log_dir, filename)
            pd.DataFrame([result]).to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))
