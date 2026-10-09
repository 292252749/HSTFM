import torch
import torch.nn as nn
import torch.nn.functional as F

from LibSTFv1.base.base_model import BaseModel
from LibSTFv1.model.SwinSTFM.loss import GeneratorLoss
from sorcery import dict_of

from .fem import Down
from .mfm import FineUp
import matplotlib.font_manager as fm
from io import BytesIO
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import matplotlib.pyplot as plt

class SwinSTFM(nn.Module):
    def __init__(self, bands, img_size):
        super(SwinSTFM, self).__init__()
        self.coarse_down = Down(bands, img_size)
        self.fine_down = Down(bands, img_size)
        self.fine_up = FineUp(img_size=img_size, output_bands=bands)

    def forward(self, c0, f0, c1):
        coarse_fea0 = self.coarse_down(c0)
        coarse_fea1 = self.coarse_down(c1)
        fine_fea = self.fine_down(f0)
        diff_fea1 = []
        for i in range(5):
            diff_fea1.append(coarse_fea1[i] - coarse_fea0[i])

        diff_fea0 = []
        for i in range(5):
            diff_fea0.append(fine_fea[i] - coarse_fea0[i])

        output_fine = self.fine_up(coarse_fea0, diff_fea1, fine_fea, coarse_fea1)

        return output_fine
    

plt.rcParams['mathtext.fontset'] = 'cm'
sns.set(style="ticks", palette="bright")


class SwinSTFMModel(BaseModel):
    def __init__(self,
                 epochs,
                 bands,
                 rgb_c,
                 dataname
                 ):
        super().__init__(
                 epochs,
                 bands,
                 rgb_c,
                 dataname
                 )
        self.automatic_optimization = False

        self.rgb_c = rgb_c
        self.model = SwinSTFM(bands, 512)
        
        self.cri_pix = GeneratorLoss()

        self.bands = bands

        self.reset_metrics()
        self.save_hyperparameters()


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs, eta_min=1e-7) # original: ReduceLROnPlateau
        return [opt], [scheduler]

    def forward(self, LR_t1, LR_t2, HR_t1, HR_t2=None):
        pred = self.model(LR_t1, HR_t1, LR_t2)
        out = dict_of(pred)
        return out

    def training_step(self, batch, batch_idx):
        LR_t1, LR_t2, HR_t1, gt = batch["LR_t1"], batch["LR_t2"], batch["HR_t1"], batch["HR_t2"]
        opt = self.optimizers()
        opt.zero_grad()

        out = self.forward(LR_t1, LR_t2, HR_t1) 
        pred = out["pred"]

        l_total = self.cri_pix(pred, gt, is_ds=False)

        self.manual_backward(l_total)
        opt.step()

        # 记录日志
        log_dict = {
            "loss": l_total.item(),
            "mse": F.mse_loss(pred.detach(), gt).item()
        }

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)


    def on_train_epoch_end(self):
        sche_pf = self.lr_schedulers()
        sche_pf.step()
