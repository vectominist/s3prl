from typing import Dict, Tuple, Union

import torch
from torch import nn, Tensor


def tivf_loss(
    c: Tensor,
    s: Tensor,
    gamma_c: float = 1.0,
    gamma_s_avg: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, Tensor]:

    b_sz, t_sz, d_c = c.shape
    d_s = s.shape[2]

    loss_c = torch.clamp_min(gamma_c - torch.sqrt(torch.var(c, dim=1) + eps), 0).mean()
    # loss_s = torch.sqrt(torch.var(s, dim=1) + eps).mean()
    loss_s = torch.var(s, dim=1).mean()
    loss_s_avg = torch.clamp_min(
        gamma_s_avg - torch.sqrt(torch.var(s.mean(1), dim=0) + eps), 0
    ).mean()
    loss_i = (
        torch.bmm(
            (c.view(b_sz * t_sz, d_c) - c.mean(1).mean(0).unsqueeze(0)).unsqueeze(2),
            (s.view(b_sz * t_sz, d_s) - s.mean(1).mean(0).unsqueeze(0)).unsqueeze(1),
        )
        .mean(0)
        .pow(2)
        .mean()
    )

    return {
        "loss_c": loss_c,
        "loss_s": loss_s,
        "loss_s_avg": loss_s_avg,
        "loss_i": loss_i,
    }


class LTIVF(nn.Module):
    def __init__(
        self,
        d: int,
        d_c: int,
        d_s: int,
        w_c: float = 1.0,
        w_s: float = 1.0,
        w_s_avg: float = 1.0,
        w_i: float = 1.0,
        gamma_c: float = 1.0,
        gamma_s_avg: float = 1.0,
        eps: float = 1e-5,
        batch_norm: bool = False,
        avg_s: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.d = d
        self.d_c = d_c
        self.d_s = d_s
        self.d_cs = d_c + d_s

        self.w_c = w_c
        self.w_s = w_s
        self.w_s_avg = w_s_avg
        self.w_i = w_i
        self.gamma_c = gamma_c
        self.gamma_s_avg = gamma_s_avg
        self.eps = eps
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm1d(d)
        self.avg_s = avg_s

        A = torch.randn((d, self.d_cs))
        B = torch.linalg.pinv(A)
        self.A_mat = nn.Parameter(A)
        self.B_mat = nn.Parameter(B + 0.02 * torch.randn_like(B))

    def disentangle(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # x: (B, T, d)
        if self.bn:
            x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        cs = torch.einsum("ij,btjk->btik", self.B_mat, x.unsqueeze(-1))
        # cs: (B, T, d_c + d_s, 1)
        c = cs[:, :, : self.d_c, 0]
        s = cs[:, :, -self.d_s :, 0]

        return cs, c, s

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        cs, c, s = self.disentangle(x)
        if self.avg_s:
            cs[:, :, -self.d_s :, :] = cs[:, :, -self.d_s :, :].mean(1, keepdim=True)
        x_rec = torch.einsum("ij,btjk->btik", self.A_mat, cs).squeeze(-1)

        return x_rec, c, s

    def compute_rec_loss(
        self,
        x: Tensor,
        x_rec: Tensor,
    ) -> Tensor:
        loss_r = (
            (self.A_mat @ self.B_mat - torch.eye(self.d, device=self.A_mat.device))
            .pow(2)
            .mean()
        ) * 0.5 + (x - x_rec).pow(2).mean() * 0.5
        return loss_r

    def compute_loss(
        self,
        x: Tensor,
        x_rec: Tensor,
        c: Tensor,
        s: Tensor,
    ) -> Dict[str, Tensor]:
        loss_r = self.compute_rec_loss(x, x_rec)
        losses = tivf_loss(
            c,
            s,
            self.gamma_c,
            self.gamma_s_avg,
            self.eps,
        )

        loss = (
            loss_r
            + losses["loss_c"] * self.w_c
            + losses["loss_s"] * self.w_s
            + losses["loss_s_avg"] * self.w_s_avg
            + losses["loss_i"] * self.w_i
        )

        losses["loss_r"] = loss_r
        losses["loss"] = loss

        return losses


def load_pretrained(path: str, device: Union[str, torch.device] = "cpu") -> LTIVF:
    ckpt = torch.load(path, map_location=device)
    model = LTIVF(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model
