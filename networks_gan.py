from __future__ import annotations

import functools
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class VAE(nn.Module):
    """Variational autoencoder for ligand shapes."""

    def __init__(self,
                 nc: int = 5,
                 ngf: int = 128,
                 ndf: int = 128,
                 latent_variable_size: int = 512) -> None:

        super().__init__()
        self.nc  = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        hidden_dim = ndf * 4
        flat_dim = hidden_dim * 3 * 3 * 3

        # Encoder
        self.e1  = nn.Conv3d(nc, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.e2  = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.e3  = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.e4  = nn.Conv3d(64, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(hidden_dim)
        self.e5  = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(hidden_dim)

        self.fc_mu     = nn.Linear(flat_dim, latent_variable_size)
        self.fc_logvar = nn.Linear(flat_dim, latent_variable_size)

        # Decoder
        self.d1  = nn.Linear(latent_variable_size, flat_dim)
        self.d2  = nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm3d(hidden_dim, eps=1e-3)
        self.d3  = nn.ConvTranspose3d(hidden_dim, ndf * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn7 = nn.BatchNorm3d(ndf * 2, eps=1e-3)
        self.d4  = nn.Conv3d(ndf * 2, ndf, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm3d(ndf, eps=1e-3)
        self.d5  = nn.ConvTranspose3d(ndf, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn9 = nn.BatchNorm3d(32, eps=1e-3)
        self.d6  = nn.Conv3d(32, nc, kernel_size=3, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu       = nn.ReLU(inplace=True)
        self.sigmoid    = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.leaky_relu(self.bn1(self.e1(x)))
        h = self.leaky_relu(self.bn2(self.e2(h)))
        h = self.leaky_relu(self.bn3(self.e3(h)))
        h = self.leaky_relu(self.bn4(self.e4(h)))
        h = self.leaky_relu(self.bn5(self.e5(h)))
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Keep original misspelled method name for backward compatibility.
    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return self.reparameterize(mu, logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.d1(z))
        h = h.view(z.size(0), self.ndf * 4, 3, 3, 3)
        h = self.leaky_relu(self.bn6(self.d2(h)))
        h = self.leaky_relu(self.bn7(self.d3(h)))
        h = self.leaky_relu(self.bn8(self.d4(h)))
        h = self.leaky_relu(self.bn9(self.d5(h)))
        return self.sigmoid(self.d6(h))

    def get_latent_var(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class EncoderCNN(nn.Module):
    """CNN encoder that maps ligand shape voxels into a vector representation."""

    def __init__(self, in_layers: int = 5) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

        layers: List[nn.Module] = []
        out_layers = 32
        for i in range(8):
            layers.extend([
                nn.Conv3d(in_layers, out_layers, kernel_size=3, bias=False, padding=1),
                nn.BatchNorm3d(out_layers),
                nn.ReLU(inplace=True)
            ])
            in_layers = out_layers
            if (i + 1) % 2 == 0:
                out_layers *= 2
                layers.append(self.pool)

        layers.pop()  # remove last pooling layer
        self.network = nn.Sequential(*layers)
        self.fc1 = nn.Linear(256, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = x.mean(dim=(2, 3, 4))

        return self.relu(self.fc1(x))


class DecoderRNN(nn.Module):
    """LSTM decoder that maps shape features into SMILES tokens."""

    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor, captions: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        packed = pack_padded_sequence(
            embeddings,
            lengths,
            batch_first=True,
            enforce_sorted=False)
        hiddens, _ = self.lstm(packed)
        return self.linear(hiddens.data)

    @torch.no_grad()
    def sample(self, features: torch.Tensor, states=None, max_len: int = 62) -> List[torch.Tensor]:
        sampled_ids: List[torch.Tensor] = []
        inputs = features.unsqueeze(1)
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(dim=1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted).unsqueeze(1)
        return sampled_ids

    @torch.no_grad()
    def sample_prob(self, features: torch.Tensor, states=None, max_len: int = 62) -> List[torch.Tensor]:
        sampled_ids: List[torch.Tensor] = []
        inputs = features.unsqueeze(1)

        for step in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            if step == 0:
                predicted = outputs.argmax(dim=1)
            else:
                probs = F.softmax(outputs, dim=1)
                predicted = torch.multinomial(probs, num_samples=1).squeeze(1)

            sampled_ids.append(predicted)
            inputs = self.embed(predicted).unsqueeze(1)
        return sampled_ids


class G_Unet_add_all3D(nn.Module):
    """Generator network of 3D-BicycleGAN."""

    def __init__(self,
                 input_nc: int = 7,
                 output_nc: int = 5,
                 nz: int = 8,
                 num_downs: int = 8,
                 ngf: int = 64,
                 norm_type: Optional[str] = None,
                 nl_layer=None,
                 use_dropout: bool = False,
                 gpu_ids: Optional[List[int]] = None,
                 upsample: str = "basic") -> None:

        super().__init__()
        self.gpu_ids = gpu_ids or []
        self.nz = nz

        unet_block = UnetBlock_with_z3D(
            ngf * 8, ngf * 8, ngf * 8, nz,
            submodule=None,
            innermost=True,
            norm_type=norm_type,
            nl_layer=nl_layer,
            upsample=upsample,
            stride=1)

        unet_block = UnetBlock_with_z3D(
            ngf * 8, ngf * 8, ngf * 8, nz,
            submodule=unet_block,
            norm_type=norm_type,
            nl_layer=nl_layer,
            use_dropout=use_dropout,
            upsample=upsample,
            stride=1)

        for i in range(num_downs - 6):
            if i == 0:
                stride = 2
            elif i == 1:
                stride = 1
            else:
                raise NotImplementedError("num_downs is too large for this architecture")

            unet_block = UnetBlock_with_z3D(
                ngf * 8, ngf * 8, ngf * 8, nz,
                submodule=unet_block,
                norm_type=norm_type,
                nl_layer=nl_layer,
                use_dropout=use_dropout,
                upsample=upsample,
                stride=stride)

        unet_block = UnetBlock_with_z3D(
            ngf * 4, ngf * 4, ngf * 8, nz,
            submodule=unet_block,
            norm_type=norm_type,
            nl_layer=nl_layer,
            upsample=upsample,
            stride=2)

        unet_block = UnetBlock_with_z3D(
            ngf * 2, ngf * 2, ngf * 4, nz,
            submodule=unet_block,
            norm_type=norm_type,
            nl_layer=nl_layer,
            upsample=upsample,
            stride=1)

        unet_block = UnetBlock_with_z3D(
            ngf, ngf, ngf * 2, nz,
            submodule=unet_block,
            norm_type=norm_type,
            nl_layer=nl_layer,
            upsample=upsample,
            stride=2)

        self.model = UnetBlock_with_z3D(
            input_nc, output_nc, ngf, nz,
            submodule=unet_block,
            outermost=True,
            norm_type=norm_type,
            nl_layer=nl_layer,
            upsample=upsample,
            stride=1)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.model(x, z)


class UnetBlock_with_z3D(nn.Module):
    """Recursive 3D U-Net block with latent z concatenated to the input."""

    def __init__(self,
                 input_nc: int,
                 outer_nc: int,
                 inner_nc: int,
                 nz: int = 0,
                 submodule: Optional[nn.Module] = None,
                 outermost: bool = False,
                 innermost: bool = False,
                 norm_type: Optional[str] = None,
                 nl_layer=None,
                 use_dropout: bool = False,
                 upsample: str = "basic",
                 padding_type: str = "zero",
                 stride: int = 2) -> None:

        super().__init__()

        if padding_type != "zero":
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        norm_layer = get_norm_layer(norm_type)
        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        self.submodule = submodule

        downconv: List[nn.Module] = [
            nn.Conv3d(input_nc + nz, inner_nc, kernel_size=3, stride=stride, padding=1)
        ]
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        uprelu = nn.ReLU(inplace=True)

        if outermost:
            upconv = upsample_layer(inner_nc * 2, outer_nc, upsample=upsample, stride=stride)
            down = downconv
            up = [uprelu] + upconv + [nn.Sigmoid()]
        elif innermost:
            upconv = upsample_layer(inner_nc, outer_nc, upsample=upsample, stride=stride)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up.append(norm_layer(outer_nc))
        else:
            upconv = upsample_layer(inner_nc * 2, outer_nc, upsample=upsample, stride=stride)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down.append(norm_layer(inner_nc))
            up = [uprelu] + upconv
            if norm_layer is not None:
                up.append(norm_layer(outer_nc))
            if use_dropout:
                up.append(nn.Dropout(0.5))

        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3), x.size(4)
            )
            x_and_z = torch.cat([x, z_img.to(device=x.device, dtype=x.dtype)], dim=1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            if self.submodule is None:
                raise RuntimeError("outermost block requires a submodule")
            x2 = self.submodule(x1, z)
            return self.up(x2)

        if self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], dim=1)

        x1 = self.down(x_and_z)
        if self.submodule is None:
            raise RuntimeError("intermediate block requires a submodule")
        x2 = self.submodule(x1, z)
        return torch.cat([self.up(x2), x], dim=1)


class D_N3Dv1LayersMulti(nn.Module):
    """Multi-scale 3D PatchGAN discriminator."""

    def __init__(self,
                 input_nc: int,
                 ndf: int = 64,
                 norm_type: str = "instance",
                 use_sigmoid: bool = False,
                 gpu_ids: Optional[List[int]] = None,
                 num_D: int = 1) -> None:
        super().__init__()
        self.gpu_ids = gpu_ids or []
        self.num_D = num_D

        if num_D == 1:
            self.model = nn.Sequential(*self.get_layers(input_nc, ndf, norm_type, use_sigmoid))
        else:
            models = []
            for i in range(num_D):
                cur_ndf = int(round(ndf / (2 ** i)))
                models.append(nn.Sequential(*self.get_layers(input_nc, cur_ndf, norm_type, use_sigmoid)))
            self.model = nn.ModuleList(models)
            self.down = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def get_layers(self,
                   input_nc: int,
                   ndf: int = 64,
                   norm_type: str = "instance",
                   use_sigmoid: bool = False) -> List[nn.Module]:

        norm_layer = get_norm_layer(norm_type)
        if norm_layer is None:
            raise ValueError("Discriminator requires a normalization layer. Use 'batch' or 'instance'.")

        kw, padw = 3, 1
        sequence: List[nn.Module] = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, 5):
            use_stride = 2 if n in (1, 3) else 1
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence.extend([
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=use_stride, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        nf_mult_prev = nf_mult
        nf_mult = 8
        sequence.extend([
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ])

        if use_sigmoid:
            sequence.append(nn.Sigmoid())
        return sequence

    def parallel_forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gpu_ids and x.is_cuda:
            return nn.parallel.data_parallel(model, x, self.gpu_ids)
        return model(x)

    def forward(self, x: torch.Tensor):
        if self.num_D == 1:
            return self.parallel_forward(self.model, x)

        results = []
        down = x
        for i, model in enumerate(self.model):
            results.append(self.parallel_forward(model, down))
            if i != self.num_D - 1:
                down = self.down(down)
        return results


class E_3DNLayers(nn.Module):
    """3D CNN encoder used in 3D-BicycleGAN."""

    def __init__(self,
                 input_nc: int = 5,
                 output_nc: int = 8,
                 ndf: int = 64,
                 norm_type: str = "instance",
                 nl_layer: Optional[str] = None,
                 gpu_ids: Optional[List[int]] = None,
                 vaeLike: bool = False) -> None:

        super().__init__()
        self.gpu_ids = gpu_ids or []
        self.vaeLike = vaeLike

        kw, padw = 3, 1
        sequence: List[nn.Module] = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw)
        ]
        sequence.extend(fetch_simple_block3d(ndf, ndf * 2, nl=nl_layer, norm_type=norm_type))
        sequence.append(nn.AvgPool3d(2, 2))
        sequence.extend(fetch_simple_block3d(ndf * 2, ndf * 2, nl=nl_layer, norm_type=norm_type))
        sequence.append(nn.AvgPool3d(2, 2))
        sequence.extend(fetch_simple_block3d(ndf * 2, ndf * 4, nl=nl_layer, norm_type=norm_type))
        sequence.append(nn.AvgPool3d(2, 2))
        sequence.extend(fetch_simple_block3d(ndf * 4, ndf * 4, nl=nl_layer, norm_type=norm_type))
        sequence.append(nn.AvgPool3d(3))

        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Linear(ndf * 4, output_nc)
        if vaeLike:
            self.fcVar = nn.Linear(ndf * 4, output_nc)

    def forward(self, x: torch.Tensor):
        x_conv = self.conv(x)
        conv_flat = torch.flatten(x_conv, start_dim=1)
        output = self.fc(conv_flat)

        if self.vaeLike:
            return output, self.fcVar(conv_flat)
        return output


# Backward-compatible alias for original camelCase name.
def upsampleLayer(inplanes: int, outplanes: int, upsample: str = "basic", padding_type: str = "zero", stride: int = 2):
    return upsample_layer(inplanes, outplanes, upsample=upsample, padding_type=padding_type, stride=stride)


def upsample_layer(inplanes: int,
                   outplanes: int,
                   upsample: str = "basic",
                   padding_type: str = "zero",
                   stride: int = 2) -> List[nn.Module]:

    if padding_type != "zero":
        raise NotImplementedError(f"padding [{padding_type}] is not implemented")
    if upsample != "basic":
        raise NotImplementedError(f"upsample layer [{upsample}] is not implemented")

    return [nn.ConvTranspose3d(inplanes, outplanes,kernel_size=3,
                               stride=stride, padding=1, output_padding=1 if stride == 2 else 0)]


def fetch_simple_block3d(in_lay: int,
                         out_lay: int,
                         nl: Optional[str],
                         norm_type: str,
                         stride: int = 1,
                         kw: int = 3,
                         padw: int = 1) -> List[nn.Module]:

    norm_layer = get_norm_layer(norm_type)
    if norm_layer is None:
        raise ValueError("fetch_simple_block3d requires norm_type='batch' or 'instance'.")

    layers: List[nn.Module] = [
        nn.Conv3d(in_lay, out_lay, kernel_size=kw, stride=stride, padding=padw)
    ]
    if nl == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif nl is not None:
        raise NotImplementedError(f"nonlinearity [{nl}] is not implemented in simple block")
    layers.append(norm_layer(out_lay))
    return layers


def get_non_linearity(layer_type: str) -> nn.Module:
    if layer_type == "relu":
        return nn.ReLU(inplace=True)
    if layer_type == "lrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    if layer_type == "elu":
        return nn.ELU(inplace=True)
    raise NotImplementedError(f"nonlinearity activation [{layer_type}] is not found")


def get_norm_layer(norm_type: Optional[str] = "instance"):
    """Return a 3D normalization layer factory."""
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    if norm_type in ("none", None):
        return None
    raise NotImplementedError(f"normalization layer [{norm_type}] is not found")
