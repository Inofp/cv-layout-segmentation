import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(c_in, c_out)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, 2, stride=2)
        self.conv = ConvBlock(c_in // 2 + c_skip, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh != 0 or dw != 0:
            x = nn.functional.pad(x, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class MiniUNet(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, base: int = 32):
        super().__init__()
        self.inc = ConvBlock(in_ch, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.mid = ConvBlock(base * 8, base * 8)
        self.u3 = Up(base * 8, base * 4, base * 4)
        self.u2 = Up(base * 4, base * 2, base * 2)
        self.u1 = Up(base * 2, base, base)
        self.outc = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        xm = self.mid(x4)
        x = self.u3(xm, x3)
        x = self.u2(x, x2)
        x = self.u1(x, x1)
        return self.outc(x)