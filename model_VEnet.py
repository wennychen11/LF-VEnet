import torch
import torch.nn as nn


def default_conv3d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_in, n_out, kernel_size, bias=True):
        super(ResBlock, self).__init__()
        m = [nn.PReLU(n_in),
             conv(n_in, n_out, kernel_size, bias=bias)]
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class VE_conv_H(nn.Module):
    def __init__(self, conv3d=default_conv3d, n_feat=32):
        super(VE_conv_H, self).__init__()
        kernel_size = 3
        self.v_conv = ResBlock(conv3d, n_feat, n_feat, kernel_size)
        self.e_conv = ResBlock(conv3d, n_feat, n_feat, kernel_size)

    def forward(self, train_data_h):
        mid_data = self.v_conv(train_data_h)  # [N, 1, 7, H, W]
        mid_data = mid_data.permute(0, 1, 4, 3, 2)  # [N, 1, W, H, 7]
        mid_data = self.e_conv(mid_data)    # [N, 1, W, H, 7]
        out = mid_data.permute(0, 1, 4, 3, 2)  # [N, 1, 7, H, W]
        return out


class VE_conv_V(nn.Module):
    def __init__(self, conv3d=default_conv3d, n_feat=32):
        super(VE_conv_V, self).__init__()
        kernel_size = 3
        self.v_conv = ResBlock(conv3d, n_feat, n_feat, kernel_size)
        self.e_conv = ResBlock(conv3d, n_feat, n_feat, kernel_size)

    def forward(self, train_data_v):
        mid_data = self.v_conv(train_data_v)  # [N, 1, 7, H, W]
        mid_data = mid_data.permute(0, 1, 3, 2, 4)  # [N, 1, H, 7, W]
        mid_data = self.e_conv(mid_data)  # [N, 1, H, 7, W]
        out = mid_data.permute(0, 1, 3, 2, 4)  # [N, 1, 7, H, W]
        return out


class AltFilter(nn.Module):
    def __init__(self, an):
        super(AltFilter, self).__init__()
        self.an = an
        self.relu = nn.ReLU(inplace=True)
        self.spaconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.angconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        N, c, h, w = x.shape  # [N*an2, c, h, w]
        N = N // (self.an * self.an)
        out = self.relu(self.spaconv(x))  # [N*an2, c, h, w]
        out = out.view(N, self.an * self.an, c, h * w)
        out = torch.transpose(out, 1, 3)
        out = out.view(N * h * w, c, self.an, self.an)  # [N*h*w, c, an, an]

        out = self.relu(self.angconv(out))  # [N*h*w, c, an, an]
        out = out.view(N, h * w, c, self.an * self.an)
        out = torch.transpose(out, 1, 3)
        out = out.view(N * self.an * self.an, c, h, w)  # [N*an2, c, h, w]
        return out


class VEnet(nn.Module):
    def __init__(self, conv3d=default_conv3d, n_view=7, scale=2, layer_num=10):
        super(VEnet, self).__init__()
        n_mid_resblock = layer_num
        n_body = 6
        kernel_size = 3
        self.n_view = n_view
        self.scale = scale
        n_feat = 64

        m_head_h = [conv3d(1, n_feat, kernel_size)]
        m_head_v = [conv3d(1, n_feat, kernel_size)]
        m_midbody_h = [VE_conv_H(conv3d, n_feat) for _ in range(n_mid_resblock)]
        m_midbody_v = [VE_conv_V(conv3d, n_feat) for _ in range(n_mid_resblock)]
        m_scale_h = [conv3d(n_feat, 1, kernel_size)]
        m_scale_v = [conv3d(n_feat, 1, kernel_size)]
        m_body = [AltFilter(n_view) for _ in range(n_body)]
        m_tail = [
            nn.ConvTranspose2d(64, 64, kernel_size=(scale + 2, scale + 2), stride=(scale, scale), padding=(1, 1))]
        m_res_head = [
            nn.ConvTranspose2d(1, 1, kernel_size=(scale + 2, scale + 2), stride=(scale, scale), padding=(1, 1))]
        self.head_h = nn.Sequential(*m_head_h)
        self.head_v = nn.Sequential(*m_head_v)
        self.mid_body_h = nn.Sequential(*m_midbody_h)
        self.mid_body_v = nn.Sequential(*m_midbody_v)
        self.scale_h = nn.Sequential(*m_scale_h)
        self.scale_v = nn.Sequential(*m_scale_v)
        self.conv = nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.scale_body = nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.body = nn.Sequential(*m_body)
        self.res_head = nn.Sequential(*m_res_head)
        self.tail = nn.Sequential(*m_tail)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, train_data):  # [N, 49, H, W]
        n, d, h, w = train_data.shape
        train_data_h = train_data.view(n, self.n_view, self.n_view, h, w)  # [N, 7, 7, H, W]
        train_data_v = train_data_h.permute(0, 2, 1, 3, 4)  # [N, 7, 7, H, W]
        for i in range(self.n_view):
            mid_data_temp_h = self.head_h(torch.unsqueeze(train_data_h[:, i, :, :, :], 1))  # [N, 1, 7, H, W]
            mid_data_temp_h = self.mid_body_h(mid_data_temp_h)  # [N, 64, 7, H, W]
            mid_data_temp_h = self.scale_h(mid_data_temp_h)  # [N, 1, 7, H, W]

            mid_data_temp_v = self.head_v(torch.unsqueeze(train_data_v[:, i, :, :, :], 1))  # [N, 1, 7, H, W]
            mid_data_temp_v = self.mid_body_v(mid_data_temp_v)  # [N, 64, 7, H, W]
            mid_data_temp_v = self.scale_v(mid_data_temp_v)  # [N, 1, 7, H, W]
            if i == 0:
                mid_data_h = mid_data_temp_h
                mid_data_v = mid_data_temp_v
            else:
                mid_data_h = torch.cat((mid_data_h, mid_data_temp_h), 1)
                mid_data_v = torch.cat((mid_data_v, mid_data_temp_v), 1)
        mid_data_v = mid_data_v.permute(0, 2, 1, 3, 4)  # [N, 7, 7, H, W]
        mid_data_h = mid_data_h.reshape(n * d, 1, h, w)  # [N*49, 1, H, W]
        mid_data_v = mid_data_v.reshape(n * d, 1, h, w)  # [N*49, 1, H, W]
        mid_data = torch.cat((mid_data_h, mid_data_v), 1)   # [N*49, 2, H,W]
        mid_data = self.relu(self.conv(mid_data))   # [N*49, 64, H, W]
        mid_data = self.body(mid_data)   # [N*49, 64, H, W]
        mid_data = self.tail(mid_data)   # [N*49, 64, 2H, 2W]
        x = self.scale_body(mid_data)   # [N*49, 1, 2H, 2W]
        res = self.res_head(mid_data_v)   # [N*49, 1, 2H, 2W]
        x += res
        x = x.view(n, d, self.scale * h, self.scale * w)   # [N, 49, 2H, 2W]
        return x
