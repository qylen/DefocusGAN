import torch 
import torch.nn as nn 
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""


    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down4(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(4),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Up4(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
        DoubleConv(in_channels, out_channels,in_channels),
        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),)

    def forward(self, x1):
        x1 = self.up(x1)
        return x1

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels*2, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels*2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConvdepth(nn.Module):
    def __init__(self, in_channels, out_channels,middle_channels):
        super(OutConvdepth, self).__init__()

        self.out1=nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.out2 =nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1=self.out1(x)
        x2=self.out2(x1)
        return x2,x1 

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.out=nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        return self.conv(x)
# Parallax-Attention Block

class PAB(nn.Module):
    def __init__(self, channels):
        super(PAB, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.query = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(channels),
        )
        self.key = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x_left, x_right, cost,maxdisp):
        '''
        :param x_left:      features from the left image  (B * C * H * W)
        :param x_right:     features from the right image (B * C * H * W)
        :param cost:        input matching cost           (B * H * W * W)
        '''
        b, c, h, w = x_left.shape
        fea_left = self.head(x_left)
        fea_right = self.head(x_right)

        # C_right2left
        Q = self.query(fea_left).permute(0, 2, 3, 1).contiguous()                     # B * H * W * C
        K = self.key(fea_right).permute(0, 2, 1, 3) .contiguous()                     # B * H * C * W
        cost_right2left = torch.matmul(Q, K) / c                                      # scale the matching cost
        cost_right2left = cost_right2left[:,:,:,:maxdisp]
        cost_right2left = cost_right2left + cost                                      #B *H*W*D


        return x_left + fea_left, \
               x_right + fea_right, \
               cost_right2left
class PAMA(nn.Module):
    def __init__(self, channels):
        super(PAMA, self).__init__()
        self.bq = nn.Conv2d(channels, channels, 1, 1, 0, groups=4, bias=True)
        self.bs = nn.Conv2d(channels, channels, 1, 1, 0, groups=4, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(channels)
        self.bn = nn.BatchNorm2d(channels)

    def __call__(self, x_left, x_right,cost,maxdisp):
        b, c0, h0, w0 = x_left.shape
        Q = self.bq(self.rb(self.bn(x_left)))
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(x_right)))
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))                    # (B*H) * C * Wr
        M_right_to_left = self.softmax(score)                                                   # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(score.permute(0, 2, 1))                                  # (B*H) * Wr * Wl
        
        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                              #  B, C0, H0, W0
        
        cost1 =(M_right_to_left.view(b,h0,w0,w0))[:,:,:,:maxdisp]+cost
        return x_leftT,x_rightT,cost1


class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x

class PAM_stage(nn.Module):
    def __init__(self, channels):
        super(PAM_stage, self).__init__()
        self.pab1 = PAB(channels)
        self.pab2 = PAB(channels)


    def forward(self, fea_left, fea_right, cost,maxdisp):
        fea_left, fea_right, cost = self.pab1(fea_left, fea_right, cost,maxdisp)
        fea_left, fea_right, cost = self.pab2(fea_left, fea_right, cost,maxdisp)
        #我们应该要截断代价卷，只取视差在固定范围的值

        return fea_left, fea_right, cost


class Disp(nn.Module):
    def __init__(self, maxdisp=40):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)

    def forward(self, x):       
        x=x.permute(0,3,1,2)
        #print(x.shape)
        # x = F.interpolate(x, [self.maxdisp, x.size()[2], x.size()[3]], mode='trilinear', align_corners=False)
        # # print('2.1', x.size())
        # x = torch.squeeze(x, 1)
        # print('2.2', x.size()) (4, 12, 192, 192)
        x = self.softmax(x)
        # print('2.3', x.size()) (4, 12, 192, 192)
        x = self.disparity(x)
        # print('2.4', x.size()) (4, 192, 192)
        xA=torch.unsqueeze(x,1)
        #print(xA.shape)
        return x,xA
class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = torch.reshape(torch.arange(0, self.maxdisp, device=torch.cuda.current_device(), dtype=torch.float32),[1,self.maxdisp,1,1])
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out