import torch.nn as nn
from PIL import Image


def get_valid_patch_sizes():
    return list((572 - (16*i) for i in range(31)))

class DownBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels*2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*2,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels*2)
        )
        self.conv1x1 = nn.Sequential(
            # down sample channels again.
            nn.Conv2d(in_channels*2, in_channels,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        out1 = self.pool(x)
        out2 = self.conv1(out1)
        out3 = self.conv2(out2)
        out4 = self.conv1x1(out3)
        return out4 + out1


def crop_tensor(tensor, target):
    """ Crop tensor to target size """
    _, _, tensor_height, tensor_width = tensor.size()
    _, _, crop_height, crop_width = target.size()
    left = (tensor_width - crop_height) // 2
    top = (tensor_height - crop_width) // 2
    right = left + crop_width
    bottom = top + crop_height
    cropped_tensor = tensor[:, :, top: bottom, left: right]
    return cropped_tensor


class UpBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels,
                               kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )

    def forward(self, x, down_out):
        out = self.conv1(x)
        cropped = crop_tensor(down_out, out)
        out = cropped + out # residual
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class UNetGNRes(nn.Module):
    def __init__(self, im_channels=3):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(im_channels, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, 64),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, 64)
            # now at 568 x 568, 64 channels
        )
        self.down1 = DownBlock(64)
        self.down2 = DownBlock(64)
        self.down3 = DownBlock(64)
        self.down4 = DownBlock(64)
        self.up1 = UpBlock(64)
        self.up2 = UpBlock(64)
        self.up3 = UpBlock(64)
        self.up4 = UpBlock(64)
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.GroupNorm(2, 2)
        )

    def forward(self, x):
        out1 = self.conv_in(x)
        out2 = self.down1(out1)
        out3 = self.down2(out2)
        out4 = self.down3(out3)
        out5 = self.down4(out4)
        out = self.up1(out5, out4)
        out = self.up2(out, out3)
        out = self.up3(out, out2)
        out = self.up4(out, out1)
        out = self.conv_out(out)
        return out


if __name__ == '__main__':
    import torch
    from torch.nn.functional import softmax
    from skimage.io import imsave
    import numpy as np
    unet = UNetGNRes()
    unet.eval()
    # test_input = np.random.rand(1, 3, 572, 572)
    test_input = np.zeros((1, 3, 572, 572))
    test_input = torch.from_numpy(test_input)
    if torch.cuda.is_available():
        test_input.cuda()
    test_input = test_input.float()
    output = unet(test_input)
    output = output.detach()
    print('output.shape', output.shape)
    softmaxed = softmax(output, 1)[:, 1, :] # just fg probability
    softmaxed = softmaxed[0] # single image.
    print('softmaxed shape = ', softmaxed.shape)

    im = Image.fromarray(np.array(softmaxed) * 256)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save('out.png')