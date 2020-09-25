import torch
import torch.nn as nn


class Conv3x3(nn.Sequential):
    def __init__(self, inp, oup, stride, pad, relu=True):
        if relu:
            relu = nn.ReLU(True)
        else:
            relu = nn.Identity()
        super(Conv3x3, self).__init__(
            nn.Conv2d(inp, oup, 3, stride, pad, bias=False),
            nn.BatchNorm2d(oup),
            relu
        )


class Conv1x1(nn.Sequential):
    def __init__(self, inp, oup, stride, pad, relu=True):
        if relu:
            relu = nn.ReLU(True)
        else:
            relu = nn.Identity()
        super(Conv1x1, self).__init__(
            nn.Conv2d(inp, oup, 1, stride, pad, bias=False),
            nn.BatchNorm2d(oup),
            relu
        )


class StemBlock(nn.Module):
    def __init__(self, inp):
        super(StemBlock, self).__init__()
        self.conv1 = Conv3x3(inp, 32, 2, 1)
        self.conv2 = nn.Sequential(
            Conv1x1(32, 16, 1, 0),
            Conv3x3(16, 32, 2, 1)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Conv1x1(64, 32, 1, 0)

    def forward(self, x):
        feat = self.conv1(x)
        feat_ = self.conv2(feat)
        feat = torch.cat([self.max_pool(feat), feat_], dim=1)
        feat = self.conv3(feat)
        return feat


class TwoWayDenseBlock(nn.Module):
    def __init__(self, inp, growth_rate, inter_ch):
        super(TwoWayDenseBlock, self).__init__()
        self.left = nn.Sequential(
            Conv1x1(inp, inter_ch, 1, 0),
            Conv3x3(inter_ch, growth_rate//2, 1, 1)
        )
        self.right = nn.Sequential(
            Conv1x1(inp, inter_ch, 1, 0),
            Conv3x3(inter_ch, growth_rate//2, 1, 1),
            Conv3x3(growth_rate//2, growth_rate//2, 1, 1)
        )

    def forward(self, x):
        feat_l = self.left(x)
        feat_r = self.right(x)
        feat = torch.cat([x, feat_l, feat_r], dim=1)
        return feat


class TransitionBlock(nn.Sequential):
    def __init__(self, inp, pool=True):
        if pool:
            pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            pool = nn.Identity()
        super(TransitionBlock, self).__init__(
            Conv1x1(inp, inp, 1, 0),
            pool
        )


class DenseStage(nn.Module):
    def __init__(self, inp, nblock, bwidth, growth_rate, pool):
        super(DenseStage, self).__init__()
        current_ch = inp
        inter_ch = int(growth_rate // 2 * bwidth / 4) * 4
        stage = nn.Sequential()
        for i in range(nblock):
            stage.add_module("dense{}".format(
                i+1), TwoWayDenseBlock(current_ch, growth_rate, inter_ch))
            current_ch += growth_rate
        stage.add_module("transition", TransitionBlock(current_ch, pool=pool))
        self.stage = stage

    def forward(self, x):
        return self.stage(x)


class CSPDenseStage(nn.Module):
    def __init__(self, inp, nblock, bwidth, growth_rate, pool, partial_ratio):
        super(CSPDenseStage, self).__init__()

        split_ch = int(inp * partial_ratio)
        inter_ch = int(growth_rate // 2 * bwidth / 4) * 4
        self.split_ch = split_ch
        dense_branch = nn.Sequential()
        current_ch = split_ch
        for i in range(nblock):
            dense_branch.add_module("dense{}".format(
                i+1), TwoWayDenseBlock(current_ch, growth_rate, inter_ch))
            current_ch += growth_rate
        dense_branch.add_module(
            "transition1", TransitionBlock(current_ch, pool=False))
        self.dense_branch = dense_branch
        self.transition2 = TransitionBlock(
            current_ch + inp - split_ch, pool=pool)

    def forward(self, x):
        x1 = x[:, :self.split_ch, ...]
        x2 = x[:, self.split_ch:, ...]

        feat1 = self.dense_branch(x1)
        feat = torch.cat([x2, feat1], dim=1)
        feat = self.transition2(feat)
        return feat


class PeleeNet(nn.Module):
    def __init__(self, inp=3, nclass=1000, growth_rate=32, nblocks=[3, 4, 8, 6],
                 bottleneck_widths=[1/2, 1, 2, 4], partial_ratio=1.0):
        super(PeleeNet, self).__init__()

        self.stem = StemBlock(inp)
        current_ch = 32
        stages = nn.Sequential()
        pool = True
        assert len(nblocks) == len(bottleneck_widths)
        for i, (nblock, bwidth) in enumerate(zip(nblocks, bottleneck_widths)):
            if (i+1) == len(nblocks):
                pool = False
            if partial_ratio < 1.0:
                stage = CSPDenseStage(
                    current_ch, nblock, bwidth, growth_rate, pool, partial_ratio)
            else:
                stage = DenseStage(current_ch, nblock,
                                   bwidth, growth_rate, pool)
            stages.add_module("stage{}".format(i+1), stage)
            current_ch += growth_rate * nblock
        self.stages = stages
        self.classifier = nn.Linear(current_ch, nclass)

    def forward(self, x):
        feat = self.stem(x)
        feat = self.stages(feat)
        feat = torch.mean(feat, dim=[2, 3])  # GAP
        pred = self.classifier(feat)
        return pred


if __name__ == '__main__':
    import torch.onnx
    net = PeleeNet(partial_ratio=0.5)
    input = torch.randn(1, 3, 224, 224)
    res = net(input)

    torch.onnx.export(net, input, "csppeleenet50.onnx", verbose=False)
    from thop import profile, clever_format
    macs, params = profile(net, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)
