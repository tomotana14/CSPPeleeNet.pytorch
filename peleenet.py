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
    def __init__(self, inp, growth_rate):
        super(TwoWayDenseBlock, self).__init__()
        self.left = nn.Sequential(
            Conv1x1(inp, 2*growth_rate, 1, 0),
            Conv3x3(2*growth_rate, growth_rate//2, 1, 1)
        )
        self.right = nn.Sequential(
            Conv1x1(inp, 2*growth_rate, 1, 0),
            Conv3x3(2*growth_rate, growth_rate//2, 1, 1),
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
    def __init__(self, inp, nblocks, growth_rate, pool):
        super(DenseStage, self).__init__()
        current_ch = inp

        stage = nn.Sequential()
        for i in range(nblocks):
            stage.add_module("dense{}".format(
                i+1), TwoWayDenseBlock(current_ch, growth_rate))
            current_ch += growth_rate
        stage.add_module("transition", TransitionBlock(current_ch, pool=pool))
        self.stage = stage

    def forward(self, x):
        return self.stage(x)


class CSPDenseStage(nn.Module):
    def __init__(self, inp, nblocks, growth_rate, pool, partial_ratio):
        super(CSPDenseStage, self).__init__()

        split_ch = int(inp * partial_ratio)
        self.split_ch = split_ch
        dense_branch = nn.Sequential()
        current_ch = split_ch
        for i in range(nblocks):
            dense_branch.add_module("dense{}".format(
                i+1), TwoWayDenseBlock(current_ch, growth_rate))
            current_ch += growth_rate
        dense_branch.add_module(
            "transition1", TransitionBlock(current_ch, pool=False))
        self.dense_branch = dense_branch
        self.transition2 = TransitionBlock(
            current_ch + inp - split_ch, pool=True)

    def forward(self, x):
        x1 = x[:, :self.split_ch, ...]
        x2 = x[:, self.split_ch:, ...]

        feat1 = self.dense_branch(x1)
        feat = torch.cat([x2, feat1], dim=1)
        feat = self.transition2(feat)
        return feat


class PeleeNet(nn.Module):
    def __init__(self, inp=3, nclass=1000, growth_rate=32, nblocks=[3, 4, 8, 6], partial_ratio=1.0):
        super(PeleeNet, self).__init__()

        self.stem = StemBlock(inp)
        current_ch = 32
        stages = nn.Sequential()
        pool = True
        for i, n in enumerate(nblocks):
            if (i+1) == len(nblocks):
                pool = False
            if partial_ratio < 1.0:
                stage = CSPDenseStage(
                    current_ch, n, growth_rate, pool, partial_ratio)
            else:
                stage = DenseStage(current_ch, n, growth_rate, pool)
            stages.add_module("stage{}".format(i+1), stage)
            current_ch += growth_rate * n
        self.stages = stages
        self.classifier = nn.Linear(current_ch, nclass)

    def forward(self, x):
        feat = self.stem(x)
        feat = self.stages(feat)
        feat = torch.mean(feat, dim=[2, 3])  # GAP
        pred = self.classifier(feat)
        return pred


if __name__ == '__main__':
    net = PeleeNet(partial_ratio=0.5)
    from ptflops import get_model_complexity_info
    from thop import profile

    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(net, inputs=(input, ))
    from thop import clever_format
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)
    # macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
    #                                         print_per_layer_stat=True, verbose=False)
    #print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #x = torch.randn(1, 3, 224, 224)
    #y = net(x)
    # print(y.shape)
