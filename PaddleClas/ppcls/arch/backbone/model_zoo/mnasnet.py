import warnings
import paddle
import paddle.nn as nn
from typing import Any
from typing import Dict
from typing import List
import numpy as np

__all__ = ['MnasNet_A1']
_BN_MOMENTUM = 1 - 0.9997


class _InvertedResidual(nn.Layer):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride:
        int, expansion_factor: int, bn_momentum: float=0.1) ->None:
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = in_ch == out_ch and stride == 1
        self.layers = nn.Sequential(
            nn.Conv2D(in_ch, mid_ch, 1, bias_attr=False,
                weight_attr=nn.initializer.KaimingNormal()),
            nn.BatchNorm2D(mid_ch, momentum=1 - bn_momentum),
            nn.ReLU(),
            nn.Conv2D(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2,
                stride=stride, groups=mid_ch, bias_attr=False,
                weight_attr=nn.initializer.KaimingNormal()),
            nn.BatchNorm2D(mid_ch, momentum=1 - bn_momentum), nn.ReLU(),
            nn.Conv2D(mid_ch, out_ch, 1, bias_attr=False,
                weight_attr=nn.initializer.KaimingNormal()),
            nn.BatchNorm2D(out_ch, momentum=1 - bn_momentum))

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack(in_ch: int, out_ch: int, kernel_size: int, stride: int,
    exp_factor: int, repeats: int, bn_momentum: float) ->paddle.nn.Sequential:
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride,
        exp_factor, bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(_InvertedResidual(out_ch, out_ch, kernel_size, 1,
            exp_factor, bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val: float, divisor: int, round_up_bias: float=0.9
    ) ->int:
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha: float) ->List[int]:
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MNASNet(paddle.nn.Layer):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1.0, num_classes=1000)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    2
    >>> y.nelement()
    1000
    """
    _version = 2

    def __init__(self, alpha: float, num_classes: int=1000, dropout: float=0.2
        ) ->None:
        super(MNASNet, self).__init__()
        assert alpha > 0.0
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths(alpha)
        layers = [
            nn.Conv2D(3, depths[0], 3, padding=1, stride=2, bias_attr=False,
                weight_attr=nn.initializer.KaimingNormal()),
            nn.BatchNorm2D(depths[0], momentum=1 - _BN_MOMENTUM),
            nn.ReLU(),

            nn.Conv2D(depths[0], depths[0], 3, padding=1, stride=1,
                groups=depths[0], bias_attr=False,
                weight_attr=nn.initializer.KaimingNormal()),
            nn.BatchNorm2D(depths[0], momentum=1 - _BN_MOMENTUM),
            nn.ReLU(), 
            nn.Conv2D(depths[0], depths[1], 1, padding=0, stride=1,
                bias_attr=False, weight_attr=nn.initializer.KaimingNormal()), 
            nn.BatchNorm2D(depths[1], momentum=1 - _BN_MOMENTUM),

            _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM),
            _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM),
            _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM),
            _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM), 

            nn.Conv2D(depths[7], 1280, 1, padding=0, stride=1, bias_attr=False,
                weight_attr=nn.initializer.KaimingNormal()), 
            nn.BatchNorm2D(1280, momentum=1 - _BN_MOMENTUM), 
            nn.ReLU()]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(
            1280, num_classes, weight_attr=nn.initializer.KaimingUniform()))
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = x.mean([2, 3])
        return self.classifier(x)

    def _initialize_weights(self) ->None:
        for layer in self.sublayers():
            if isinstance(layer, paddle.nn.BatchNorm2D):
                layer.weight.set_value(np.ones(layer.weight.shape).astype("float32"))
                layer.bias.set_value(np.zeros(layer.bias.shape).astype("float32"))


def _load_pretrained(model_name: str, model: nn.Layer, progress: bool) ->None:
    if model_name not in _MODEL_URLS or _MODEL_URLS[model_name] is None:
        raise ValueError('No checkpoint is available for model type {}'.
            format(model_name))
    checkpoint_url = _MODEL_URLS[model_name]


def mnasnet0_5(pretrained: bool=False, progress: bool=True, **kwargs: Any
    ) ->MNASNet:
    """MNASNet with depth multiplier of 0.5 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.5, **kwargs)
    if pretrained:
        _load_pretrained('mnasnet0_5', model, progress)
    return model


def mnasnet0_75(pretrained: bool=False, progress: bool=True, **kwargs: Any
    ) ->MNASNet:
    """MNASNet with depth multiplier of 0.75 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.75, **kwargs)
    if pretrained:
        _load_pretrained('mnasnet0_75', model, progress)
    return model


def mnasnet1_0(pretrained: bool=False, progress: bool=True, **kwargs: Any
    ) ->MNASNet:
    """MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.0, **kwargs)
    if pretrained:
        _load_pretrained('mnasnet1_0', model, progress)
    return model


def mnasnet1_3(pretrained: bool=False, progress: bool=True, **kwargs: Any
    ) ->MNASNet:
    """MNASNet with depth multiplier of 1.3 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.3, **kwargs)
    if pretrained:
        _load_pretrained('mnasnet1_3', model, progress)
    return model

def MnasNet_A1(**kwargs):
    model = MNASNet(1.0, **kwargs)
    return model
