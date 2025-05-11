from torch import nn

from u2fold.utils.track import track


@track(tag="model/mock")
class MockModel(nn.Module):
    ...
