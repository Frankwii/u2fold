import torch
from torch import Tensor

def color_minimizable(input: Tensor, ground_truth: Tensor) -> Tensor:
    """Color dissimilarity between input and ground truth.

    Minimizing this metric should be intuitively equivalent to the images being
    channel-wise similar in a cosine similarity sense.

    Since the input and ground truth are both images, they have nonnegative entries 
    and thus lie in the fisrt 2^n-adrant of R^n. Therefore the angle between them must be 
    between 0 and pi/2. Hence, the cosine between their angles is always nonnegative and
    this function will always return values in the [0, 1] interval, reaching 0 only for
    identical images.

    Algebraically, this can be seen via the following equality:
    
    \\[
        \\cos(\\hat{uv}) = \\frac{<u, v>}{\\|u\\|\\|v\\|},
    \\]
    which is trivially nonnegative whenever \\(u, v \\geq 0\\).
    """
    return 1 - torch.cosine_similarity(input, ground_truth, dim=1).mean()

def color_minimizable_calibrated(input: Tensor, ground_truth: Tensor) -> Tensor:

    uieb_average = 0.035978469997644424

    return color_minimizable(input, ground_truth) / uieb_average
