# def __estimate_background_light_switching_dimensions(
#     image: Tensor, split_horizontal: bool
# ) -> tuple[float, float, float]:
#     if image.numel() <= 48:  # 48=3*4*4; 4x4 patch is the largest allowed
#         background_light = __linear_search_background_light(image)
#
#         return cast(
#             tuple[float, float, float], tuple(background_light.flatten())
#         )
#
#     split_dimension = -(1 + split_horizontal)
#     dimension_size = image.size(split_dimension)
#     half_size = dimension_size // 2
#     split_sizes = [half_size, dimension_size - half_size]
#
#     half1, half2 = torch.split(image, split_sizes, dim=split_dimension)
#
#     if __assign_scores(half1) < __assign_scores(half2):
#         return __estimate_background_light_switching_dimensions(
#             half2, not split_horizontal
#         )
#
#     return __estimate_background_light_switching_dimensions(
#         half1, not split_horizontal
#     )
#
#
# def estimate_background_light_switching_dimensions(
#     image: Tensor,
# ) -> tuple[float, float, float]:
#     return __estimate_background_light_switching_dimensions(image, False)
#

### padding = (patch_radius,) * 4
### padded_data = torch.nn.functional.pad(
###     all_data, pad=padding, mode="constant", value=1.0
### )  # (B, 4, H + 2 * patch_radius, W + 2 * patch_radius)

### K: Kernel size (= 2 * patch_radius + 1)
### P := K ** 2
### L: Number of patches = H*W


### patches = torch.nn.functional.unfold(
###     padded_data,
###     kernel_size=patch_radius * 2 + 1,
###     padding=0,
### ).reshape(batch_size, 4, -1, height * width)  # (B, 4, P, H*W)
###
### patch_minima = patches.min(dim=2).values  # (B, 4, H*W)

