# Black and white image denoising with Gibbs sampler

## Grid graph

Each pixel is a graph node.
It a node is not on the image edge, it has 4 neighbors:
- left: `0`
- top: `1`
- right: `2`
- bottom: `3`

Each node has two labels: `0` (black color) and `1` (white color).
There are edges between all neighbor nodes connecting all labels.
Edge weight is `0` if it connects the same labels and
`beta` if the labels are different.

## Image generation

To generate an image run
```bash
python image_generation image_height image_width edge_weight epsilon
```
from `src/` directory.

Firstly, a random image is generated with discrete uniform distribution.
This image contains `0` and `1` in each cell with probability `0.5`.

Then the image is sampling.
One iteration of sampling is the next.
We fix an image and go through each pixel.

![Alt text](images/image_sampling.png)

For a pixel we calculate `a = zero_weight` (sum of yellow edges)
and `b = unit_weight` (sum of blue edges).
Then `t = exp(-a) / (exp(-a) + exp(-b))`.
We generate a random number `x` from uniform distribution `U([0, 1])`.
If `x < t`, then `0` label is fixed in the pixel.
If `x >= t`, then `1` label is fixed in the pixel.

After proceeding this for all pixels many times image obtains noise:
each pixel changes its color with probability `epsilon.`


## Testing

To test functions run
```bash
pytest
```
from `test/` directory.
