from PIL import Image, ImageDraw
import re

class ColoursUtil():
  def hsl_to_hsv(hsl):
    '''
    Args:
      `hsl`: HSL colours where h in [0, 360], s in [0, 100], and l in [0, 100]

    Returns:
      tuple of hsv where each value is 8-bit, i.e. in the range [0, 255]
    '''
    h, s, l = hsl
    # Convert S and L values into the range [0, 1]
    s = s/100
    l = l/100

    # Convert to HSV, where h in [0, 360], s in [0, 1], and v in [0, 1]
    v = l + s * min(l, 1-l)
    s = 0 if v == 0 else 2*(1 - l/v)
    
    # Convert HSV values into the range expected by `Image.new()`, i.e. tuple of 8-bit values
    h = int(h/360*255)
    s = int(s*255)
    v = int(v*255)
    return h, s, v

  def show_colour(hsl):
    '''
    Args:
      `hsl`: HSL colours where h in [0, 360], s in [0, 100], and l in [0, 100]
    '''
    im = Image.new('HSV', (50, 50), color=ColoursUtil.hsl_to_hsv(hsl))
    im.show()
    print(f"{hsl} \n")

  def show_colours(colours):
    '''
    Args:
      `colours`: Tensor of HSL colours where h in [0, 360], s in [0, 100], and l in [0, 100]
    '''
    num_colours = colours.shape[0]
    height = int(50/num_colours)
    for i in range(num_colours):
      im = Image.new('HSV', (50, height), color=ColoursUtil.hsl_to_hsv(colours[i].tolist()))
      im.show()
  
  # https://github.com/futurulus/coop-nets
  RANGES_HSV = (361.0, 101.0, 101.0)
  def vectorize_all(colors, resolution):
    '''
    >>> normalize = lambda v: np.where(v.round(2) == 0.0, 0.0, v.round(2))
    >>> normalize(FourierVectorizer([2]).vectorize_all([(255, 0, 0), (0, 255, 255)]))
    array([[ 1.,  1.,  1.,  1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.],
            [ 1., -1., -1.,  1.,  1., -1., -1.,  1.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.]], dtype=float32)
    '''
    if len(resolution) == 1:
      resolution = resolution * 3

    colors = np.array([colors])
    assert len(colors.shape) == 3, colors.shape
    assert colors.shape[2] == 3, colors.shape

    ranges = np.array(ColoursUtil.RANGES_HSV)
    color_0_1 = colors / (ranges - 1.0)

    # Using a Fourier representation causes colors at the boundary of the
    # space to behave as if the space is toroidal: red = 255 would be
    # about the same as red = 0. We don't want this...
    xyz = color_0_1[0] / 2.0
    xyz[:, 0] *= 2.0

    #ax, ay, az = [np.hstack([np.arange(0, g / 2), np.arange(r - g / 2, r)])
    #               for g, r in zip(resolution, ranges)]
    ax, ay, az = [np.arange(0, g) for g, r in zip(resolution, ranges)]
    gx, gy, gz = np.meshgrid(ax, ay, az)
    arg = (np.multiply.outer(xyz[:, 0], gx) +
            np.multiply.outer(xyz[:, 1], gy) +
            np.multiply.outer(xyz[:, 2], gz))
    assert arg.shape == (xyz.shape[0],) + tuple(resolution), arg.shape
    repr_complex = np.exp(-2j * np.pi * (arg % 1.0)).swapaxes(1, 2).reshape((xyz.shape[0], -1))
    result = np.hstack([repr_complex.real, repr_complex.imag]).astype(np.float32)
    
    normalize = lambda v: np.where(v.round(2) == 0.0, 0.0, v.round(2))
    return normalize(result)
    
    
class Tokeniser():
  # https://github.com/futurulus/coop-nets

  WORD_RE_STR = r"""
  (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
  |
  (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
  |
  (?:[\w_]+)                     # Words without apostrophes or dashes.
  |
  (?:\.(?:\s*\.){1,})            # Ellipsis dots.
  |
  (?:\*{1,})                     # Asterisk runs.
  |
  (?:\S)                         # Everything else that isn't whitespace.
  """

  WORD_RE = re.compile(r"(%s)" % WORD_RE_STR, re.VERBOSE | re.I | re.UNICODE)

  def basic_unigram_tokenizer(s, lower=True):
      words = Tokeniser.WORD_RE.findall(s)
      if lower:
          words = [w.lower() for w in words]
      return words

  def heuristic_ending_tokenizer(s, lower=True):
      words = Tokeniser.basic_unigram_tokenizer(s, lower=lower)
      return [seg for w in words for seg in Tokeniser.heuristic_segmenter(w)]

  ENDINGS = ['er', 'est', 'ish']

  def heuristic_segmenter(word):
      for ending in Tokeniser.ENDINGS:
          if word.endswith(ending):
              return [word[:-len(ending)], '+' + ending]
      return [word]