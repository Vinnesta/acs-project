import numpy as np
from PIL import Image, ImageDraw
import re
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator

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
      
  def yield_tokens(data_iter):
    for text in data_iter:
        yield Tokeniser.heuristic_ending_tokenizer(text)
        
  def create_vocab(df, min_freq, start_token, end_token):
    pad_token = "<pad>"
    unk_token = "<unk>"
    vocab = build_vocab_from_iterator(Tokeniser.yield_tokens(df['allSpeakerContents']), specials=[pad_token, unk_token, start_token, end_token], min_freq=min_freq)
    vocab.set_default_index(vocab[unk_token])
    return vocab


class ColourDataset(Dataset):
  def __init__(self, x, c, y, colour_vector_dim):
    self.x = x
    self.c = c
    self.y = y
    self.colour_vector_dim = colour_vector_dim
    self.c_vec = self.vectorise_colours()

  def vectorise_colours(self) -> torch.Tensor:
    resolution = [3]
    c_np = self.c.cpu().detach().numpy()
    n = c_np.shape[0]
    assert(c_np.shape[1] == 3)
    assert(c_np.shape[2] == 3)
    return torch.Tensor(np.reshape(ColoursUtil.vectorize_all(np.reshape(c_np, (n*3, 3)), resolution), (n, 3, self.colour_vector_dim)))
  
  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return (self.x[idx], self.c_vec[idx], self.c[idx]), self.y[idx]

class DatasetFactory():
  def __init__(self, max_seq_len, colour_vector_dim, start_token, end_token):
    self.max_seq_len = max_seq_len
    self.colour_vector_dim = colour_vector_dim
    self.start_token = start_token
    self.end_token = end_token

  def process_listener_data(self, df, vocab):
    # Convert speaker utterances into sequences of one-hot encoded embeddings
    # But rather than actual one-hot vectors, just use the vocabulary index for each token
    # Dim: Number of samples * MAX_SEQ_LEN
    X = []
    for text in df['allSpeakerContents']:
      text_tensor = torch.tensor(vocab(Tokeniser.heuristic_ending_tokenizer(text)))
      text_tensor = text_tensor[:self.max_seq_len]
      
      # Pad the text tensor to MAX_SEQ_LEN so that every sample has the same sequence length
      pad_length = self.max_seq_len - len(text_tensor)
      text_tensor = F.pad(text_tensor, (0, pad_length))
      X.append(text_tensor)

    X = torch.stack(X)

    # Get HSL colour context of each sample
    # Dim: Number of samples * number of colours in context (3) * HSL values (3)
    targetH = df['targetH'].tolist()
    targetS = df['targetS'].tolist()
    targetL = df['targetL'].tolist()
    distr1H = df['distr1H'].tolist()
    distr1S = df['distr1S'].tolist()
    distr1L = df['distr1L'].tolist()
    distr2H = df['distr2H'].tolist()
    distr2S = df['distr2S'].tolist()
    distr2L = df['distr2L'].tolist()

    targets = torch.transpose(torch.tensor([targetH, targetS, targetL]), dim0=0, dim1=1)
    distractors1 = torch.transpose(torch.tensor([distr1H, distr1S, distr1L]), dim0=0, dim1=1)
    distractors2 = torch.transpose(torch.tensor([distr2H, distr2S, distr2L]), dim0=0, dim1=1)
    C = torch.stack([targets, distractors1, distractors2], dim=1)

    # Set the target labels as zero for all samples, since the first colour in context is the target
    # Dim: Number of samples
    Y = torch.zeros(X.shape[0], dtype=torch.long)

    dataset = ColourDataset(x=X, c=C, y=Y, colour_vector_dim=self.colour_vector_dim)
    return dataset
    
  def process_speaker_data(self, df, vocab, orig_colour_order=False):
    # Convert speaker utterances into sequences of one-hot encoded embeddings
    # But rather than actual one-hot vectors, just use the vocabulary index for each token
    # Dim: Number of samples * MAX_SEQ_LEN
    X = []
    Y = []
    for text in df['allSpeakerContents']:
      text_tokens = Tokeniser.heuristic_ending_tokenizer(text)
      text_tokens.insert(0, self.start_token)
      text_tensor = torch.tensor(vocab(text_tokens))
      text_tensor = text_tensor[:self.max_seq_len+1] # Add 1 here to account for the START_TOKEN

      # Pad the text tensor to MAX_SEQ_LEN so that every sample has the same sequence length
      pad_length = self.max_seq_len + 1 - len(text_tensor)
      x_tensor = F.pad(text_tensor, (0, pad_length))
      X.append(x_tensor)

      # To get target label y, remove START_TOKEN and add END_TOKEN
      text_tensor = torch.cat((text_tensor[1:], torch.tensor(vocab([self.end_token]))))
      y_tensor = F.pad(text_tensor, (0, pad_length))
      Y.append(y_tensor)

    X = torch.stack(X)
    Y = torch.stack(Y)

    # Get HSL colour context of each sample
    # Dim: Number of samples * number of colours in context (3) * HSL values (3)
    targetH = df['targetH'].tolist()
    targetS = df['targetS'].tolist()
    targetL = df['targetL'].tolist()
    distr1H = df['distr1H'].tolist()
    distr1S = df['distr1S'].tolist()
    distr1L = df['distr1L'].tolist()
    distr2H = df['distr2H'].tolist()
    distr2S = df['distr2S'].tolist()
    distr2L = df['distr2L'].tolist()

    targets = torch.transpose(torch.tensor([targetH, targetS, targetL]), dim0=0, dim1=1)
    distractors1 = torch.transpose(torch.tensor([distr1H, distr1S, distr1L]), dim0=0, dim1=1)
    distractors2 = torch.transpose(torch.tensor([distr2H, distr2S, distr2L]), dim0=0, dim1=1)
    
    if orig_colour_order:
      C = torch.stack([distractors1, distractors2, targets], dim=1) # Place `targets` at the end as per 'Colors in Context'
    else:
      C = torch.stack([targets, distractors1, distractors2], dim=1)

    dataset = ColourDataset(x=X, c=C, y=Y, colour_vector_dim=self.colour_vector_dim)
    return dataset