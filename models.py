from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Linear, ReLU, Module, Sequential
from torch.nn.utils.rnn import pack_padded_sequence

class LiteralListener(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, colour_vector_dim, vocab_size, dropout=0.0):
    super(LiteralListener, self).__init__()
    self.colour_vector_dim = colour_vector_dim

    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.embedding_dropout = nn.Dropout(p=dropout)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    self.lstm_dropout = nn.Dropout(p=dropout)
    self.description_representation = nn.Linear(hidden_dim, self.colour_vector_dim)
    self.rep_dropout = nn.Dropout(p=dropout)
    self.covariance = nn.Linear(hidden_dim, self.colour_vector_dim*self.colour_vector_dim)

  def forward(self, x, c_vec):
    embeds = self.word_embeddings(x)
    embeds = self.embedding_dropout(embeds)
    seq_lengths = torch.count_nonzero(x, dim=1)
    packed_embeds = pack_padded_sequence(embeds, seq_lengths, batch_first=True, enforce_sorted=False)
    lstm_out, (h_n, c_n) = self.lstm(packed_embeds)
    final_out = self.lstm_dropout(h_n[-1])
    rep = self.description_representation(final_out)
    rep = self.rep_dropout(rep)
    cov_vector = self.covariance(final_out)
    cov_matrix = torch.reshape(cov_vector, (cov_vector.shape[0], self.colour_vector_dim, self.colour_vector_dim))
    
    scores = []
    for i in range(3):
      cv = c_vec[:, i, :]
      # N.B. First dimension (dim=0) is the batch size, so operations on the individual matrices start at index 1
      delta = torch.unsqueeze(cv - rep, dim=2)
      delta_t = torch.transpose(delta, dim0=2, dim1=1)
      score = -torch.matmul(torch.matmul(delta_t, cov_matrix), delta)
      score = score.squeeze(dim=-1)
      scores.append(score)
    return torch.cat(scores, dim=-1)

  def predict(self, x, c_vec):
    self.eval()
    with torch.no_grad():
      forward = self(x, c_vec)
    return F.softmax(forward, dim=1)


class LiteralSpeaker(nn.Module):
  def __init__(self, embedding_dim, colour_dim, hidden_dim, colour_vector_dim, vocab_size, dropout=0.0):
    super(LiteralSpeaker, self).__init__()
    
    self.colour_mlp = Sequential(
      Linear(colour_vector_dim, colour_dim), ReLU(),
      Linear(colour_dim, colour_dim), ReLU()
    )
    self.colour_dropout = nn.Dropout(p=dropout)
    
    # Layers for generating utterances
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.embedding_dropout = nn.Dropout(p=dropout)

    lstm_input_size = embedding_dim + (colour_dim*2)
    self.utterance_lstm = nn.LSTM(lstm_input_size, hidden_dim, batch_first=True)
    self.lstm_dropout = nn.Dropout(p=dropout)
    self.linear = nn.Linear(hidden_dim, vocab_size)
    
  def forward(self, x, c_vec, h_0=None, c_0=None):
    # c_vec of shape (num_samples, num_colours, COLOUR_VECTOR_DIM), where first colour is target colour
    num_colours = c_vec.shape[1] 
    colour_encodings = []
    for i in range(num_colours):
      colour_encodings.append(self.colour_mlp(c_vec[:, i]))

    # Aggregate encodings of distractor colours
    distractor_colours_rep = colour_encodings[1] + colour_encodings[2]
    colours_rep = torch.cat((colour_encodings[0], distractor_colours_rep), dim=-1)
    colours_rep = self.colour_dropout(colours_rep)

    embeds = self.word_embeddings(x)
    embeds = self.embedding_dropout(embeds)
    total_length = embeds.shape[1]

    # Concatenate colour representation to the embeddings of each token
    utterance_input = torch.cat((colours_rep.unsqueeze(1).expand(-1, total_length, -1), embeds), dim=-1)

    seq_lengths = torch.count_nonzero(x, dim=1)
    packed_utterance = pack_padded_sequence(utterance_input, seq_lengths, batch_first=True, enforce_sorted=False)
    if h_0 is None or c_0 is None:
      lstm_out, (h_n, c_n) = self.utterance_lstm(packed_utterance)
    else:
      lstm_out, (h_n, c_n) = self.utterance_lstm(packed_utterance, (h_0, c_0))
    
    lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=total_length)
    lstm_out = self.lstm_dropout(lstm_out)
    y_hat = self.linear(lstm_out)
    return y_hat, (h_n, c_n)

  def train_model(self, inputs, y, optimiser):
    x = inputs[0]
    c_vec = inputs[1]

    self.train()
    optimiser.zero_grad()
    y_hat, _ = self(x, c_vec)
    vocab_size = y_hat.shape[-1]
    y_flatten = y.view(-1)
    y_hat_flatten = y_hat.view(-1, vocab_size)
    
    nonzeros = y_flatten.nonzero(as_tuple=True)[0]
    loss = F.cross_entropy(y_hat_flatten[nonzeros], y_flatten[nonzeros])
    loss.backward()
    total_loss = loss.data
    optimiser.step()
    return total_loss


class OriginalLiteralSpeaker(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, colour_vector_dim, vocab_size):
    super(OriginalLiteralSpeaker, self).__init__()
    
    # LSTM for representations of colours in context
    self.colours_lstm = nn.LSTM(colour_vector_dim, hidden_dim, batch_first=True)
    
    # Layers for generating utterances
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    lstm_input_size = embedding_dim + hidden_dim
    self.utterance_lstm = nn.LSTM(lstm_input_size, hidden_dim, batch_first=True)
    self.linear = nn.Linear(hidden_dim, vocab_size)
    
  def forward(self, x, c_vec, h_0=None, c_0=None):
    _, (_, colours_c_n) = self.colours_lstm(c_vec)
    colours_rep = colours_c_n[-1]

    embeds = self.word_embeddings(x)
    total_length = embeds.shape[1]

    # Concatenate colour representation to the embeddings of each token
    utterance_input = torch.cat((colours_rep.unsqueeze(1).expand(-1, total_length, -1), embeds), dim=-1)

    seq_lengths = torch.count_nonzero(x, dim=1)
    packed_utterance = pack_padded_sequence(utterance_input, seq_lengths, batch_first=True, enforce_sorted=False)
    if h_0 is None or c_0 is None:
      lstm_out, (h_n, c_n) = self.utterance_lstm(packed_utterance)
    else:
      lstm_out, (h_n, c_n) = self.utterance_lstm(packed_utterance, (h_0, c_0))
    
    lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=total_length)
    y_hat = self.linear(lstm_out)
    return y_hat, (h_n, c_n)

  def train_model(self, inputs, y, optimiser):
    x = inputs[0]
    c_vec = inputs[1]

    self.train()
    optimiser.zero_grad()
    y_hat, _ = self(x, c_vec)
    vocab_size = y_hat.shape[-1]
    y_flatten = y.view(-1)
    y_hat_flatten = y_hat.view(-1, vocab_size)
    
    nonzeros = y_flatten.nonzero(as_tuple=True)[0]
    loss = F.cross_entropy(y_hat_flatten[nonzeros], y_flatten[nonzeros])
    loss.backward()
    total_loss = loss.data
    optimiser.step()
    return total_loss
    

class Correlation(Enum):
  INDEPENDENT=1
  MAXIMUM=2
  INDEPENDENT_MOD=3
  MAXIMUM_MOD=4


class ProbLiteralListener(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, colour_vector_dim, vocab_size, correlation, dropout=0.0):
    super(ProbLiteralListener, self).__init__()
    self.colour_vector_dim = colour_vector_dim

    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.embedding_dropout = nn.Dropout(p=dropout)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    self.lstm_dropout = nn.Dropout(p=dropout)
    self.description_representation = nn.Linear(hidden_dim, self.colour_vector_dim)
    self.rep_dropout = nn.Dropout(p=dropout)
    self.covariance = nn.Linear(hidden_dim, self.colour_vector_dim*self.colour_vector_dim)
    self.correlation = correlation
  
  def joint_choice(self, marginals):
    p1 = marginals[0]
    p2 = marginals[1]
    p3 = marginals[2]

    # Triplet of [t,f] representing the bool values of categories 1, 2, 3
    # e.g. `ttf` := 1 is true, 2 is true, 3 is false
    if self.correlation == Correlation.INDEPENDENT:
      ttt = p1*p2*p3
      ttf = p1*p2*(1-p3)
      tft = p1*(1-p2)*p3
      tff = p1*(1-p2)*(1-p3)
      ftt = (1-p1)*p2*p3
      ftf = (1-p1)*p2*(1-p3)
      fft = (1-p1)*(1-p2)*p3
      fff = (1-p1)*(1-p2)*(1-p3)
    elif self.correlation == Correlation.MAXIMUM:
      ttt = torch.min(torch.cat((p1, p2, p3), dim=1), dim=1, keepdim=True)[0]
      ttf = torch.clamp(torch.min(torch.cat((p1-p3, p2-p3), dim=1), dim=1, keepdim=True)[0], min=0)
      tft = torch.clamp(torch.min(torch.cat((p1-p2, p3-p2), dim=1), dim=1, keepdim=True)[0], min=0)
      tff = torch.clamp(p1 - torch.max(torch.cat((p2, p3), dim=1), dim=1, keepdim=True)[0], min=0)
      ftt = torch.clamp(torch.min(torch.cat((p2-p1, p3-p1), dim=1), dim=1, keepdim=True)[0], min=0)
      ftf = torch.clamp(p2 - torch.max(torch.cat((p1, p3), dim=1), dim=1, keepdim=True)[0], min=0)
      fft = torch.clamp(p3 - torch.max(torch.cat((p1, p2), dim=1), dim=1, keepdim=True)[0], min=0)
      fff = 1 - torch.max(torch.cat((p1, p2, p3), dim=1), dim=1, keepdim=True)[0]
    elif self.correlation == Correlation.INDEPENDENT_MOD:
      ttt = p1*p2*p3
      ttf = p1*p2*(1-p3)
      tft = p1*(1-p2)*p3
      tff = p1*(1-p2)*(1-p3)
      ftt = (1-p1)*p2*p3
      ftf = (1-p1)*p2*(1-p3)
      fft = (1-p1)*(1-p2)*p3
      fff = 0
      
      # Normalise probability masses because `fff` is excluded
      prob_mass = ttt + ttf + tft + tff + ftt + ftf + fft
      ttt = ttt / prob_mass
      ttf = ttf / prob_mass
      tft = tft / prob_mass
      tff = tff / prob_mass
      ftt = ftt / prob_mass
      ftf = ftf / prob_mass
      fft = fft / prob_mass
    elif self.correlation == Correlation.MAXIMUM_MOD:
      ttt = torch.min(torch.cat((p1, p2, p3), dim=1), dim=1, keepdim=True)[0]
      ttf = torch.clamp(torch.min(torch.cat((p1-p3, p2-p3), dim=1), dim=1, keepdim=True)[0], min=0)
      tft = torch.clamp(torch.min(torch.cat((p1-p2, p3-p2), dim=1), dim=1, keepdim=True)[0], min=0)
      tff = torch.clamp(p1 - torch.max(torch.cat((p2, p3), dim=1), dim=1, keepdim=True)[0], min=0)
      ftt = torch.clamp(torch.min(torch.cat((p2-p1, p3-p1), dim=1), dim=1, keepdim=True)[0], min=0)
      ftf = torch.clamp(p2 - torch.max(torch.cat((p1, p3), dim=1), dim=1, keepdim=True)[0], min=0)
      fft = torch.clamp(p3 - torch.max(torch.cat((p1, p2), dim=1), dim=1, keepdim=True)[0], min=0)
      fff = 0
      
      # Normalise probability masses because `fff` is excluded
      prob_mass = ttt + ttf + tft + tff + ftt + ftf + fft
      ttt = ttt / prob_mass
      ttf = ttf / prob_mass
      tft = tft / prob_mass
      tff = tff / prob_mass
      ftt = ftt / prob_mass
      ftf = ftf / prob_mass
      fft = fft / prob_mass
    else:
      raise NotImplementedError
    
    # `c_xyz` := the probability of choosing each category given the joint outcome xyz
    c_ttt = torch.tensor([1/3, 1/3, 1/3])
    c_ttf = torch.tensor([1/2, 1/2, 0])
    c_tft = torch.tensor([1/2, 0, 1/2])
    c_tff = torch.tensor([1, 0, 0])
    c_ftt = torch.tensor([0, 1/2, 1/2])
    c_ftf = torch.tensor([0, 1, 0])
    c_fft = torch.tensor([0, 0, 1])
    c_fff = torch.tensor([1/3, 1/3, 1/3])
    choice_prob = ttt*c_ttt + ttf*c_ttf + tft*c_tft + tff*c_tff + ftt*c_ftt + ftf*c_ftf + fft*c_fft + fff*c_fff
    return choice_prob

  def forward(self, x, c_vec, return_marginals=False):
    embeds = self.word_embeddings(x)
    embeds = self.embedding_dropout(embeds)
    seq_lengths = torch.count_nonzero(x, dim=1)
    packed_embeds = pack_padded_sequence(embeds, seq_lengths, batch_first=True, enforce_sorted=False)
    lstm_out, (h_n, c_n) = self.lstm(packed_embeds)
    final_out = self.lstm_dropout(h_n[-1])
    rep = self.description_representation(final_out)
    rep = self.rep_dropout(rep)
    cov_vector = self.covariance(final_out)
    cov_matrix = torch.reshape(cov_vector, (cov_vector.shape[0], self.colour_vector_dim, self.colour_vector_dim))

    marginals = []
    for i in range(3):
      cv = c_vec[:, i, :]
      # N.B. First dimension (dim=0) is the batch size, so operations on the individual matrices start at index 1
      delta = torch.unsqueeze(cv - rep, dim=2)
      delta_t = torch.transpose(delta, dim0=2, dim1=1)
      score = -torch.matmul(torch.matmul(delta_t, cov_matrix), delta)
      score = score.squeeze(dim=-1)

      # `score` is generally negative but can sometimes be positive (due to positive semi-definite matrix),
      # so clamp value so that marginal prob doesn't exceed 1
      if self.correlation == Correlation.INDEPENDENT_MOD or self.correlation == Correlation.MAXIMUM_MOD:
        # Modified joint probability can have close to zero probability mass, resulting in exploding gradients, so clamp the score further
        score = score.clamp(max=0, min=-20)
      else:
        score = score.clamp(max=0)
      marginal = torch.exp(score)
      marginals.append(marginal)
    choices = self.joint_choice(marginals)
    choices = torch.clamp(choices, min=1e-40)
    if return_marginals:
      return torch.log(choices), marginals
    else:
      return torch.log(choices)

  def predict(self, x, c_vec):
    self.eval()
    with torch.no_grad():
      forward = self(x, c_vec)
    return torch.exp(forward)