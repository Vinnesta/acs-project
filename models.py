import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Linear, ReLU, Module, Sequential
from torch.nn.utils.rnn import pack_padded_sequence

class LiteralListener(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, colour_vector_dim, vocab_size, dropout=0.0):
    super(LiteralListener, self).__init__()

    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.embedding_dropout = nn.Dropout(p=dropout)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    self.lstm_dropout = nn.Dropout(p=dropout)
    self.description_representation = nn.Linear(hidden_dim, colour_vector_dim)
    self.rep_dropout = nn.Dropout(p=dropout)
    self.covariance = nn.Linear(hidden_dim, colour_vector_dim*colour_vector_dim)

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
    cov_matrix = torch.reshape(cov_vector, (cov_vector.shape[0], colour_vector_dim, colour_vector_dim))
    
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
    self.encoding_dropout = nn.Dropout(p=dropout)
    
    # Layers for generating utterances
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
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

    embeds = self.word_embeddings(x)
    seq_len = embeds.shape[1]

    # Concatenate colour representation to the embeddings of each token
    utterance_input = torch.cat((colours_rep.unsqueeze(1).expand(-1, seq_len, -1), embeds), dim=-1)
    utterance_input = self.encoding_dropout(utterance_input)

    if h_0 is None or c_0 is None:
      lstm_out, (h_n, c_n) = self.utterance_lstm(utterance_input)
    else:
      lstm_out, (h_n, c_n) = self.utterance_lstm(utterance_input, (h_0, c_0))
    final_out = self.lstm_dropout(lstm_out)
    y_hat = []
    for i in range(final_out.shape[1]):
      h_i = final_out[:, i]
      y_hat.append(self.linear(h_i))
    return y_hat, (h_n, c_n)

  def train_model(self, inputs, y, optimiser):
    x = inputs[0]
    c_vec = inputs[1]

    self.train()
    optimiser.zero_grad()
    y_hat, _ = self(x, c_vec)
    total_loss = 0
    for i, y_hat_i in enumerate(y_hat):
      # For the current timestep, which samples in the batch still have non-padded values?
      nonzero_samples = y[:, i].nonzero(as_tuple=True)[0]
      if nonzero_samples.numel() == 0:
        break

      # Only calculate loss for the non-padded values
      loss = F.cross_entropy(y_hat_i[nonzero_samples], y[nonzero_samples, i])
      loss.backward(retain_graph=True)
      total_loss += loss.data
    optimiser.step()
    return total_loss