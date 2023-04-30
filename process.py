from enum import Enum
import random
import torch
import torch.nn.functional as F

class GenerationMethod(Enum):
  GREEDY = 1
  BEAMSEARCH = 2
  SAMPLE = 3

class UtteranceFactory():
  def __init__(self, max_seq_len, pad_token, start_token, end_token, vocab):
    self.max_seq_len = max_seq_len
    self.pad_token = pad_token
    self.start_token = start_token
    self.end_token = end_token
    self.vocab = vocab
  
  def generate_utterance(self, model, c_vec, method=GenerationMethod.SAMPLE, return_index=False):
    '''
      `c_vec`: colour vectors tensor of dim (N, 3, COLOUR_VECTOR_DIM)
      `return_index`: if True, returns the token indices based on `vocab` instead of the token strings
    '''
    num_samples = c_vec.shape[0]
    start = torch.tensor(self.vocab([self.start_token]))
    start = start.repeat((num_samples, 1))
    next_token = start
    model.eval()

    tokens = []
    h_n = None
    c_n = None
    while len(tokens) < self.max_seq_len:
      with torch.no_grad():
        y_hat, (h_n, c_n) = model(next_token, c_vec, h_n, c_n)
      
      if method == GenerationMethod.SAMPLE:
        prob = F.softmax(y_hat[0].squeeze(), dim=0).tolist()
        pred = random.choices(range(len(prob)), prob)[0]
      elif method == GenerationMethod.GREEDY:
        pred = torch.argmax(y_hat[0], dim=1).item()
      else:
        raise NotImplementedError
      
      token = self.vocab.lookup_token(pred)
      if token == self.end_token:
        break
      
      if return_index:
        tokens.append(pred)
      else:
        tokens.append(token)
      next_token = torch.unsqueeze(torch.tensor([pred]), dim=0)
    return tokens

  def generate_utterances(self, model, c_vec, method=GenerationMethod.SAMPLE, return_index=False):
    num_samples = c_vec.shape[0]
    model.eval()

    preds = [self.vocab([self.start_token]) for _ in range(num_samples)]
    tokens = [[] for _ in range(num_samples)]
    ended = [False for _ in range(num_samples)]

    h_n = None
    c_n = None

    while all([len(x) < self.max_seq_len for x in preds]) and not all(ended):
      next_token = torch.tensor([[pred[-1]] for pred in preds])
      with torch.no_grad():
        y_hat, (h_n, c_n) = model(next_token, c_vec, h_n, c_n)

      prob = F.softmax(y_hat[0], dim=1)
      vocab_len = prob.shape[1]
      
      # Always set the probability of PAD_TOKEN to 0
      pad_token_idx = self.vocab.__getitem__(self.pad_token)
      prob[:, pad_token_idx] = 0
      
      # Set the probability of END_TOKEN to 0 if this is the first round of generation so as to avoid null utterances
      if len(tokens[0]) == 0:
        end_token_idx = self.vocab.__getitem__(self.end_token)
        prob[:, end_token_idx] = 0

      for i in range(num_samples):
        if ended[i]:
          continue
        
        if method == GenerationMethod.SAMPLE:
          pred = random.choices(range(vocab_len), prob[i].tolist())[0]
        elif method == GenerationMethod.GREEDY:
          pred = torch.argmax(prob[i]).item()
        else:
          raise NotImplementedError

        token = self.vocab.lookup_token(pred)
        if token == self.end_token:
          ended[i] = True
          continue
        tokens[i].append(token)
        preds[i].append(pred)
    # Remove the zeroth element in each pred before returning as they are START_TOKENs
    indexes = [pred[1:] for pred in preds]
    return indexes if return_index else tokens


class ListenerMetrics():
  def listener_accuracy(y_hat):
    pred = torch.argmax(y_hat, dim=1)
    correct_pred = torch.count_nonzero(pred == 0).item()
    return (correct_pred, )
    
  def prob_listener_score(y_hat):
    rounding_decimals = 20
    target_idx = 0
    distractor_1_idx = 1
    distractor_2_idx = 2

    score = 0
    argmax = 0
    pairs = 0
    triplets = 0
    for i in range(y_hat.shape[0]):
      rounded = y_hat[i].round(decimals=rounding_decimals)
      target_val = rounded[target_idx].item()
      max_other_val = max(rounded[distractor_1_idx].item(), rounded[distractor_2_idx].item())
      min_other_val = min(rounded[distractor_1_idx].item(), rounded[distractor_2_idx].item())

      if target_val == max_other_val:
        if target_val == min_other_val:
          # 1/3 chance for all three colours
          score += 1/3
          triplets += 1
        else:
          # 50-50 between target and one other colour
          score += 1/2
          pairs += 1
      elif torch.argmax(rounded) == target_idx:
        score += 1
        argmax += 1
    return (score, argmax, pairs, triplets)
    
class ListenerProcess():
  def train(inputs, y, model, optimiser, criterion, metrics_fn):
    x = inputs[0]
    c_vec = inputs[1]

    model.train()
    optimiser.zero_grad()
    y_hat = model(x, c_vec)
    loss = criterion(y_hat, y)
    loss.backward()
    optimiser.step()

    metrics = metrics_fn(y_hat)
    return loss.data, metrics

  def eval(val_dataloader, model, criterion, metrics_fn):
    model.eval()
    val_loss = 0
    val_samples = 0
    val_correct = 0
    val_accuracy = 0
    with torch.no_grad():
      for inputs, y in val_dataloader:
        x = inputs[0]
        c_vec = inputs[1]
        sample_size = y.shape[0]
        
        y_hat = model(x, c_vec)
        loss = criterion(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        metrics = metrics_fn(y_hat)
        correct_pred = metrics[0]
        
        val_loss += loss.data * sample_size
        val_samples += sample_size
        val_correct += correct_pred
    if val_samples > 0:
      val_loss = val_loss / val_samples
      val_accuracy = val_correct / val_samples
    return val_loss, val_accuracy

  def train_eval(train_dataloader, val_dataloader, model, criterion, optimiser, epochs, metrics_fn):
    for epoch in range(epochs):
      train_loss = 0
      train_samples = 0
      train_correct = 0
      for inputs, y in train_dataloader:
        batch_loss, batch_metrics = ListenerProcess.train(inputs, y, model, optimiser, criterion, metrics_fn)
        sample_size = y.shape[0]
        train_loss += batch_loss * sample_size
        train_samples += sample_size
        train_correct += batch_metrics[0]

      # Compute metrics for this epoch
      train_accuracy = 0
      if train_samples > 0:
        train_loss = train_loss / train_samples
        train_accuracy = train_correct / train_samples

      val_loss, val_accuracy = ListenerProcess.eval(val_dataloader, model, criterion, metrics_fn)
      print(f'[Epoch {epoch+1}] Train Metrics - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}; Validation Metrics - Loss:{val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
    return val_loss, val_accuracy
    
    
    
class SpeakerProcess():
  def eval(s0_model, l0_model, speaker_dataloader, utterance_factory, num_sampled_utt=8):
    total_samples = 0
    gen_correct = 0
    for (batch_x, batch_c_vec, _), _ in speaker_dataloader:
      num_samples = batch_c_vec.shape[0]
      total_samples += num_samples
      
      # Duplicate c_vec to repeat the utterance sampling process by `num_sampled_utt` times
      c_vec = batch_c_vec.repeat((num_sampled_utt, 1, 1))
      generated_utts = utterance_factory.generate_utterances(s0_model, c_vec, method=GenerationMethod.SAMPLE, return_index=True)
      
      # Pad the generated utterances so it can be fed to the listener model as a batch
      max_utt_len = max(len(utt) for utt in generated_utts)
      padded_utts = [F.pad(torch.tensor(utt), (0, max_utt_len-len(utt))) for utt in generated_utts]
      padded_utts = torch.stack(padded_utts)

      gen_y_hat = l0_model.predict(padded_utts, c_vec)
      gen_pred = torch.argmax(gen_y_hat, dim=1)
      gen_correct += torch.count_nonzero(gen_pred == 0).item()

    gen_accuracy = gen_correct / (total_samples * num_sampled_utt)
    return gen_accuracy

  def train_eval(train_dataloader, val_dataloader, model, l0_model, optimiser, epochs, utterance_factory):
    # Evaluate L0 listener accuracy on the groundtruth utterances
    l0_correct = 0
    val_samples = 0
    for (batch_x, batch_c_vec, _), _ in val_dataloader:
      val_samples += batch_c_vec.shape[0]
      batch_x = batch_x[:, 1:] # Remove the zeroth token, which is the <start> token
      y_hat = l0_model.predict(batch_x, batch_c_vec)
      l0_pred = torch.argmax(y_hat, dim=1)
      l0_correct += torch.count_nonzero(l0_pred == 0).item()
    l0_accuracy = l0_correct / val_samples
    print(f'Baseline validation accuracy of l0_model: {l0_accuracy:.4f}')

    for epoch in range(epochs):
      train_loss = 0
      train_samples = 0
      for inputs, y in train_dataloader:
        batch_loss = model.train_model(inputs, y, optimiser)
        sample_size = y.shape[0]
        train_loss += batch_loss * sample_size
        train_samples += sample_size

      # Compute metrics for this epoch
      if train_samples > 0:
        train_loss = train_loss / train_samples
      gen_accuracy = SpeakerProcess.eval(model, l0_model, val_dataloader, utterance_factory)
      print(f'[Epoch {epoch+1}] Metrics - Train Loss: {train_loss:.4f}; L0 Val Accuracy: {gen_accuracy:.4f}')
    return gen_accuracy
    

class PragmaticProcess():
  def pragmatic_eval(dataloader, l0_model, s0_model, metrics_fn, rsa, num_l2_repeats=8, orig_colour_order=False):
    # Fix the random seed for reproducibility
    random.seed(42)
    
    l0_model.eval()
    s0_model.eval()

    total = 0
    l0_correct = 0
    l2_correct = 0
    for (batch_x, batch_c_vec, batch_hsl), _ in dataloader:
      num_samples = batch_x.shape[0]
      total += num_samples
      
      # L0 Accuracy
      y_hat = l0_model(batch_x, batch_c_vec)
      metrics = metrics_fn(y_hat)
      correct_pred = metrics[0]
      l0_correct += correct_pred
      
      for i in range(num_samples):
        x = batch_x[i]
        c_vec = torch.unsqueeze(batch_c_vec[i], dim=0)
        if orig_colour_order:
          # Original speaker model expects the target colour to be the last element of c_vec
          c_vec = c_vec[:, [2, 1, 0]]
        length = torch.count_nonzero(x)
        tokens = vocab.lookup_tokens(x[:length].tolist())
        l2_choice = np.zeros(3)
        for _ in range(num_l2_repeats):
          l2_choice = rsa.pragmatic_listener(tokens, c_vec, l0_model, s0_model, orig_colour_order)
        if np.argmax(l2_choice) == 0:
          l2_correct += 1
      if int(total/num_samples) % 1 == 0:
        print(f"{total} samples processed, {l0_correct} l0 correct, {l2_correct} l2 correct")
        
    l0_accuracy = l0_correct / total
    l2_accuracy = l2_correct / total
    print(f"L0 Accuracy: {l0_accuracy:.3f}, L2 Accuracy: {l2_accuracy:.3f}")
    return l0_accuracy, l2_accuracy