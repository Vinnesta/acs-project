import itertools
import torch
import torch.nn.functional as F

class RSA():
  def __init__(self, num_samples, vocab, utterance_factory):
    self.num_samples = num_samples
    self.vocab = vocab
    self.utterance_factory = utterance_factory
  
  def swap_targets(c_vec, orig_colour_order):
    '''
    `c_vec`: colour vectors tensor of dim (1, 3, COLOUR_VECTOR_DIM)
    `new_target_idx`: index of the new target in {0, 1, 2}
    '''
    if orig_colour_order:
      # Last colour is the target
      c_vec_alt_1 = c_vec[:, [2, 1, 0]]
      c_vec_alt_2 = c_vec[:, [0, 2, 1]]
    else:
      # First colour is the target
      c_vec_alt_1 = c_vec[:, [1, 0, 2]]
      c_vec_alt_2 = c_vec[:, [2, 1, 0]]
    return c_vec_alt_1, c_vec_alt_2

  def literal_listener(self, utt, c_vec, l0_model):
    if isinstance(utt[0], list):
      # `utt` is a batch of utterances
      c_vec = c_vec.repeat((len(utt), 1, 1))

      # Pad the utterances so it can be fed to the listener model as a batch
      max_utt_len = max(len(u) for u in utt)
      padded_utts = [F.pad(torch.tensor(self.vocab(u), dtype=torch.long), (0, max_utt_len-len(u))) for u in utt]
      listener_tokens = torch.stack(padded_utts)
    else:
      listener_tokens = torch.tensor(self.vocab(utt), dtype=torch.long)
      listener_tokens = torch.unsqueeze(listener_tokens, dim=0)
    prob = l0_model.predict(listener_tokens, c_vec)
    return prob

  def sample_utterances(self, c_vec, s0_model):
    # Duplicate c_vec to repeat the utterance sampling process by `num_sampled_utt` times
    c_vec = c_vec.repeat((self.num_samples, 1, 1))
    generated_utts = self.utterance_factory.generate_utterances(s0_model, c_vec, return_index=False)
    return generated_utts

  def pragmatic_speaker(self, utt, c_vec, l0_model, s0_model, orig_colour_order=False):
    '''
    `utt`: a given test utterance that is compared against sample utterances from literal speaker
    `c_vec`: colour vectors tensor of dim (1, 3, COLOUR_VECTOR_DIM), where the last colour is the target
    '''
    applicable_utt = [utt]
    
    # Swap colours in c_vec to make the two distractors the target
    c_vec_alt_1, c_vec_alt_2 = RSA.swap_targets(c_vec, orig_colour_order)
    
    # For each of the colours in c_vec, sample utterances from the literal_speaker to get a multiset
    applicable_utt += self.sample_utterances(c_vec, s0_model)
    applicable_utt += self.sample_utterances(c_vec_alt_1, s0_model)
    applicable_utt += self.sample_utterances(c_vec_alt_2, s0_model)

    # Group sampled utterances
    unique_utts = []
    unique_utt_counts = []
    applicable_utt.sort()
    for utt, g in itertools.groupby(applicable_utt):
      unique_utts.append(utt)
      unique_utt_counts.append(len(list(g)))

    l0_choices = self.literal_listener(unique_utts, c_vec, l0_model)
    target = 2 if orig_colour_order else 0
    prob_mass = l0_choices[:, target].tolist()

    # Weight the probability masses by the number of duplicate utterances
    weighted_prob_mass = [mass * count for mass, count in zip(prob_mass, unique_utt_counts)]
    softmax = [prob / sum(weighted_prob_mass) for prob in weighted_prob_mass]
    
    utt_prob = {}
    for utt, prob in zip(unique_utts, softmax):
      utt_str = ' '.join(utt)
      utt_prob[utt_str] = prob
    return utt_prob

  def pragmatic_listener(self, utt, c_vec, l0_model, s0_model, orig_colour_order=False):
    '''
    Args:
      `utt`: the given utterance
      `c_vec`: colour vectors tensor of dim (1, 3, COLOUR_VECTOR_DIM), where the last colour is the target

    Returns:
      `inferred_prob`: the probabilities for each of the three colours respectively in c_vec
    '''

    obj_prob_mass = []
    
    # Swap colours in c_vec to make the two distractors the "target"
    c_vec_alt_1, c_vec_alt_2 = RSA.swap_targets(c_vec, orig_colour_order)

    speaker_utt_prob_target = self.pragmatic_speaker(utt, c_vec, l0_model, s0_model, orig_colour_order)
    speaker_utt_prob_alt_1 = self.pragmatic_speaker(utt, c_vec_alt_1, l0_model, s0_model, orig_colour_order)
    speaker_utt_prob_alt_2 = self.pragmatic_speaker(utt, c_vec_alt_2, l0_model, s0_model, orig_colour_order)

    # TO DO: Fix handling of utt list vs utt string
    utt_str = ' '.join(utt)
    prob_mass = [speaker_utt_prob_target[utt_str], speaker_utt_prob_alt_1[utt_str], speaker_utt_prob_alt_2[utt_str]]
    inferred_prob = [mass/sum(prob_mass) for mass in prob_mass]
    return inferred_prob