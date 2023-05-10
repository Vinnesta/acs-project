import itertools
import torch
import torch.nn.functional as F

class RSA():
  def __init__(self, vocab, utterance_factory):
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

  def get_applicable_utterances(self, utt, num_samples, c_vec, s0_model, orig_colour_order=False):
    # Swap colours in c_vec to make the two distractors the "target"
    c_vec_alt_1, c_vec_alt_2 = RSA.swap_targets(c_vec, orig_colour_order)

    applicable_utt = [utt]
    # For each of the colours in c_vec, generate utterances from the literal_speaker
    applicable_utt += self.utterance_factory.generate_utterances_beam(s0_model, c_vec, beam_width=num_samples, return_index=False)
    applicable_utt += self.utterance_factory.generate_utterances_beam(s0_model, c_vec_alt_1, beam_width=num_samples, return_index=False)
    applicable_utt += self.utterance_factory.generate_utterances_beam(s0_model, c_vec_alt_2, beam_width=num_samples, return_index=False)
    
    # Remove duplicate utterances since `utt` and same utterances could have been generated across different colours
    applicable_utt.sort()
    applicable_utt = list(u for u, _ in itertools.groupby(applicable_utt))
    return applicable_utt

  def pragmatic_speaker(self, applicable_utt, c_vec, l0_model, orig_colour_order=False):
    '''
    `applicable_utt`: list of applicable utterances for colours in `c_vec`
    `c_vec`: colour vectors tensor of dim (1, 3, COLOUR_VECTOR_DIM)
    '''

    l0_choices = self.literal_listener(applicable_utt, c_vec, l0_model)
    target = 2 if orig_colour_order else 0
    prob_mass = l0_choices[:, target].tolist()
    if sum(prob_mass) == 0:
      softmax = [0] * len(prob_mass)
    else:
      softmax = [prob / sum(prob_mass) for prob in prob_mass]

    utt_prob = {}
    for utt, prob in zip(applicable_utt, softmax):
      utt_str = ' '.join(utt)
      utt_prob[utt_str] = prob
    return utt_prob

  def pragmatic_listener(self, utt, c_vec, l0_model, s0_model, num_samples=8, orig_colour_order=False):
    '''
    Args:
      `utt`: the given utterance
      `c_vec`: colour vectors tensor of dim (1, 3, COLOUR_VECTOR_DIM), where the last colour is the target
      `num_samples`: number of alternative utterances to generate per colour

    Returns:
      `inferred_prob`: the probabilities for each of the three colours respectively in c_vec
    '''

    obj_prob_mass = []
    
    # Swap colours in c_vec to make the two distractors the "target"
    c_vec_alt_1, c_vec_alt_2 = RSA.swap_targets(c_vec, orig_colour_order)
    applicable_utt = self.get_applicable_utterances(utt, num_samples, c_vec, s0_model, orig_colour_order)

    speaker_utt_prob_target = self.pragmatic_speaker(applicable_utt, c_vec, l0_model, orig_colour_order)
    speaker_utt_prob_alt_1 = self.pragmatic_speaker(applicable_utt, c_vec_alt_1, l0_model, orig_colour_order)
    speaker_utt_prob_alt_2 = self.pragmatic_speaker(applicable_utt, c_vec_alt_2, l0_model, orig_colour_order)

    utt_str = ' '.join(utt)
    prob_mass = [speaker_utt_prob_target[utt_str], speaker_utt_prob_alt_1[utt_str], speaker_utt_prob_alt_2[utt_str]]
    if sum(prob_mass) == 0:
      inferred_prob = [1/3, 1/3, 1/3]
    else:
      inferred_prob = [mass/sum(prob_mass) for mass in prob_mass]
    return inferred_prob