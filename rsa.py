import itertools
import torch

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
    listener_tokens = torch.tensor(self.vocab(utt), dtype=torch.long)
    listener_tokens = torch.unsqueeze(listener_tokens, dim=0)
    prob = l0_model.predict(listener_tokens, c_vec).squeeze().tolist()
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

    applicable_utt.sort()
    # Deduplicate sampled utterances
    applicable_utt = list(utt for utt,_ in itertools.groupby(applicable_utt))

    # TO DO: weight repeated utterances
    prob_mass = []
    for utt in applicable_utt:
      l0_choices = self.literal_listener(utt, c_vec, l0_model)
      target = 2 if orig_colour_order else 0
      prob_mass.append(l0_choices[target])
    softmax = [prob / sum(prob_mass) for prob in prob_mass]

    utt_prob = {}
    for utt, prob in zip(applicable_utt, softmax):
      utt_str = ' '.join(utt)
      if utt_str in utt_prob:
        utt_prob[utt_str] += prob
      else:
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