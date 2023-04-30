import torch

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