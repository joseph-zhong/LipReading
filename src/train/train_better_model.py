import torch
import torch.nn.functional as F

from src.data.data_loader import BOS, EOS, PAD
from .ctc_loss import ctc_loss

def train(encoder, decoding_step, data_loader, opt, device,
          char2idx, teacher_forcing_ratio=1, grad_norm=None):
  """
  Assumes that the sequences given all begin with BOS and end with EOS
  data_loader yields:
      frames: FloatTensor
      frame_lens: LongTensor
      chars: LongTensor
      char_lens: LongTensor
  """
  use_ctc = encoder.enable_ctc

  avg_decoder_loss = 0
  avg_ctc_loss = 0

  encoder.train()
  decoding_step.train()
  for frames, frame_lens, chars, char_lens in data_loader:
    frames, frame_lens, chars, char_lens = frames.to(device), frame_lens.to(device), chars.to(device), char_lens.to(device)
    assert (chars[:,0].squeeze() == char2idx[BOS]).all()
    assert (chars.gather(1, (char_lens - 1).unsqueeze(dim=1)).squeeze() == char2idx[EOS]).all()
    if use_ctc:
      assert (frame_lens >= char_lens).all()  # otherwise ctc loss will produce inf

    labels = chars[:,1:]
    label_lens = char_lens - 1
    assert (labels != char2idx[PAD]).sum() == label_lens.sum()

    batch_size = frames.shape[0]
    max_label_len = label_lens.max()

    decoder_loss = 0
    if use_ctc:
      ctc_loss_ = 0

    frames, frame_lens = frames.to(device), frame_lens.to(device)
    chars, char_lens = chars.to(device), char_lens.to(device)

    if use_ctc:
      encoder_outputs, encoder_hidden_states, prev_state = encoder(frames, frame_lens)

      curr_ctc_loss = ctc_loss(encoder_outputs, labels, frame_lens, label_lens, 'mean', device)
      if curr_ctc_loss is None:
        continue
      ctc_loss_ += curr_ctc_loss
    else:
      encoder_hidden_states, prev_state = encoder(frames, frame_lens)
    prev_output = torch.LongTensor([char2idx[BOS]] * batch_size).to(device)

    for i in range(max_label_len):
      teacher_forcing = torch.rand(1) < teacher_forcing_ratio
      input_ = chars[:,i] if teacher_forcing else prev_output

      output_log_probs, prev_state = decoding_step(input_, prev_state,
                                                   frame_lens, encoder_hidden_states)
      decoder_loss += F.nll_loss(output_log_probs, labels[:,i], ignore_index=char2idx[PAD], reduction='sum')
      prev_output = output_log_probs.exp().multinomial(1).squeeze(dim=-1)

    decoder_loss /= (labels != char2idx[PAD]).sum()

    opt.zero_grad()
    # print(f'\tTraining decoder_loss: {decoder_loss}')
    decoder_loss.backward(retain_graph=use_ctc)
    avg_decoder_loss += decoder_loss.cpu().detach().numpy()

    if use_ctc:
      # print(f'\tTraining ctc_loss: {ctc_loss_}')
      ctc_loss_.backward()
      avg_ctc_loss += ctc_loss_.cpu().detach().numpy()

    if grad_norm is not None:
      torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_norm)
      torch.nn.utils.clip_grad_norm_(decoding_step.parameters(), grad_norm)
    opt.step()

  avg_decoder_loss /= len(data_loader)
  print(f'\tTraining decoder_loss: {avg_decoder_loss}')
  if use_ctc:
    avg_ctc_loss /= len(data_loader)
    print(f'\tTraining ctc_loss: {avg_ctc_loss}')
  return avg_decoder_loss, avg_ctc_loss

def eval(encoder, decoding_step, data_loader, device, char2idx):
  use_ctc = encoder.enable_ctc

  encoder.eval()
  decoding_step.eval()

  decoder_loss = 0
  if use_ctc:
    ctc_loss_ = 0
  correct = 0
  count = 0
  with torch.no_grad():
    for frames, frame_lens, chars, char_lens in data_loader:
      frames, frame_lens, chars, char_lens = frames.to(device), frame_lens.to(device), chars.to(device), char_lens.to(device)
      assert (chars[:,0].squeeze() == char2idx[BOS]).all()
      assert (chars.gather(1, (char_lens - 1).unsqueeze(dim=1)).squeeze() == char2idx[EOS]).all()

      labels = chars[:,1:].to(device)
      label_lens = char_lens - 1
      assert (labels != char2idx[PAD]).sum() == label_lens.sum()

      batch_size = frames.shape[0]
      max_label_len = label_lens.max()

      if use_ctc:
        encoder_outputs, encoder_hidden_states, prev_state = encoder(frames, frame_lens)

        curr_ctc_loss = ctc_loss(encoder_outputs, labels, frame_lens, label_lens, 'sum', device)
        if curr_ctc_loss is None:
          continue
        ctc_loss_ += curr_ctc_loss
      else:
        encoder_hidden_states, prev_state = encoder(frames, frame_lens)

      prev_output = torch.LongTensor([char2idx[BOS]] * batch_size).to(device)
      for i in range(max_label_len):
        input_ = chars[:,i]

        output_log_probs, prev_state = decoding_step(input_, prev_state,
                                                     frame_lens, encoder_hidden_states)
        decoder_loss += F.nll_loss(output_log_probs, labels[:,i], ignore_index=char2idx[PAD], reduction='sum')
        prev_output = output_log_probs.exp().multinomial(1).squeeze(dim=-1)  # (batch_size, )

        mask = labels[:, i] != char2idx[PAD]
        correct += ((prev_output == labels[:, i]) * mask).sum()

      count += (labels != char2idx[PAD]).sum().float()

  decoder_loss /= count
  # print(f'\ttest decoder_loss: {decoder_loss}')

  if use_ctc:
    ctc_loss_ /= len(data_loader)
    # print(f'\ttest ctc_loss: {ctc_loss_}')
  return decoder_loss, correct, count
