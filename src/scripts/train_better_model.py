import torch
import torch.nn.functional as F

from src.data.data_loader import BOS, EOS, PAD

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

    encoder.train()
    decoding_step.train()
    for frames, frame_lens, chars, char_lens in data_loader:
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
            ctc_loss = 0

        frames, frame_lens = frames.to(device), frame_lens.to(device)
        chars, char_lens = chars.to(device), char_lens.to(device)

        if use_ctc:
            encoder_outputs, encoder_hidden_states, prev_state = encoder(frames, frame_lens)
            ctc_loss += F.ctc_loss(encoder_outputs.transpose(0, 1), labels, frame_lens, label_lens, blank=encoder.adj_vocab_size - 1, reduction='mean')
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
        print(f'training decoder_loss: {decoder_loss}')
        decoder_loss.backward(retain_graph=use_ctc)

        if use_ctc:
            print(f'training ctc_loss: {ctc_loss}')
            ctc_loss.backward()

        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_norm)
            torch.nn.utils.clip_grad_norm_(decoding_step.parameters(), grad_norm)
        opt.step()

def eval(encoder, decoding_step, data_loader, device, char2idx):
    use_ctc = encoder.enable_ctc

    encoder.eval()
    decoding_step.eval()

    decoder_loss = 0
    if use_ctc:
        ctc_loss = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for frames, frame_lens, chars, char_lens in data_loader:
            assert (chars[:,0].squeeze() == char2idx[BOS]).all()
            assert (chars.gather(1, (char_lens - 1).unsqueeze(dim=1)).squeeze() == char2idx[EOS]).all()

            labels = chars[:,1:]
            label_lens = char_lens - 1
            assert (labels != char2idx[PAD]).sum() == label_lens.sum()

            batch_size = frames.shape[0]
            max_label_len = label_lens.max()

            frames, frame_lens = frames.to(device), frame_lens.to(device)
            chars, char_lens = chars.to(device), char_lens.to(device)

            if use_ctc:
                encoder_outputs, encoder_hidden_states, prev_state = encoder(frames, frame_lens)
                ctc_loss += F.ctc_loss(encoder_outputs.transpose(0, 1), labels, frame_lens, label_lens, blank=encoder.adj_vocab_size - 1, reduction='sum')
            else:
                encoder_hidden_states, prev_state = encoder(frames, frame_lens)

            prev_output = torch.LongTensor([char2idx[BOS]] * batch_size).to(device)
            for i in range(max_label_len):
                input_ = chars[:,i]

                output_log_probs, prev_state = decoding_step(input_, prev_state,
                                                             frame_lens, encoder_hidden_states)
                decoder_loss += F.nll_loss(output_log_probs, labels[:,i], ignore_index=char2idx[PAD], reduction='sum')
                prev_output = output_log_probs.exp().multinomial(1).squeeze(dim=-1)  # (batch_size, )

                mask = labels[:,i] != char2idx[PAD]
                correct += ((prev_output == labels[:,i]) * mask).sum()

            count += (labels != char2idx[PAD]).sum()

    decoder_loss /= count
    print(f'test decoder_loss: {decoder_loss}')

    if use_ctc:
        ctc_loss /= len(data_loader)
        print(f'test ctc_loss: {ctc_loss}')
    print(f'CER: {(count - correct).float() / count}')
    return decoder_loss, correct, count
