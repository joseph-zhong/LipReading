import torch
import torch.nn.functional as F

from src.data.data_loader import BOS, EOS

def inference(encoder, decoding_step, frames, frame_lens, chars, char_lens, device,
          char2idx, beam_width=5, max_label_len=100):
    """
    Assumes that the sequences given all begin with BOS and end with EOS
    data_loader yields:
        frames: FloatTensor
        frame_lens: LongTensor
    """
    batch_size = frames.shape[0]
    idx2char = {val: key for key, val in char2idx.items()}
    outputs = []
    gt = []

    use_ctc = encoder.enable_ctc

    for i in range(batch_size):
        frame = frames[i].unsqueeze(dim=0)
        frame_len = frame_lens[i].unsqueeze(dim=0)
        if use_ctc:
            _, encoder_hidden_states, prev_state = encoder(frame, frame_len)
        else:
            encoder_hidden_states, prev_state = encoder(frame, frame_len)

        prev_output = torch.LongTensor([char2idx[BOS]])[0].to(device)
        output_log_probs, prev_state = decoding_step(prev_output.unsqueeze(dim=0), prev_state,
                                                frame_len, encoder_hidden_states)
        output = []
        output.append(prev_output)
        eos_idx = char2idx[EOS]
        beams = [([], output_log_probs, prev_state, 0)]
        while True:
            new_beam = []
            for history, output_log_probs, prev_state, ll in beams:
                if (len(history) > 0 and history[-1] == eos_idx) or len(history) > max_label_len:
                    new_beam.append((history, output_log_probs, prev_state, ll))
                    continue
                candidates = output_log_probs.exp().multinomial(beam_width, replacement=False).view(-1)
                for candidate in candidates:
                    new_output_log_probs, new_prev_state = decoding_step(candidate.unsqueeze(dim=0), prev_state,
                                            frame_len, encoder_hidden_states)
                    new_beam.append((history + [candidate.item()], new_output_log_probs, new_prev_state,
                                                ll + output_log_probs.view(-1)[candidate.item()]))

            beams = sorted(new_beam, key=lambda path: path[3], reverse=True)[:beam_width]
            brk = True
            for history, _, _, _ in beams:
                if len(history) == 0 or (history[-1] != eos_idx and len(history) <= max_label_len):
                    brk = False
                    break
            if brk:
                break
        output = [char2idx[BOS]] + beams[0][0]
        outputs.append(''.join([idx2char[int(ind)] for ind in output]))
        gt.append(''.join([idx2char[int(ind.item())] for ind in chars[i][:char_lens[i]]]))
    return outputs, gt
