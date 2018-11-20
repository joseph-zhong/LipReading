from allennlp.nn.util import sort_batch_by_length
import torch
import torch.nn as nn
import torch.nn.functional as F

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())

class VideoEncoder(nn.Module):
    def __init__(self, frame_dim, hidden_size,
                 rnn_type=nn.LSTM, num_layers=1, bidirectional=True, rnn_dropout=0):
        super(VideoEncoder).__init__()
        self.rnn = rnn_type(frame_dim, hidden_size,
                            num_layers=num_layers, bidirectional=bidirectional,
                            batch_first=True, dropout=rnn_dropout)

    def forward(self,
                frames: torch.FloatTensor,
                frame_lens: torch.LongTensor):
        """
        frames: (batch_size, seq_len, frame_dim)
        frame_lens: (batch_size, )
        """
        (sorted_frames, sorted_frame_lens,
            restoration_indices, _) = sort_batch_by_length(frames, frame_lens)

        packed_frames = nn.utils.rnn.pack_padded_sequence(sorted_frames,
                            sorted_frame_lens.data.cpu().numpy() if sorted_frame_lens.is_cuda else sorted_frame_lens.data.numpy(),
                            batch_first=True)

        # encoder_final_state: (num_layers * num_dir, batch_size, hidden_size) (*2 if LSTM)
        packed_encoder_hidden_states, encoder_final_state = self.encoder(packed_frames)

        # (batch_size, seq_len, num_dir * hidden_size)
        encoder_hidden_states, _ = nn.utils.rnn.pad_packed_sequence(packed_encoder_hidden_states, batch_first=True)

        encoder_hidden_states = encoder_hidden_states.index_select(0, restoration_indices)
        if isinstance(encoder_final_state, tuple):
            encoder_final_state = (encoder_final_state[0].index_select(1, restoration_indices),
                                   encoder_final_state[1].index_select(1, restoration_indices))
        else:
            encoder_final_state = encoder_final_state.index_select(1, restoration_indices)

        return encoder_hidden_states, encoder_final_state
