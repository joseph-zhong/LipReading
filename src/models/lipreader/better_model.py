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

        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_dropout = rnn_dropout

        self.rnn = self.rnn_type(self.frame_dim, self.hidden_size,
                                 num_layers=self.num_layers, bidirectional=self.bidirectional,
                                 batch_first=True, dropout=self.rnn_dropout)

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

        # final_state: (num_layers * num_dir, batch_size, hidden_size) (*2 if LSTM)
        packed_hidden_states, final_state = self.rnn(packed_frames)

        # (batch_size, seq_len, num_dir * hidden_size)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(packed_hidden_states, batch_first=True)

        # (num_layers, batch_size, hidden_size * num_dir) (*2 if LSTM)
        if self.bidirectional:
            final_state = self._cat_directions(final_state)

        hidden_states = hidden_states.index_select(0, restoration_indices)
        if isinstance(final_state, tuple):  # LSTM
            final_state = (final_state[0].index_select(1, restoration_indices),
                           final_state[1].index_select(1, restoration_indices))
        else:
            final_state = final_state.index_select(1, restoration_indices)

        return hidden_states, final_state

    def _cat_directions(self, final_state):
        """
        final_state must come from a bidirectional RNN
        (num_layers * num_dir, batch_size, hidden_size) -->
        (num_layers, batch_size, hidden_size * num_dir)
        """
        def _cat(s):
            return torch.cat([s[0:s.shape[0]:2], s[1:s.shape[0]:2]], dim=2)

        if isinstance(final_state, tuple):  # LSTM
            final_state = tuple(_cat(s) for s in final_state)
        else:
            final_state = _cat(final_state)

        return final_state
