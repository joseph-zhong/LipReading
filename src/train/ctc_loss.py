import torch
import torch.nn.functional as F

def ctc_fallback(encoder_outputs, labels, frame_lens, label_lens, blank):
    assert len(encoder_outputs) == len(labels) == len(frame_lens) == len(label_lens)
    skipped_indices = []
    working_indices = []
    for i, (encoder_output, label, frame_len, label_len) in enumerate(zip(encoder_outputs, labels, frame_lens, label_lens)):
        if torch.isinf(F.ctc_loss(encoder_outputs[i:i+1].transpose(0, 1), labels[i:i+1], frame_lens[i:i+1], label_lens[i:i+1], blank=blank)):
            skipped_indices.append(i)
        else:
            working_indices.append(i)
    return skipped_indices, working_indices

def filter_data_on_len(label_lens, max_len=256):
    skipped_indices = []
    working_indices = []
    for i, label_len in enumerate(label_lens):
        if label_len > max_len:
            skipped_indices.append(i)
        else:
            working_indices.append(i)
    return skipped_indices, working_indices

def transform_data(f, *args):
    return [f(arg) for arg in args]

def ctc_loss(encoder_outputs, labels, frame_lens, label_lens, reduction, device):
    """
    All sorts of stupid restrictions from documentation:

    In order to use CuDNN, the following must be satisfied:
    1. targets must be in concatenated format,
    2. all input_lengths must be T.
    3. blank=0
    4. target_lengths \leq 256,
    5. the integer arguments must be of dtype torch.int32.
    """
    assert (frame_lens[1:] - frame_lens[:-1] >= 0).all()  # assert in increasing len

    # req (5)
    labels, frame_lens, label_lens = transform_data(lambda data: torch.tensor(data, dtype=torch.int32),
                                                    labels, frame_lens, label_lens)

    # req (4)
    skipped_indices, working_indices = filter_data_on_len(label_lens, max_len=256)
    if len(skipped_indices) > 0:
        print('some labels too long, unable to compute CTC...')
        if len(working_indices) == 0:
            print('skipping entire batch')
            return None
        print('skipping indices in batch: ' + str(skipped_indices))
        working_indices = torch.LongTensor(working_indices).to(device)
        (encoder_outputs, labels, frame_lens,
            label_lens) = transform_data(lambda data: data.index_select(0, working_indices),
                                         encoder_outputs, labels, frame_lens, label_lens)

    # frame_lens      1, 1, 2, 3, 3, 3, 4
    # frame_len[1:]   1, 2, 3, 3, 3, 4
    # frame_lebs[:-1] 1, 1, 2, 3, 3, 3
    # diff            0, 1, 1, 0, 0, 1
    # nonzero_idx        1, 2,       5
    # change_points         2, 3,       6
    change_points = (frame_lens[1:] - frame_lens[:-1]).nonzero().squeeze(dim=-1) + 1
    change_points = torch.cat([change_points, torch.LongTensor([len(frame_lens)])]).to(device)  # add last portion

    # req 2
    prev_change_point = 0
    total_loss = 0
    count = 0
    global_encoder_outputs, global_labels, global_frame_lens, global_label_lens = encoder_outputs, labels, frame_lens, label_lens
    for change_point in change_points:
        # we call this a minibatch
        minibatch_size = len(frame_lens)
        (encoder_outputs, labels, frame_lens,
            label_lens) = transform_data(lambda data: data[prev_change_point:change_point],
                                         global_encoder_outputs, global_labels, global_frame_lens, global_label_lens)

        # req 1
        labels = torch.cat([label[:label_len] for label, label_len in zip(labels, label_lens)])
        # req 3; moves up so that we leave idx=0 to blank
        labels = labels + 1

        loss = F.ctc_loss(encoder_outputs.transpose(0, 1), labels, frame_lens, label_lens, blank=0, reduction=reduction)

        if torch.isinf(loss):
            print('inf CTC loss occurred...')
            skipped_indices, working_indices = ctc_fallback(encoder_outputs, labels, frame_lens, label_lens, 0)
            if len(working_indices) == 0:
                print('skipping the entire minibatch')
                continue
            print('skipping indices in minibatch: ' + str(skipped_indices))
            working_indices = torch.LongTensor(working_indices).to(device)
            (encoder_outputs, labels, frame_lens,
                label_lens) = transform_data(lambda data: data.index_select(0, working_indices),
                                             encoder_outputs, labels, frame_lens, label_lens)

            loss = F.ctc_loss(encoder_outputs.transpose(0, 1), labels, frame_lens, label_lens, blank=0, reduction=reduction)
            minibatch_size = len(working_indices)

        if reduction == 'mean':
            loss *= minibatch_size
            count += minibatch_size
        total_loss += loss

        prev_change_point = change_point

    if total_loss == 0:
        # all data points failed
        return None

    return total_loss / count if reduction == 'mean' else total_loss
