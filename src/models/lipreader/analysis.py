import torch
import torch.nn.functional as F

from src.data.data_loader import BOS, EOS, PAD

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def get_data(encoder, decoding_step, data_loader, device, char2idx):
    use_ctc = encoder.enable_ctc
    encoder.eval()
    decoding_step.eval()
    y_test = []
    y_pred = []
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
                _, encoder_hidden_states, prev_state = encoder(frames, frame_lens)
            else:
                encoder_hidden_states, prev_state = encoder(frames, frame_lens)

            prev_output = torch.LongTensor([char2idx[BOS]] * batch_size).to(device)
            for i in range(max_label_len):
                input_ = chars[:,i]
                output_log_probs, prev_state = decoding_step(input_, prev_state,
                                                        frame_lens, encoder_hidden_states)
                prev_output = output_log_probs.exp().multinomial(1).squeeze(dim=-1)  # (batch_size, )
                y_test.extend(list(prev_output.reshape(-1).cpu().numpy() if prev_output.is_cuda else prev_output.reshape(-1).numpy()))
                y_pred.extend(list(labels[:,i].reshape(-1).cpu().numpy() if labels.is_cuda else labels[:,i].reshape(-1).numpy()))
    return y_test, y_pred

def get_confusion_matrix(encoder, decoding_step, data_loader, device, char2idx, num_epochs):
    class_names = ['a', 'e', 'i', 'y', 'o', 'u', 'w',
                   'b', 'p', 'm',
                   'f', 'v',
                   't', 'd', 'n', 's', 'z', 'l', 'r',
                   'j',
                   'k', 'q', 'c', 'g', 'x',
                   'h']

    class_names_set = set(class_names)

    y_test, y_pred = get_data(encoder, decoding_step, data_loader, device, char2idx)
    filtered_y_test = []
    filtered_y_pred = []
    idx2char = {val: key for key, val in char2idx.items()}
    for test, pred in zip([idx2char[x] for x in y_test], [idx2char[x] for x in y_pred]):
        if test in class_names_set and pred in class_names_set:
            filtered_y_test.append(test)
            filtered_y_pred.append(pred)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(filtered_y_test, filtered_y_pred, labels=class_names)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(10,10), dpi=100)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                        title='Confusion matrix, without normalization')
    plt.savefig('{}{}'.format(int(num_epochs), '_confusion_matrix.png'))

    # Plot normalized confusion matrix
    plt.figure(figsize=(10,10), dpi=100)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')
    plt.savefig('{}{}'.format(int(num_epochs), '_norm_confusion_matrix.png'))
