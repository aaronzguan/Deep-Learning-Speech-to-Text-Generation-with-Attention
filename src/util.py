chr2idx = {'<sos>': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17,\
'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26, '-': 27, "'": 28, '.': 29, '_': 30, '+': 31, ' ': 32, '<eos>': 33}

idx2chr = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',\
'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ', '<eos>']


def transform_letter_to_index(transcript):
    '''
    :param transcript : Transcripts are the text input
    :return letter_to_index_list : Returns a list for all the transcript sentence to index
    '''

    letter_to_index_list = []
    for utterance in transcript:
        letters = []
        for word in utterance:
            word = word.decode('utf8')
            letters += [chr2idx[character] for character in word]
            letters += [chr2idx[' ']]
        letters += [chr2idx['<eos>']]
        letter_to_index_list.append(letters)

    return letter_to_index_list


def labels2str(labels, label_sizes):
    output = []
    for l, s in zip(labels, label_sizes):
        output.append(''.join(idx2chr[i] for i in l[:s-1]))  # only keep s-1 because remove the last <eos> label
    return output


def greedy_decode(probs):
    # probs: FloatTensor (batch_size, seq_len, vocab_size)
    out = []
    for seq in probs:
        s = ''
        for t in seq:
            c = idx2chr[t.argmax()]
            if c == '<eos>':
                break
            s += c
        out.append(s)
    return out