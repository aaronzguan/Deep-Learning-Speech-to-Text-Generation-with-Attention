import os
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import Levenshtein as Lev
from util import *
import random


class LockedDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        # x: (batch_size, seq_len, feature_size)
        if not self.training or not self.dropout:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x


class pBLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_unit='LSTM'):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn, rnn_unit.upper())
        # feature dimension will be doubled because of time resolution reduction
        self.BLSTM = self.rnn_unit(input_size * 2, hidden_size, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, input_x, lens_x):
        """
        BLSTM layer for pBLSTM
        Step 1. Reduce time resolution to half
        Step 2. Run through BLSTM

        :param input_x :(batch_size, T, C) input to the pBLSTM
        :return output: (batch_size, T//2, C*2) encoded sequence from pyramidal Bi-LSTM
        """
        batch_size, timestep, feature_dim = input_x.shape
        # make input len even number
        if timestep % 2 != 0:
            input_x = input_x[:, :-1, :]
            timestep -= 1
        # Reduce time resolution
        input_x = input_x.contiguous().reshape(batch_size, timestep // 2, feature_dim * 2)
        # Bidirectional RNN
        x_packed = pack_padded_sequence(input_x, lens_x // 2, batch_first=True, enforce_sorted=False)
        output_packed, hidden = self.BLSTM(x_packed)
        output, output_lens = pad_packed_sequence(output_packed, batch_first=True)
        return output, output_lens, hidden


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        # Listener RNN layer
        self.num_layers = args.encoder_num_layers
        self.lstm = []
        self.dropout = []
        self.lstm.append(nn.LSTM(input_size=args.encoder_input_size, hidden_size=args.encoder_hidden_size, num_layers=1, bias=False, bidirectional=True))
        self.dropout.append(LockedDropout(dropout=args.encoder_dropout))
        for i in range(1, self.num_layers):
            self.lstm.append(pBLSTMLayer(args.encoder_hidden_size * 2, args.encoder_hidden_size))
            self.dropout.append(LockedDropout(dropout=args.encoder_dropout))
        # For python list of pytorch layer, nn.ModuleList need to convert the python list to Modulelist
        # Such that store submodules parameters in your model when you are using model.to(torch.device('cuda'))
        self.lstm = torch.nn.ModuleList(self.lstm)
        self.dropout = torch.nn.ModuleList(self.dropout)

    def forward(self, x_padded, x_lens):
        """
        :param x_padded: (batch_size, seq_len, feature_dim) Padded input sequence
        :param x_lens: (batch_size, ) Lengths of original input sequence
        :return:
                output (batch_size, seq_len_encoder, encoder_hidden_dim*2)
                Encoded output whose time resolution is reduced while dimension is increased
        """
        output, _ = self.lstm[0](x_padded)
        output = self.dropout[0](output)
        output_lens = x_lens
        for i in range(1, self.num_layers):
            output, output_lens, hidden = self.lstm[i](output, output_lens)
            output = self.dropout[i](output)
        return output, output_lens, hidden


class Attention(nn.Module):
    """
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)

    It is using Scaled dot product attention
    """
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args

    def forward(self, key, value, query, lens):
        '''
        :param key: (batch_size, seq_len_encoder, key_size) Key Projection from Encoder per time step
        :param value: (batch_size, seq_len_encoder, value_size) Value Projection from Encoder per time step
        :param query :(batch_size, key_size) Query is the output of last LSTMCell from Decoder
        :param lens: (batch_size, ) Lengths of input sequences after processed by the Encoder
        :return output: (batch_size, value_size) Weighted-attention Context
        :return attention: (batch_size, seq_len_encoder) Attention vector
        '''
        # key (batch_size, seq_len, key_size), query (batch_size, key_size, 1) -> attention (batch_size, seq_len, 1)
        attention = torch.bmm(key, query.unsqueeze(2)).squeeze(2) / (self.args.key_size**0.5)

        # Create an (batch_size, seq_len_encoder) boolean mask, all padding positions will be 1
        mask = torch.arange(key.size(1)).unsqueeze(0) >= lens.unsqueeze(1)
        mask = mask.to(self.args.device)
        # Set attention logits at padding positions to negative infinity.
        attention.masked_fill_(mask, -1e9)
        # Take softmax over the "source length" dimension.
        attention = nn.functional.softmax(attention, dim=1)
        # Compute attention-weighted sum of context vectors
        # attention (batch_size, seq_len), value (batch_size, seq_len, value_size) -> out (batch_size, 1, value_size)
        output = torch.bmm(attention.unsqueeze(1), value).squeeze(1)

        return output, attention


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_size)
        self.key_network = nn.Linear(args.encoder_hidden_size * 2, args.key_size)
        self.value_network = nn.Linear(args.encoder_hidden_size * 2, args.value_size)
        self.query_network = nn.Linear(args.embedding_size, args.key_size)

        self.attention1 = Attention(args)
        self.attention2 = Attention(args)
        self.lstm1 = nn.LSTMCell(input_size=args.embedding_size + args.value_size, hidden_size=args.decoder_hidden_size)
        self.drop1 = LockedDropout(dropout=args.decoder_dropout)
        self.lstm2 = nn.LSTMCell(input_size=args.decoder_hidden_size, hidden_size=args.key_size)
        self.drop2 = LockedDropout(dropout=args.decoder_dropout)

        self.fc = nn.Linear(args.key_size + args.value_size, args.embedding_size)
        self.tanh = nn.Hardtanh(inplace=True)
        self.character_prob = nn.Linear(args.embedding_size, args.vocab_size)
        self.character_prob.weight = self.embedding.weight

    def forward(self, encoder_output, lens=None, text=None, hidden=None, validation=False):
        """
        :param encoder_output: (batch_size, seq_len_encoder, encoder_hidden_dim*2)
                                Output of the Encoder, the seq_len is also modified by pBLSTM
        :param lens: (batch_size, ) Lengths of the input sequence after processed by the Encoder
        :param text: (batch_size, text_len)  Padded target transcript
        :param hidden: hidden state from the Encoder
        :param validation: validation or training mode
        :return: predictions: Returns the character prediction probability matrix
        """
        # Project the output of encoder to key and value for the attention module
        # The key shape is (batch_size, encoder_seq_len, key_size)
        # The value shape is (batch_size, encoder_seq_len, value_size)
        key = self.key_network(encoder_output)
        value = self.value_network(encoder_output)

        if validation:
            max_len = 250
        else:
            embeddings = self.embedding(text)  # (batch_size, max_len, embed_size)
            max_len = text.shape[1]

        predictions = []
        hidden_states = [hidden, None]
        # Create a (batch_size, 1) all zeros to indicate for each input, the fist element is 0, which is <sos> label
        # Make sure the 0 corresponds with the <sos> label
        prediction = torch.zeros(encoder_output.shape[0], 1).to(self.args.device)

        for i in range(max_len):
            # prediction from the last time step (batch_size, vocab_size)

            if not validation:
                if random.random() > self.args.teacher_forcing_ratio:
                    # Pass in the generated char from the previous time step as input to the current time step
                    # Prediction in shape (batch_size, vocab_size)
                    # if use Gumbel noise, add the gumbel noise to the prediction then pass to the embedding
                    # prediction.argmax(dim=-1) gives the label with maximum probability for each element in batch
                    # prediction.argmax(dim=-1) in shape (batch_size, 1)
                    char_embed = self.embedding(prediction.argmax(dim=-1))
                else:
                    if i == 0:
                        # Make sure the first input for the sequence is <sos>
                        char_embed = self.embedding(prediction.argmax(dim=-1))
                    else:
                        # Other than the first input, if not teacher forcing, then use the ground truth of last time
                        char_embed = embeddings[:, i - 1, :]
            # Don't use teacher force during validation
            # Validation should always use the prediction from the last time step to predict the next
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1))

            # char_embed: (batch_size, embed_size) and output the query: (batch_size, key_size)
            query = self.query_network(char_embed)

            context, attention = self.attention1(key, value, query, lens)  # context: (batch_size, value_size)
            # char_embed: (batch_size, embed_size) context: (batch_size, value_size)
            # Concatenate the weighted context and the char_embedding of the previous time step for current time step
            inp = torch.cat([char_embed, context], dim=1)
            inp = self.drop1(inp.unsqueeze(1)).squeeze(1)
            # The first hidden state is for the first LSTM Cell, which is shape of (batch_size, decoder_hidden_size)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            # The hidden state is [output, (hidden, cell)], therefore the 0-th is actual output
            # Get the output of the first LSTM Cell then pass to the second LSTM Cell
            # The output inp_2 shape is (batch_size, decoder_hidden_size)
            inp_2 = hidden_states[0][0]
            inp_2 = self.drop2(inp_2.unsqueeze(1)).squeeze(1)
            # The second hidden state is for the second LSTM Cell, which is shape of (batch_size, decoder_hidden_size)
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            # The hidden state is [output, (hidden, cell)], therefore the 0-th is actual output
            # Get the output of the first LSTM Cell then pass to the second LSTM Cell
            # The output is (batch_size, key_size)
            output = hidden_states[1][0]
            # Pass the output of the second LSTM cell to attention as the query
            # Get the weighted-context, shape (batch_size, value_size)
            context, attention = self.attention2(key, value, output, lens)

            # Concatenate the output of the second/last LSTM cell and the weighted-context
            # Pass to linear and get the fianl probability matrix
            fc_outputs = self.fc(torch.cat([output, context], dim=1))
            fc_outputs = self.tanh(fc_outputs)
            # output (batch_size, key_size), context (batch_size, value_size)
            # prediction (batch_size, vocab_size)
            prediction = self.character_prob(fc_outputs)

            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)

    def BeamSearch(self, encoder_out, lens=None, hidden=None):
        """
        Beam Search decoding during testing
        Only works if batch size of test dataset is 1.
        Dropout will not be used during testing

        :param encoder_out: (batch_size, seq_len_encoder, encoder_hidden_dim*2)
                            Output of the Encoder, the seq_len is also modified by pBLSTM
        :param lens: (batch_size, ) Lengths of the input sequence after processed by the Encoder
        :param hidden: hidden state from the Encoder
        :return: predictions: Returns the character prediction probability matrix
        """

        max_len = 250

        key = self.key_network(encoder_out)
        value = self.value_network(encoder_out)

        Path = [{'hypothesis': torch.LongTensor([0]).to(self.args.device), 'score': 0, 'hidden_states': [hidden, None],
                 'history_path': []}]
        complete_hypothesis = []

        for idx in range(max_len):
            tmp_path = []
            for path in Path:
                char_embed = self.embedding(path['hypothesis'])

                query = self.query_network(char_embed.squeeze()).unsqueeze(0)

                context, atten = self.attention1(key, value, query, lens)

                inp = torch.cat([char_embed, context], dim=1)

                # extract hidden states from path
                hidden_states = path['hidden_states']

                hidden_states[0] = self.lstm1(inp, hidden_states[0])
                inp_2 = hidden_states[0][0]

                hidden_states[1] = self.lstm2(inp_2, hidden_states[1])
                output = hidden_states[1][0]

                context, atten = self.attention2(key, value, output, lens)

                fc_outputs = self.fc(torch.cat([output, context], dim=1))
                fc_outputs = self.tanh(fc_outputs)
                character_prob_distrib = self.character_prob(fc_outputs)

                local_score, local_idx = torch.topk(nn.functional.log_softmax(character_prob_distrib, dim=1), self.args.beam_width, dim=1)

                for tmp_beam_idx in range(self.args.beam_width):
                    tmp_dict = {}
                    tmp_dict['score'] = path['score'] + local_score[0][tmp_beam_idx]
                    tmp_dict['hypothesis'] = local_idx[:, tmp_beam_idx]
                    tmp_dict['history_path'] = path['history_path'] + [local_idx[:, tmp_beam_idx]]
                    tmp_dict['hidden_states'] = hidden_states[:]
                    tmp_path.append(tmp_dict)

            tmp_path = sorted(tmp_path, key=lambda p: p['score'], reverse=True)[:self.args.beam_width]
            if idx == max_len - 1:
                for path in tmp_path:
                    path['hypothesis'] = 33
                    path['history_path'] = path['history_path'] + [33]

            Path = []
            for path in tmp_path:
                # if the idx is <eos> idx, get it to the complete hypothesis set
                if path['hypothesis'] == 33:
                    normalization = (5 + len(path['history_path'])) ** 0.65 / 6 ** 0.65
                    path['score'] /= normalization
                    complete_hypothesis.append(path)
                # else, store it and compare the score at the end
                else:
                    Path.append(path)

            if len(Path) == 0:
                break

        best_one = sorted(complete_hypothesis, key=lambda p: p['score'], reverse=True)[0]

        return best_one['history_path'][:-1]


class LASModel(nn.Module):
    """
    End-to-end sequence model comprising of Encoder and Decoder
    """
    def __init__(self, args, isTest=False):
        super(LASModel, self).__init__()
        self.args = args
        self.encoder = Encoder(args).to(args.device)
        self.decoder = Decoder(args).to(args.device)
        self.isTest = isTest

    def forward(self, input, input_lens, target=None, validation=False):
        """
        :param input: (batch_size, seq_len, feature_dim) padded input sequence
        :param input_lens: (batch_size, ) original length of input sequence
        :param target: (batch_size, text_len) padded target transcript
        :param validation: validation mode or training mode, which will affect the teacher forcing in Decoder
        :return:
            predict_labels: (batch_size, seq_len, vocab_size), probability matrix
        """
        output, output_lens, hidden = self.encoder(input, input_lens)
        # Change the hidden shape to (batch_size, decoder_hidden_size) to fit the Decoder
        # hidden is originally in shape of (num_layer, batch_size, hidden_units)
        # (batch_size, hidden_units * num_layer) = (batch_size, decoder_hidden_size)
        hidden = (hidden[0].reshape(hidden[0].shape[1], -1), hidden[1].reshape(hidden[1].shape[1], -1))
        if not self.isTest:
            predict_labels = self.decoder(output, output_lens, target, hidden, validation)
        else:
            predict_labels = self.decoder.BeamSearch(output, output_lens, hidden)
        # probability matrix (batch_size, seq_len, vocab_size)
        return predict_labels


class create_model(nn.Module):
    def __init__(self, args, isTest=False):
        super(create_model, self).__init__()
        self.args = args
        self.LASModel = LASModel(args, isTest)
        self.LASModel = self.LASModel.to(args.device)
        # Use reduction='none' to get the loss per element with same shape instead of averaged/summed over observations
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.criterion = self.criterion.to(args.device)
        self.isTest = isTest

    def train_setup(self):
        self.lr = self.args.lr
        self.optimizer = torch.optim.Adam(self.LASModel.parameters(), lr=self.args.lr)
        if self.args.use_reduce_schedule:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=1)
        else:
            self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.decay_steps, gamma=self.args.lr_gamma)
        self.train()

    def forward(self, input, input_lens, target=None, target_lens=None, validation=False):
        # Get the predicted probability from the LAS model
        # predict_labels (Batch_size, seq_len, Vocab_size)
        predict_labels = self.LASModel(input, input_lens, target, validation)

        if not self.isTest:
            # Use greedy decoding to decode the probability and get the predicted transcript
            self.predict_transcipt = greedy_decode(predict_labels)
            # Transform the indexed padded target transcript (only number) to the actual transcript in string
            # It needs to remove all the padding in the padded transcript
            self.target_transcript = labels2str(target, target_lens)
            # Get the Levenshtein distance between each transcript
            edit_distance, total_seq = 0, 0
            for predict_seq, target_seq in zip(self.predict_transcipt, self.target_transcript):
                edit_distance += Lev.distance(target_seq, predict_seq)
                total_seq += 1
            # distance-per-sequence
            edit_distance /= total_seq

            # Get the loss for each single element by setting the reduction='none' in cross-entropy
            loss = self.criterion(predict_labels[:, :target.size(1)].permute(0, 2, 1), target)
            # Generate mask to remove out the padding element in the loss such that they won't be back-propagated
            mask = torch.arange(target.shape[1]).unsqueeze(0) >= target_lens.unsqueeze(1)
            mask = mask.to(self.args.device)
            loss.masked_fill_(mask, 0.0)
            # Get loss-per-word, "word" here is whatever token you have in your vocabulary.
            # For character-level model, the loss-per-word is acutally the loss-per-character.
            # target_lens.sum() is the total seq lens in this batch
            self.loss = loss.sum() / target_lens.sum()
            # get exponential of the loss-per-word
            perplexity = torch.exp(self.loss).item()

            return predict_labels, perplexity, edit_distance
        else:
            return predict_labels

    def get_transcipt(self):
        return self.predict_transcipt, self.target_transcript

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        nn.utils.clip_grad_norm_(self.LASModel.parameters(), 2)
        self.optimizer.step()

    def update_learning_rate(self, dist=None):
        if self.args.use_reduce_schedule:
            self.scheduler.step(dist)
        else:
            self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']

    def get_learning_rate(self):
        return self.lr

    def train(self):
        try:
            self.LASModel.train()
        except:
            print('train() cannot be implemented as model does not exist.')

    def eval(self):
        try:
            self.LASModel.eval()
        except:
            print('eval() cannot be implemented as model does not exist.')

    def save_model(self, which_epoch):
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = os.path.join(self.args.expr_dir, save_filename)
        if torch.cuda.is_available():
            try:
                torch.save(self.LASModel.module.cpu().state_dict(), save_path)
            except:
                torch.save(self.LASModel.cpu().state_dict(), save_path)
        else:
            torch.save(self.LASModel.cpu().state_dict(), save_path)

        self.LASModel.to(self.args.device)

    def load_model(self, model_path):
        self.LASModel.load_state_dict(torch.load(model_path))


