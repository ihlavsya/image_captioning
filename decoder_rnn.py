"""LSTM model, that takes as input
features and classifies word on each time-step"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class DecoderRNN(nn.Module):
    """LSTM model, that takes as input
    features and classifies word on each time-step"""

    def __init__(self, wv_wrapper, embed_size):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        wv = wv_wrapper["wv"]
        vocab_size = len(wv.vectors)
        num_layers = wv_wrapper["num_layers"]
        hidden_size = embed_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights(wv.vectors)

    def init_weights(self, pretrained_weights):
        """Initialize weights."""
        self.embed.weight.data.copy_(torch.tensor(pretrained_weights))
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        # maximum sampling length
        for _ in range(20):
            # (batch_size, 1, hidden_size),
            hiddens, states = self.lstm(inputs, states)
            # (batch_size, vocab_size)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            # (batch_size, 1, embed_size)
            inputs = inputs.unsqueeze(1)
        # (batch_size, 20)
        sampled_ids = torch.cat(sampled_ids, 1)
        return sampled_ids.squeeze()
