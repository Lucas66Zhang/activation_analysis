import torch
from torch import nn

class ConfidenceNet_lstm(nn.Module):
    def __init__(self, embed_dim, lstm_hidden_size, lstm_num_layers,
                 bias=True, batch_first=True, dropout=0.0, bidirectional=False,
                 decoder_num_layers=2, decoder_hidden_size=256, decoder_act='relu'):
        super(ConfidenceNet_lstm, self).__init__()
        self.x_fc1_encoder = nn.Linear(120, embed_dim)
        self.x_fc2_encoder = nn.Linear(84, embed_dim)
        self.x_fc3_encoder = nn.Linear(10, embed_dim)
        self.maxpool2_encoder = nn.Linear(256, embed_dim)
        self.batch_norm = [nn.BatchNorm1d(decoder_hidden_size) for _ in range(decoder_num_layers - 1)]

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            bias=bias,
                            batch_first=batch_first,
                            dropout=dropout,
                            bidirectional=bidirectional)

        # activation function

        # activation function
        if decoder_act == 'relu':
            self._activation = nn.ReLU()
        elif decoder_act == 'leaky_relu':
            self._activation = nn.LeakyReLU()
        elif decoder_act == 'elu':
            self._activation = nn.ELU()
        elif decoder_act == 'gelu':
            self._activation = nn.GELU()
        else:
            raise ValueError('activation must be relu, leaky_relu, elu or gelu')

        decoder_layers = [nn.Linear(lstm_hidden_size, decoder_hidden_size), self.batch_norm[0]]
        decoder_layers.append(self._activation)
        for _ in range(decoder_num_layers - 2):
            decoder_layers.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))
            decoder_layers.append(self._activation)
            decoder_layers.append(self.batch_norm[_ + 1])
        decoder_layers.append(nn.Linear(decoder_hidden_size, 1))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self,x_maxpool2, x_fc1, x_fc2, x_fc3):
        x_fc1 = self.x_fc1_encoder(x_fc1).unsqueeze(1)
        x_fc2 = self.x_fc2_encoder(x_fc2).unsqueeze(1)
        x_fc3 = self.x_fc3_encoder(x_fc3).unsqueeze(1)
        x_maxpool2 = self.maxpool2_encoder(x_maxpool2).unsqueeze(1)

        x = torch.cat((x_maxpool2, x_fc1, x_fc2, x_fc3), dim=1)

        x, _ = self.lstm(x)
        x = self.decoder(x[:, -1, :].squeeze(1))
        return x
    
class ConfidenceNet_lstm_cls(nn.Module):
    def __init__(self, embed_dim, lstm_hidden_size, lstm_num_layers,
                 bias=True, batch_first=True, dropout=0.0, bidirectional=False,
                 decoder_num_layers=2, decoder_hidden_size=256, decoder_act='relu'):
        super(ConfidenceNet_lstm_cls, self).__init__()
        self.x_fc1_encoder = nn.Linear(120, embed_dim)
        self.x_fc2_encoder = nn.Linear(84, embed_dim)
        self.x_fc3_encoder = nn.Linear(10, embed_dim)
        self.maxpool2_encoder = nn.Linear(256, embed_dim)
        self.batch_norm = [nn.BatchNorm1d(decoder_hidden_size) for _ in range(decoder_num_layers - 1)]

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            bias=bias,
                            batch_first=batch_first,
                            dropout=dropout,
                            bidirectional=bidirectional)

        # activation function

        # activation function
        if decoder_act == 'relu':
            self._activation = nn.ReLU()
        elif decoder_act == 'leaky_relu':
            self._activation = nn.LeakyReLU()
        elif decoder_act == 'elu':
            self._activation = nn.ELU()
        elif decoder_act == 'gelu':
            self._activation = nn.GELU()
        else:
            raise ValueError('activation must be relu, leaky_relu, elu or gelu')

        decoder_layers = [nn.Linear(lstm_hidden_size, decoder_hidden_size), self.batch_norm[0]]
        decoder_layers.append(self._activation)
        for _ in range(decoder_num_layers - 2):
            decoder_layers.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))
            decoder_layers.append(self._activation)
            decoder_layers.append(self.batch_norm[_ + 1])
        decoder_layers.append(nn.Linear(decoder_hidden_size, 2))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self,x_maxpool2, x_fc1, x_fc2, x_fc3):
        x_fc1 = self.x_fc1_encoder(x_fc1).unsqueeze(1)
        x_fc2 = self.x_fc2_encoder(x_fc2).unsqueeze(1)
        x_fc3 = self.x_fc3_encoder(x_fc3).unsqueeze(1)
        x_maxpool2 = self.maxpool2_encoder(x_maxpool2).unsqueeze(1)

        x = torch.cat((x_maxpool2, x_fc1, x_fc2, x_fc3), dim=1)

        x, _ = self.lstm(x)
        x = self.decoder(x[:, -1, :].squeeze(1))
        return x

class ConfidenceNet_rnn(nn.Module):
    def __init__(self, embed_dim, rnn_hidden_size, rnn_num_layers, nonlinearity,
                 bias=True, batch_first=True, dropout=0.0, bidirectional=False,
                 decoder_num_layers=2, decoder_hidden_size=256, decoder_act='relu'):
        super(ConfidenceNet_rnn, self).__init__()
        self.x_fc1_encoder = nn.Linear(120, embed_dim)
        self.x_fc2_encoder = nn.Linear(84, embed_dim)
        self.x_fc3_encoder = nn.Linear(10, embed_dim)
        self.maxpool2_encoder = nn.Linear(256, embed_dim)
        self.batch_norm = [nn.BatchNorm1d(decoder_hidden_size) for _ in range(decoder_num_layers - 1)]

        self.rnn = nn.RNN(input_size=embed_dim,
                          hidden_size=rnn_hidden_size,
                          num_layers=rnn_num_layers,
                          nonlinearity=nonlinearity,
                          bias=bias,
                          batch_first=batch_first,
                          dropout=dropout,
                          bidirectional=bidirectional)

        # activation function
        if decoder_act == 'relu':
            self._activation = nn.ReLU()
        elif decoder_act == 'leaky_relu':
            self._activation = nn.LeakyReLU()
        elif decoder_act == 'elu':
            self._activation = nn.ELU()
        elif decoder_act == 'gelu':
            self._activation = nn.GELU()
        else:
            raise ValueError('activation must be relu, leaky_relu, elu or gelu')

        decoder_layers = [nn.Linear(rnn_hidden_size, decoder_hidden_size), self.batch_norm[0]]
        decoder_layers.append(self._activation)
        for _ in range(decoder_num_layers - 2):
            decoder_layers.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))
            decoder_layers.append(self._activation)
            decoder_layers.append(self.batch_norm[_ + 1])
        decoder_layers.append(nn.Linear(decoder_hidden_size, 1))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self,x_maxpool2, x_fc1, x_fc2, x_fc3):
        x_maxpool2 = self.maxpool2_encoder(x_maxpool2).unsqueeze(1)
        x_fc1 = self.x_fc1_encoder(x_fc1).unsqueeze(1)
        x_fc2 = self.x_fc2_encoder(x_fc2).unsqueeze(1)
        x_fc3 = self.x_fc3_encoder(x_fc3).unsqueeze(1)

        x = torch.cat((x_maxpool2, x_fc1, x_fc2, x_fc3), dim=1)

        x, _ = self.rnn(x)
        x = self.decoder(x[:, -1, :].squeeze(1))

        return x
    
class ConfidenceNet_rnn_cls(nn.Module):
    def __init__(self, embed_dim, rnn_hidden_size, rnn_num_layers, nonlinearity,
                 bias=True, batch_first=True, dropout=0.0, bidirectional=False,
                 decoder_num_layers=2, decoder_hidden_size=256, decoder_act='relu'):
        super(ConfidenceNet_rnn_cls, self).__init__()
        self.x_fc1_encoder = nn.Linear(120, embed_dim)
        self.x_fc2_encoder = nn.Linear(84, embed_dim)
        self.x_fc3_encoder = nn.Linear(10, embed_dim)
        self.maxpool2_encoder = nn.Linear(256, embed_dim)
        self.batch_norm = [nn.BatchNorm1d(decoder_hidden_size) for _ in range(decoder_num_layers - 1)]

        self.rnn = nn.RNN(input_size=embed_dim,
                          hidden_size=rnn_hidden_size,
                          num_layers=rnn_num_layers,
                          nonlinearity=nonlinearity,
                          bias=bias,
                          batch_first=batch_first,
                          dropout=dropout,
                          bidirectional=bidirectional)

        # activation function
        if decoder_act == 'relu':
            self._activation = nn.ReLU()
        elif decoder_act == 'leaky_relu':
            self._activation = nn.LeakyReLU()
        elif decoder_act == 'elu':
            self._activation = nn.ELU()
        elif decoder_act == 'gelu':
            self._activation = nn.GELU()
        else:
            raise ValueError('activation must be relu, leaky_relu, elu or gelu')

        decoder_layers = [nn.Linear(rnn_hidden_size, decoder_hidden_size), self.batch_norm[0]]
        decoder_layers.append(self._activation)
        for _ in range(decoder_num_layers - 2):
            decoder_layers.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))
            decoder_layers.append(self._activation)
            decoder_layers.append(self.batch_norm[_ + 1])
        decoder_layers.append(nn.Linear(decoder_hidden_size, 2))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self,x_maxpool2, x_fc1, x_fc2, x_fc3):
        x_maxpool2 = self.maxpool2_encoder(x_maxpool2).unsqueeze(1)
        x_fc1 = self.x_fc1_encoder(x_fc1).unsqueeze(1)
        x_fc2 = self.x_fc2_encoder(x_fc2).unsqueeze(1)
        x_fc3 = self.x_fc3_encoder(x_fc3).unsqueeze(1)

        x = torch.cat((x_maxpool2, x_fc1, x_fc2, x_fc3), dim=1)

        x, _ = self.rnn(x)
        x = self.decoder(x[:, -1, :].squeeze(1))

        return x

class ConfidenceNet_gru(nn.Module):
    def __init__(self, embed_dim, gru_hidden_size, gru_num_layers,
               bias=True, batch_first=True, dropout=0.0, bidirectional=False,
               decoder_num_layers=2, decoder_hidden_size=256, decoder_act='relu'):
        super(ConfidenceNet_gru, self).__init__()
        self.x_fc1_encoder = nn.Linear(120, embed_dim)
        self.x_fc2_encoder = nn.Linear(84, embed_dim)
        self.x_fc3_encoder = nn.Linear(10, embed_dim)
        self.maxpool2_encoder = nn.Linear(256, embed_dim)
        self.batch_norm = [nn.BatchNorm1d(decoder_hidden_size) for _ in range(decoder_num_layers - 1)]

        self.gru = nn.GRU(input_size=embed_dim,
                          hidden_size=gru_hidden_size,
                          num_layers=gru_num_layers,
                          bias=bias,
                          batch_first=batch_first,
                          dropout=dropout,
                          bidirectional=bidirectional)

        # activation function
        if decoder_act == 'relu':
            self._activation = nn.ReLU()
        elif decoder_act == 'leaky_relu':
            self._activation = nn.LeakyReLU()
        elif decoder_act == 'elu':
            self._activation = nn.ELU()
        elif decoder_act == 'gelu':
            self._activation = nn.GELU()
        else:
            raise ValueError('activation must be relu, leaky_relu, elu or gelu')

        decoder_layers = [nn.Linear(gru_hidden_size, decoder_hidden_size), self.batch_norm[0]]
        decoder_layers.append(self._activation)
        for _ in range(decoder_num_layers - 2):
            decoder_layers.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))
            decoder_layers.append(self._activation)
            decoder_layers.append(self.batch_norm[_ + 1])
        decoder_layers.append(nn.Linear(decoder_hidden_size, 1))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self,x_maxpool2, x_fc1, x_fc2, x_fc3):
        x_maxpool2 = self.maxpool2_encoder(x_maxpool2).unsqueeze(1)
        x_fc1 = self.x_fc1_encoder(x_fc1).unsqueeze(1)
        x_fc2 = self.x_fc2_encoder(x_fc2).unsqueeze(1)
        x_fc3 = self.x_fc3_encoder(x_fc3).unsqueeze(1)

        x = torch.cat((x_maxpool2, x_fc1, x_fc2, x_fc3), dim=1)

        x, _ = self.gru(x)
        x = self.decoder(x[:, -1, :].squeeze(1))

        return x

class ConfidenceNet_gru_cls(nn.Module):
    def __init__(self, embed_dim, gru_hidden_size, gru_num_layers,
               bias=True, batch_first=True, dropout=0.0, bidirectional=False,
               decoder_num_layers=2, decoder_hidden_size=256, decoder_act='relu'):
        super(ConfidenceNet_gru_cls, self).__init__()
        self.x_fc1_encoder = nn.Linear(120, embed_dim)
        self.x_fc2_encoder = nn.Linear(84, embed_dim)
        self.x_fc3_encoder = nn.Linear(10, embed_dim)
        self.maxpool2_encoder = nn.Linear(256, embed_dim)
        self.batch_norm = [nn.BatchNorm1d(decoder_hidden_size) for _ in range(decoder_num_layers - 1)]

        self.gru = nn.GRU(input_size=embed_dim,
                          hidden_size=gru_hidden_size,
                          num_layers=gru_num_layers,
                          bias=bias,
                          batch_first=batch_first,
                          dropout=dropout,
                          bidirectional=bidirectional)

        # activation function
        if decoder_act == 'relu':
            self._activation = nn.ReLU()
        elif decoder_act == 'leaky_relu':
            self._activation = nn.LeakyReLU()
        elif decoder_act == 'elu':
            self._activation = nn.ELU()
        elif decoder_act == 'gelu':
            self._activation = nn.GELU()
        else:
            raise ValueError('activation must be relu, leaky_relu, elu or gelu')

        decoder_layers = [nn.Linear(gru_hidden_size, decoder_hidden_size), self.batch_norm[0]]
        decoder_layers.append(self._activation)
        for _ in range(decoder_num_layers - 2):
            decoder_layers.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))
            decoder_layers.append(self._activation)
            decoder_layers.append(self.batch_norm[_ + 1])
        decoder_layers.append(nn.Linear(decoder_hidden_size, 2))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self,x_maxpool2, x_fc1, x_fc2, x_fc3):
        x_maxpool2 = self.maxpool2_encoder(x_maxpool2).unsqueeze(1)
        x_fc1 = self.x_fc1_encoder(x_fc1).unsqueeze(1)
        x_fc2 = self.x_fc2_encoder(x_fc2).unsqueeze(1)
        x_fc3 = self.x_fc3_encoder(x_fc3).unsqueeze(1)

        x = torch.cat((x_maxpool2, x_fc1, x_fc2, x_fc3), dim=1)

        x, _ = self.gru(x)
        x = self.decoder(x[:, -1, :].squeeze(1))

        return x

class ConfidenceNet_transformer(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward, num_transformer_layers, dropout=0.1, activation="relu",
               layer_norm_eps=1e-5, batch_first=True, norm_first=False, bias=True,
               decoder_num_layers=2, decoder_hidden_size=256, decoder_act='relu'):
        super(ConfidenceNet_transformer, self).__init__()
        self.x_fc1_encoder = nn.Linear(120, embed_dim)
        self.x_fc2_encoder = nn.Linear(84, embed_dim)
        self.x_fc3_encoder = nn.Linear(10, embed_dim)
        self.maxpool2_encoder = nn.Linear(256, embed_dim)
        self.batch_norm = [nn.BatchNorm1d(decoder_hidden_size) for _ in range(decoder_num_layers - 1)]

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation=activation,
                                                   layer_norm_eps=layer_norm_eps, batch_first=batch_first,
                                                   norm_first=norm_first, bias=bias)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # activation function
        if decoder_act == 'relu':
            self._activation = nn.ReLU()
        elif decoder_act == 'leaky_relu':
            self._activation = nn.LeakyReLU()
        elif decoder_act == 'elu':
            self._activation = nn.ELU()
        elif decoder_act == 'gelu':
            self._activation = nn.GELU()
        else:
            raise ValueError('activation must be relu, leaky_relu, elu or gelu')

        decoder_layers = [nn.Linear(embed_dim, decoder_hidden_size), self.batch_norm[0]]
        decoder_layers.append(self._activation)
        for _ in range(decoder_num_layers - 2):
            decoder_layers.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))
            decoder_layers.append(self._activation)
            decoder_layers.append(self.batch_norm[_ + 1])
        decoder_layers.append(nn.Linear(decoder_hidden_size, 1))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self,x_maxpool2, x_fc1, x_fc2, x_fc3):
        x_maxpool2 = self.maxpool2_encoder(x_maxpool2).unsqueeze(1)
        x_fc1 = self.x_fc1_encoder(x_fc1).unsqueeze(1)
        x_fc2 = self.x_fc2_encoder(x_fc2).unsqueeze(1)
        x_fc3 = self.x_fc3_encoder(x_fc3).unsqueeze(1)

        x = torch.cat((x_maxpool2, x_fc1, x_fc2, x_fc3), dim=1)

        x = self.transformer_encoder(x)[:, -1, :].squeeze(1)
        x = self.decoder(x)

        return x

class ConfidenceNet_transformer_cls(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward, num_transformer_layers, dropout=0.1, activation="relu",
               layer_norm_eps=1e-5, batch_first=True, norm_first=False, bias=True,
               decoder_num_layers=2, decoder_hidden_size=256, decoder_act='relu'):
        super(ConfidenceNet_transformer_cls, self).__init__()
        self.x_fc1_encoder = nn.Linear(120, embed_dim)
        self.x_fc2_encoder = nn.Linear(84, embed_dim)
        self.x_fc3_encoder = nn.Linear(10, embed_dim)
        self.maxpool2_encoder = nn.Linear(256, embed_dim)
        self.batch_norm = [nn.BatchNorm1d(decoder_hidden_size) for _ in range(decoder_num_layers - 1)]

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation=activation,
                                                   layer_norm_eps=layer_norm_eps, batch_first=batch_first,
                                                   norm_first=norm_first, bias=bias)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # activation function
        if decoder_act == 'relu':
            self._activation = nn.ReLU()
        elif decoder_act == 'leaky_relu':
            self._activation = nn.LeakyReLU()
        elif decoder_act == 'elu':
            self._activation = nn.ELU()
        elif decoder_act == 'gelu':
            self._activation = nn.GELU()
        else:
            raise ValueError('activation must be relu, leaky_relu, elu or gelu')

        decoder_layers = [nn.Linear(embed_dim, decoder_hidden_size), self.batch_norm[0]]
        decoder_layers.append(self._activation)
        for _ in range(decoder_num_layers - 2):
            decoder_layers.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))
            decoder_layers.append(self._activation)
            decoder_layers.append(self.batch_norm[_ + 1])
        decoder_layers.append(nn.Linear(decoder_hidden_size, 2))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self,x_maxpool2, x_fc1, x_fc2, x_fc3):
        x_maxpool2 = self.maxpool2_encoder(x_maxpool2).unsqueeze(1)
        x_fc1 = self.x_fc1_encoder(x_fc1).unsqueeze(1)
        x_fc2 = self.x_fc2_encoder(x_fc2).unsqueeze(1)
        x_fc3 = self.x_fc3_encoder(x_fc3).unsqueeze(1)

        x = torch.cat((x_maxpool2, x_fc1, x_fc2, x_fc3), dim=1)

        x = self.transformer_encoder(x)[:, -1, :].squeeze(1)
        x = self.decoder(x)

        return x



