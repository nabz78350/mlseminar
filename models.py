import torch
import torch.nn as nn
import math
from typing import Tuple


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is expected to have shape [n_batches, input_size, d_model]
        assert x.shape[1] <= self.pe.shape[1], "Input size is greater than max_len"
        x = x + self.pe[:, :x.shape[1], :]
        return x


class CustomTransformerTimeSeries(nn.Module):
    def __init__(self, input_size, n_feat, hidden_size, num_layers, num_heads, dropout_prob=0.1):
        super(CustomTransformerTimeSeries, self).__init__()
        self.input_size = input_size
        self.n_feat = n_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.timeembed = nn.Linear(1, hidden_size)
        self.fc = nn.Linear(n_feat, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=input_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size),
            num_layers,
        )
        self.output = nn.Linear(hidden_size, n_feat)

    def forward(self, x, t):
        x = self.fc(x)
        x = self.pos_encoder(x)
        x = self.transformer(x + self.timeembed(t).view(-1,1,self.hidden_size))
        #x = x + self.timeembed(t).view(-1,1,self.hidden_size)
        x = self.output(x)
        return x


class ResidualBlockTS(torch.nn.Module):
    """Residual block based on the work of Gorishniy et al., 2023
    (https://arxiv.org/abs/2106.11959).
    We follow the implementation found in
    https://github.com/Yura52/rtdl/blob/main/rtdl/nn/_backbones.py
    This class is for Time-Series data where we add Tranformers to
    encode time-based/feature-based context."""

    def __init__(
        self,
        dim_input: int,
        size_window: int = 10,
        dim_embedding: int = 128,
        dim_feedforward: int = 64,
        nheads_feature: int = 8,
        nheads_time: int = 8,
        num_layers_transformer: int = 1,
    ):
        """Residual block based on the work of Gorishniy et al., 2023
        (https://arxiv.org/abs/2106.11959).
        We follow the implementation found in
        https://github.com/Yura52/rtdl/blob/main/rtdl/nn/_backbones.py
        This class is for Time-Series data where we add Tranformers to
        encode time-based/feature-based context.

        Parameters
        ----------
        dim_input : int
            Input dimension
        size_window : int, optional
            Size of window, by default 10
        dim_embedding : int, optional
            Embedding dimension, by default 128
        dim_feedforward : int, optional
            Feedforward layer dimension, by default 64
        nheads_feature : int, optional
            Number of heads to encode feature-based context, by default 5
        nheads_time : int, optional
            Number of heads to encode time-based context, by default 8
        num_layers_transformer : int, optional
            Number of transformer layer, by default 1
        """
        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(dim_input)

        encoder_layer_time = torch.nn.TransformerEncoderLayer(
            d_model=dim_embedding,
            nhead=nheads_time,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            dropout=0,
        )
        self.time_layer = torch.nn.TransformerEncoder(
            encoder_layer_time, num_layers=num_layers_transformer
        )

        self.linear_out = torch.nn.Linear(dim_embedding, dim_input)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return an output of a residual block

        Parameters
        ----------
        x : torch.Tensor
            Data input
        t : torch.LongTensor
            Noise step

        Returns
        -------
        torch.Tensor
            Data output, noise predicted
        """
        batch_size, size_window, dim_emb = x.shape

        x_emb = self.layer_norm(x)
        x_emb_time = self.time_layer(x_emb)
        t_emb = t.repeat(1, size_window).reshape(batch_size, size_window, dim_emb)

        x_t = x + x_emb_time + t_emb
        x_t = self.linear_out(x_t)

        return x + x_t, x_t


class AutoEncoder(torch.nn.Module):
    """Epsilon_theta model of the Algorithm 1 in
    Ho et al., 2020 (https://arxiv.org/abs/2006.11239).
    This implementation is based on the work of
    Tashiro et al., 2021 (https://arxiv.org/abs/2107.03502).
    Their code: https://github.com/ermongroup/CSDI/blob/main/diff_models.py"""

    def __init__(
        self,
        num_noise_steps: int,
        dim_input: int,
        dim_output: int,
        residual_block: torch.nn.Module,
        dim_embedding: int = 128,
        num_blocks: int = 1,
        p_dropout: float = 0.0,
    ):
        """Epsilon_theta model in Algorithm 1 in
        Ho et al., 2020 (https://arxiv.org/abs/2006.11239)

        Parameters
        ----------
        num_noise_steps : int
            Number of steps in forward/reverse processes
        dim_input : int
            Input dimension
        dim_embedding : int, optional
            Embedding dimension, by default 128
        num_blocks : int, optional
            Number of residual blocks, by default 1
        p_dropout : float, optional
            Dropout probability, by default 0.0
        """
        super().__init__()

        self.layer_x = torch.nn.Linear(dim_input, dim_embedding)

        self.register_buffer(
            "embedding_noise_step",
            self._build_embedding(num_noise_steps, int(dim_embedding / 2)),
            persistent=False,
        )
        self.layer_t_1 = torch.nn.Linear(dim_embedding, dim_embedding)
        self.layer_t_2 = torch.nn.Linear(dim_embedding, dim_embedding)

        self.layer_out_1 = torch.nn.Linear(dim_embedding, dim_embedding)
        self.layer_out_2 = torch.nn.Linear(dim_embedding, dim_output)
        self.dropout_out = torch.nn.Dropout(p_dropout)

        self.residual_layers = torch.nn.ModuleList([residual_block for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        """Predict a noise

        Parameters
        ----------
        x : torch.Tensor
            Data input
        t : torch.LongTensor
            Noise step

        Returns
        -------
        torch.Tensor
            Data output, noise predicted
        """
        # Noise step embedding
        t_emb = torch.as_tensor(self.embedding_noise_step)[t].squeeze()
        t_emb = self.layer_t_1(t_emb)
        t_emb = torch.nn.functional.silu(t_emb)
        t_emb = self.layer_t_2(t_emb)
        t_emb = torch.nn.functional.silu(t_emb)

        x_emb = torch.nn.functional.relu(self.layer_x(x))

        skip = []
        for layer in self.residual_layers:
            x_emb, skip_connection = layer(x_emb, t_emb)
            skip.append(skip_connection)

        out = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        out = torch.nn.functional.relu(self.layer_out_1(out))
        out = self.dropout_out(out)
        out = self.layer_out_2(out)

        return out

    def _build_embedding(self, num_noise_steps: int, dim: int = 64) -> torch.Tensor:
        """Build an embedding for noise step.
        More details in section E.1 of Tashiro et al., 2021
        (https://arxiv.org/abs/2107.03502)

        Parameters
        ----------
        num_noise_steps : int
            Number of noise steps
        dim : int, optional
            output dimension, by default 64

        Returns
        -------
        torch.Tensor
            List of embeddings for noise steps
        """

        steps = torch.arange(num_noise_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class RNNEncDec(nn.Module):
    def __init__(self, dim_input, dim_model, num_layers, device, max_len=128):
        super().__init__()
        self.device = device
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                   nhead=2,
                                                   dropout=0,
                                                   batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model,
                                                   nhead=2,
                                                   dropout=0,
                                                   batch_first=True)

        self.transform_encoder = nn.TransformerEncoder(encoder_layer,
                                                       num_layers=num_layers).\
            to(device)
        self.emb = nn.Sequential(
            nn.Linear(dim_input, dim_model),
            PositionalEncoding(dim_model, max_len=max_len),
        ).to(device)
        self.transform_decoder = nn.TransformerDecoder(decoder_layer,
                                                       num_layers=num_layers).\
            to(device)
        self.last_layer = nn.Linear(dim_model, dim_input).to(device)

    def encoder(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(
            x.shape[1]).to(self.device)
        x = self.emb(x)
        return self.transform_encoder(x, mask=mask)

    def decoder(self, x, h):
        mask = nn.Transformer.generate_square_subsequent_mask(
            x.shape[1]).to(self.device)
        x = self.emb(x)
        x = self.transform_decoder(x, h, tgt_mask=mask)
        return self.last_layer(x)

    def forward(self, x):
        # x : (batch_size, window_size, dim_input)

        x_prev = x[:, :-1, :]

        # encoder
        h = self.encoder(x_prev)

        # decoder
        x_ = self.decoder(x_prev, h)
        return torch.cat((x[:, 0, :].unsqueeze(1), x_), dim=1)

    def predict(self, start, memory):
        # start/x_0 : (batch_size, dim_input)
        # memory/h : (batch_size, window_size, dim_model)
        start = start.unsqueeze(1)
        output = torch.zeros(start.shape[0],
                             memory.shape[1],
                             start.shape[2]).to(self.device)
        output[:, 0, :] = start.squeeze(1)
        for i in range(1, memory.shape[1]):
            tgt = output[:, :i, :]
            output[:, i, :] = self.decoder(tgt,
                                           memory)[:, -1, :]

        # output : (batch_size, window_size, dim_input)
        return output


# class conv_block(nn.Module):

#     def __init__(self, in_ch, hidden_ch, out_ch):

#         super(conv_block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1),  # no change in dimensions of  volume
#             nn.InstanceNorm1d(hidden_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(hidden_ch, out_ch, kernel_size=3, stride=1, padding=1),  # no change in dimensions of  volume
#             nn.InstanceNorm1d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class up_conv_block(nn.Module):

#     def __init__(self, in_ch, out_ch, scale_tuple):

#         super(up_conv_block, self).__init__()

#         self.up_conv = nn.Sequential(
#             nn.Upsample(scale_factor=scale_tuple, mode='trilinear'),
#             nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),  # no change in dimensions of  volume
#             nn.InstanceNorm1d(out_ch),
#             nn.ReLU(inplace=True),  # increasing the depth by adding one below
#             nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),  # no change in dimensions of  volume
#             nn.InstanceNorm1d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.up_conv(x)
#         return x


# class Unet_model(nn.Module):

#     def __init__(self, num_noise_steps, dim_model):
#         super(Unet_model, self).__init__()

#         filters_ = [8, 16, 32, 64, 128, 256]

#         self.Conv_1 = conv_block(in_ch_SA, filters_[0], filters_[1])
#         self.Conv_2 = conv_block(filters_[1], filters_[1], filters_[2])
#         self.Conv_3 = conv_block(filters_[2], filters_[2], filters_[3])
#         self.Conv_4 = conv_block(filters_[3], filters_[3], filters_[4])
#         self.Conv_5 = conv_block(filters_[4], filters_[4], filters_[5])

#         self.MaxPool_1 = nn.MaxPool1d(kernel_size=(2, 1, 2), stride=(2, 1, 2))
#         self.MaxPool_2 = nn.MaxPool1d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.MaxPool_3 = nn.MaxPool1d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.MaxPool_4 = nn.MaxPool1d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

#         self.up_sample_1 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
#         self.up_sample_2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
#         self.up_sample_3 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
#         self.up_sample_4 = nn.Upsample(scale_factor=(2, 1, 2), mode='trilinear')

#         self.up_conv_1 = conv_block(filters_[5]+filters_[4], filters_[4], filters_[4])
#         self.up_conv_2 = conv_block(filters_[4]+filters_[3], filters_[3], filters_[3])
#         self.up_conv_3 = conv_block(filters_[3]+filters_[2], filters_[2], filters_[2])
#         self.up_conv_4 = conv_block(filters_[2]+filters_[1], filters_[1], filters_[1])

#         self.Conv_final = nn.Conv1d(filters_[1], out_ch_SA, kernel_size=1, stride=1, padding=0)

#         self.layer_t_1 = torch.nn.Linear(dim_model, dim_model)
#         self.layer_t_2 = torch.nn.Linear(dim_model, dim_model)
#         self.embedding_noise_step = self._build_embedding(num_noise_steps, dim_model//2)

#     def forward(self, x, t):
#         # x : (batch_size, window_size, dim_input)
#         # t 
#         # Noise step embedding
#         t_emb = torch.as_tensor(self.embedding_noise_step)[t].squeeze()
#         t_emb = self.layer_t_1(t_emb)
#         t_emb = torch.nn.functional.silu(t_emb)
#         t_emb = self.layer_t_2(t_emb)
#         t_emb = torch.nn.functional.silu(t_emb)

#         x = x + t_emb


#         # network's encoder

#         e1 = self.Conv_1(x)
#         # print("E1:", e1.shape)
#         x = self.MaxPool_1(e1)
#         # print("E2_:", x.shape)
#         e2 = self.Conv_2(x)
#         # print("E3:", e2.shape)
#         x = self.MaxPool_2(e2)
#         # print("E4_:", x.shape)
#         e3 = self.Conv_3(x)
#         # print("E5:", e3.shape)
#         x = self.MaxPool_3(e3)
#         # print("E6_:", x.shape)
#         e4 = self.Conv_4(x)
#         # print("E7:", e4.shape)
#         x = self.MaxPool_4(e4)
#         # print("E8_:", x.shape)
#         x = self.Conv_5(x)

#         # network's decoder

#         x = self.up_sample_1(x)
#         # print("D1:", x.shape)
#         x = torch.cat([e4, x], dim=1)
#         del (e4)
#         # print("D2:", x.shape)
#         x = self.up_conv_1(x)
#         # print("D3:", x.shape)
#         x = self.up_sample_2(x)
#         # print("D4:", x.shape)
#         x = torch.cat([e3, x], dim=1)
#         del (e3)
#         # print("D5:", x.shape)
#         x = self.up_conv_2(x)
#         # print("D6:", x.shape)
#         x = self.up_sample_3(x)
#         # print("D7:", x.shape)
#         x = torch.cat([e2, x], dim=1)
#         del (e2)
#         # print("D8:", x.shape)
#         x = self.up_conv_3(x)
#         # print("D9:", x.shape)
#         x = self.up_sample_4(x)
#         # print("D10:", x.shape)
#         x = torch.cat([e1, x], dim=1)
#         del (e1)
#         # print("D11:", x.shape)
#         x = self.up_conv_4(x)
#         # print("D12:", x.shape)
#         x = self.Conv_final(x)
#         # print("D13:", d_SA.shape)

#         return x

#     def _build_embedding(self, num_noise_steps: int, dim: int = 64):
#         """Build an embedding for noise step.
#         More details in section E.1 of Tashiro et al., 2021
#         (https://arxiv.org/abs/2107.03502)

#         Parameters
#         ----------
#         num_noise_steps : int
#             Number of noise steps
#         dim : int, optional
#             output dimension, by default 64

#         Returns
#         -------
#         torch.Tensor
#             List of embeddings for noise steps
#         """

#         steps = torch.arange(num_noise_steps).unsqueeze(1)  # (T,1)
#         frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
#         table = steps * frequencies  # (T,dim)
#         table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
#         return table
