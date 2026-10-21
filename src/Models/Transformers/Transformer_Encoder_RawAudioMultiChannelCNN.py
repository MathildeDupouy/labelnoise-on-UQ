#!/usr/bin/env python3

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchinfo import summary

################################################################################
################################################################################

class PositionalEncoding(nn.Module):
    """
        Modified version of the code in https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        to put the batch first
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, pos_encoder_type='Original'):
        super().__init__()
        self.pos_encoder_type = pos_encoder_type
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if (d_model % 2 == 1):
            pe[0, :, 1::2] = torch.cos(position * div_term[:-1]) # We do not take the
            # last component because d_model is odd so (last_odd_index - 1) = second_to_last_even_index
            # For instance, if d_model = 125, we will have a list of indices
            # from 0 to 124, so the last even index is 123 and 123-1=122 and
            # 122 is the second to last even number
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # The following lines can be used if the trained model HAS NOT the self.pos_encoder_type
        # attribute BUT it is NOT COMPATIBLE with JIT.
        try:
            if (self.pos_encoder_type.lower() == 'mlhc'):
                x = x + self.pe[:, x.size(1), :]
            elif (self.pos_encoder_type.lower() == 'original'):
                x = x + self.pe[:, 0:x.size(1), :]
        except:
            # Using the one from MLHC
            print("\n\nWARNING: USING MLHC POSITIONAL ENCODING BECAUSE NO POSITIONAL ENCODING WAS FOUND\n\n")
            x = x + self.pe[:, x.size(1), :]

        # # The following lines can be used if the trained model HAS the self.pos_encoder_type
        # # attribute AND it is COMPATIBLE with JIT.
        # if (self.pos_encoder_type.lower() == 'mlhc'):
        #     x = x + self.pe[:, x.size(1), :]
        # elif (self.pos_encoder_type.lower() == 'original'):
        #     x = x + self.pe[:, 0:x.size(1), :]

        return self.dropout(x)




class TransformerEncoderMultichannelCNN(nn.Module):

    def __init__(
                    self,
                    in_channels: int,
                    nhead: int,
                    d_hid: int,
                    nlayers: int,
                    dropout: float = 0.5,
                    d_model: int = 64,
                    classification_pool: str = 'ClassToken',
                    n_conv_layers: int = 2,
                    pos_encoder_type: str = 'Original'
                ):
        super().__init__()
        # Parameters of the model
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.classification_pool = classification_pool
        self.in_channels = in_channels
        self.pos_encoder_type = pos_encoder_type

        # Layers
        # Input embedding layers
        self.n_conv_layers = n_conv_layers
        self.conv1 = torch.nn.Conv1d(in_channels=self.in_channels, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(in_channels=128, out_channels=self.d_model, kernel_size=3, stride=1, padding=1)
        self.conv = torch.nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=3, stride=1, padding=0)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2)

        # Class token parameter (if used)
        if (self.classification_pool.lower() == 'classtoken'):
            self.class_token = nn.Parameter(torch.randn(self.d_model)) # Normal random tensor

        # Positional Encoder Layer
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, pos_encoder_type=self.pos_encoder_type)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
        """
        # Extraction of the embeddings
            # print("src", src.shape, src)
            # print("Has NaNs in src:", torch.isnan(src).any())
            # print("Has Infs in src:", torch.isinf(src).any())
            # print("src min:", src.min().item())
            # print("src max:", src.max().item())
        x = F.leaky_relu(self.conv1(src))
        x = F.leaky_relu(self.conv2(x))
        for _ in range(self.n_conv_layers):
          x = F.leaky_relu(self.conv(x))
          x = self.maxpool(x)
        x = torch.swapaxes(x, 1, 2) # Because the input of the PE and the Transformer
        # has to be under the format (batch_size, seq_len, feat_dim)

        # Class token if used
        if (self.classification_pool.lower() == 'classtoken'):
            batch_size = x.shape[0]
            class_tokens = self.class_token.expand(batch_size, -1)
            class_tokens = class_tokens.unsqueeze(dim=1)
            x = torch.cat([class_tokens, x], dim=1)


        # Positional encoding
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Transformer encoding
        encoding = self.transformer_encoder(x)

        # Embedded representation
        if (self.classification_pool.lower() == 'classtoken'):
            # class_hidden_features = encoding.mean(dim=1)
            class_hidden_features = encoding[:, 0]
        else:
            class_hidden_features = encoding

        return class_hidden_features


class TransformerClassifierMultichannelCNN(nn.Module):

    def __init__(
                    self,
                    in_channels: int,
                    nhead: int,
                    d_hid: int,
                    nlayers: int,
                    dropout: float = 0.5,
                    nb_features_projection: float = 50,
                    d_model: int = 64,
                    num_classes: int = 3,
                    classification_pool: str = 'ClassToken',
                    n_conv_layers: int = 2,
                    pos_encoder_type: str = 'Original'
                ):
        super().__init__()
        # Parameters of the model
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.classification_pool = classification_pool
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pos_encoder_type = pos_encoder_type

        # Encoder model
        self.encoder = TransformerEncoderMultichannelCNN(
                                                            in_channels=in_channels,
                                                            nhead=nhead,
                                                            d_hid=d_hid,
                                                            nlayers=nlayers,
                                                            dropout=dropout,
                                                            d_model=d_model,
                                                            classification_pool=classification_pool,
                                                            n_conv_layers=n_conv_layers,
                                                            pos_encoder_type=pos_encoder_type
                                                        )
        self.d_model = self.encoder.d_model

        # Latent representation
        self.latent_representation = None
        
        # Classification layers
        self.layer_norm_1 = nn.LayerNorm(self.d_model)
        self.out_1 = nn.Linear(self.d_model, nb_features_projection)
        self.layer_norm_2 = nn.LayerNorm(nb_features_projection)
        self.out_2 = nn.Linear(nb_features_projection, self.num_classes)
        

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
        """
        # Encoding using the Transformer model
        class_hidden_features = self.encoder(src)
        self.latent_representation = class_hidden_features

        # Classification layers
        # output = self.out_1(class_hidden_features)
        output = self.out_1(self.layer_norm_1(class_hidden_features))
        # output = self.out_2(output)
        output = self.out_2(self.layer_norm_2(output))

        if (self.classification_pool.lower() == 'avg'):
            output = output.mean(dim=1)
            
        if not self.training:  # If in evaluation mode, apply Softmax
            output = F.softmax(output, dim=1)

        return (output,)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


################################################################################
################################################################################
"""
    MAIN CLASS
"""
if __name__=='__main__':
    # Device to use for computation
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Creating the model
    in_channels = 1
    # in_channels = 2
    
    # PTB config
    nhead = 8 # corresponds to the number of layers in the Encoder/Decoder
    d_hid = 64
    nlayers = 8
    dropout = 0.2
    d_model = 128
    n_conv_layers = 4
    nb_features_projection = 10
    num_classes = 2
    
    # ESR config
    # nhead = 4 # corresponds to the number of layers in the Encoder/Decoder
    # d_hid = 8
    # nlayers = 4
    # dropout = 0.2
    # d_model = 64
    # n_conv_layers = 2
    # nb_features_projection = 4
    # num_classes = 5
    
    # classification_pool = 'Avg'
    # classification_pool = 'ClassToken'
    # pos_encoder_type = 'MLHC'
    
    model = TransformerClassifierMultichannelCNN(
                                                    in_channels,
                                                    nhead=nhead,
                                                    d_hid=d_hid,
                                                    nlayers=nlayers,
                                                    dropout=dropout,
                                                    nb_features_projection=nb_features_projection,
                                                    d_model=d_model,
                                                    num_classes=num_classes,
                                                    # classification_pool,
                                                    n_conv_layers=n_conv_layers,
                                                    # pos_encoder_type
                                                )
    model.to(device)

    # # Creating dummy data
    # audio_length = 1400
    audio_length = 187 # PTB
    # audio_length = 178 # ESR
    bs = 4
    dummy_data = torch.randn((bs, in_channels, audio_length)).to(device)
    print("Shape of the input data: {}".format(dummy_data.shape))

    # Testing the model with dummy data
    output = model(dummy_data)

    latent_representation = model.latent_representation
    print("Output shape: ", output[0].shape)
    print("Latent representation shape: ", latent_representation.shape)

    if (output[0].shape != dummy_data.shape):
        print("WARNING: the input and the output do not have the same shape !!!")
        print("The input shape is {} and the output shape is {}".format(dummy_data.shape, output[0].shape))

    # Summary of the model
    summary(model=model, input_size=(bs, in_channels, audio_length))

    # # Parameters of the network
    # for name, param in model.named_parameters():
    #     print(name)
    
    # Number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters in the model: ", total_params)

    # # Saving the model
    # model_torchscript = torch.jit.script(model) # Export to TorchScript
    # model_torchscript.save('../../../tmp/model_raw.pt') # Save
