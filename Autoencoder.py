import torch
import torch.nn as nn
import os
torch.set_default_tensor_type(torch.FloatTensor)
#torch.set_default_tensor_type(torch.DoubleTensor)

nn_activations = {
    'relu':    nn.ReLU(),
    'silu':    nn.SiLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh':    nn.Tanh()
}

class ConvAutoencoder(nn.Module):
    def __init__(self, number_of_matters, conv_layer_kernal_size = (), conv_layer_stride = (), img_dim = 0, activation_type = 'relu'):
        super(ConvAutoencoder, self).__init__()
        if len(conv_layer_kernal_size) != len(conv_layer_stride):
            print("Error: conv_layer_kernal_size and conv_layer_stride should have the same sizes.")
            exit()
        self.encoder_layer_ = nn.ModuleList()   # dim_out = (dim_in - 1 - (kernal_size-1))//stride + 1
        self.decoder_layer_ = nn.ModuleList()   # dim_out = (dim_in - 1                  ) *stride + 1 + (kernal_size-1)
        if len(conv_layer_kernal_size) > 0:
            self.encoder_layer_size_ = [img_dim]
            self.number_of_layer_ = len(conv_layer_kernal_size)
            for index in range(self.number_of_layer_):
                self.encoder_layer_.append(nn.Conv3d(number_of_matters, number_of_matters, conv_layer_kernal_size[index], stride = conv_layer_stride[index]))
                self.encoder_layer_size_.append((self.encoder_layer_size_[-1] - 1 - (conv_layer_kernal_size[index]-1))//conv_layer_stride[index] + 1)
            self.decoder_layer_size_ = [self.encoder_layer_size_[-1]]
            for index in range(self.number_of_layer_):
                self.decoder_layer_.append(nn.ConvTranspose3d(number_of_matters, number_of_matters, conv_layer_kernal_size[index], stride = conv_layer_stride[index]))
                self.decoder_layer_size_.append((self.decoder_layer_size_[-1] - 1)*conv_layer_stride[index] + 1 + (conv_layer_kernal_size[index]-1))
            self.actv_ = nn_activations[activation_type]
        else:
            print("Error: no layer size information")
            exit()

    def print_info(self):
        print("%35s"%("Autoencoder dim: "), end = "")
        for index in range(self.number_of_layer_+1):  print("%d %d %d -> "%(self.encoder_layer_size_[index],self.encoder_layer_size_[index],self.encoder_layer_size_[index]), end = "")
        for index in range(1, self.number_of_layer_): print("%d %d %d -> "%(self.decoder_layer_size_[index],self.decoder_layer_size_[index],self.decoder_layer_size_[index]), end = "")
        print("%d %d %d"%(self.decoder_layer_size_[-1],self.decoder_layer_size_[-1],self.decoder_layer_size_[-1]))

    def encode(self, x):
        #x: (batch_size, number_of_matters, matter_dim, matter_dim, matter_dim)
        y = x
        for index in range(self.number_of_layer_):
            y = self.encoder_layer_[index](y)
            y = self.actv_(y)
        return y
    
    def decode(self, y):
        x = y
        for index in range(self.number_of_layer_ - 1):
            x = self.decoder_layer_[index](x)
            x = self.actv_(x)
        x = self.decoder_layer_[-1](x)
        return x
    
    def forward(self, x):
        return self.decode(self.encode(x))

class MLPAutoencoder(nn.Module):
    def __init__(self, input_size, encode_size, layer_size = (8, 8), activation_type = 'tanh'):
        super(MLPAutoencoder, self).__init__()

        self.encoder_layer_ = nn.ModuleList()
        self.decoder_layer_ = nn.ModuleList()
        if len(layer_size) > 0:
            self.encoder_layer_size_ = [input_size] +  list(layer_size)       + [encode_size]
            self.decoder_layer_size_ = [encode_size] + list(layer_size)[::-1] + [input_size]
            self.number_of_layer_ = len(self.encoder_layer_size_) - 1
            for index in range(self.number_of_layer_):
                self.encoder_layer_.append(nn.Linear(self.encoder_layer_size_[index], self.encoder_layer_size_[index + 1]))
            for index in range(self.number_of_layer_):
                self.decoder_layer_.append(nn.Linear(self.decoder_layer_size_[index], self.decoder_layer_size_[index + 1]))
            self.actv_ = nn_activations[activation_type]
        else:
            print("Error: no layer size information")
            exit()
    
    def print_info(self):
        print("%35s"%("Autoencoder dim: "), end = "")
        for index in range(self.number_of_layer_+1):      print("%d -> "%(self.encoder_layer_size_[index]), end = "")
        for index in range(1, self.number_of_layer_): print("%d -> "%(self.decoder_layer_size_[index]), end = "")
        print("%d"%(self.decoder_layer_size_[-1]))

    def encode(self, x):
        #x: (batch_size, input_size)
        y = x
        for index in range(self.number_of_layer_):
            y = self.encoder_layer_[index](y)
            y = self.actv_(y)
        return y
    
    def decode(self, y):
        x = y
        for index in range(self.number_of_layer_ - 1):
            x = self.decoder_layer_[index](x)
            x = self.actv_(x)
        x = self.decoder_layer_[-1](x)
        return x
    
    def forward(self, x):
        return self.decode(self.encode(x))

class DummyAutoencoder:
    def __init__(self):   pass
    def encode(self, x):  return x
    def decode(self, x):  return x
    def parameters(self): return list()
    def to(self, device): pass
