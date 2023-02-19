import torch
import torch.nn as nn
from EncoderModule import ConvAutoencoder, MLPAutoencoder, DummyAutoencoder
# torch.set_default_tensor_type(torch.FloatTensor)
# torch.set_default_tensor_type(torch.DoubleTensor)

class LatticeMD(nn.Module):
    #This is a LSTM model which predicts: rho(t), S(t), and V(t).
    #
    # rho(t) = 3-D image of matters of interest.
    # S(t)   = stress tensor.
    # V(t)   = potential energy.
    #
    #
    #
    # number_of_matters:             int
    # matter_dim:                    int
    # matter_conv_ae_kernal_size: (int, ...)
    # matter_conv_ae_stride:      (int, ...)
    # matter_conv_ae_stride:      (int, ...)
    # matter_linear_ae_hidden_size:  (int, ...)
    # matter_linear_ae_encoded_size: int
    #
    # number_of_nonmatter_features:     int
    # nonmatter_encoder_hidden_size:  (int, ...)
    #
    def __init__(self, number_of_matters, matter_dim, \
                 matter_encoder_conv_ae_kernal_size, matter_encoder_conv_ae_stride, matter_encoder_activation, \
                 matter_decoder_conv_ae_kernal_size, matter_decoder_conv_ae_stride, matter_decoder_activation, \
                 nonmatter_encoder_hidden_size, nonmatter_encoder_activation, nonmatter_encoded_size, 
                 nonmatter_decoder_hidden_size, nonmatter_decoder_activation, \
                 number_of_nonmatter_features, number_of_lstm_layer = 1):
        super(LatticeMD, self).__init__()
        if len(matter_encoder_conv_ae_kernal_size) != len(matter_encoder_conv_ae_stride):
            print("Error: matter_encoder_conv_ae_kernal_size and matter_encoder_conv_ae_stride should have the same sizes.")
            exit()
        if len(matter_decoder_conv_ae_kernal_size) != len(matter_decoder_conv_ae_stride):
            print("Error: matter_decoder_conv_ae_kernal_size and matter_decoder_conv_ae_stride should have the same sizes.")
            exit()
        self.number_of_matters_ = number_of_matters
        self.matter_dim_ = matter_dim
        self.matter_dim3_ = matter_dim**3
        self.number_of_nonmatter_features_ = number_of_nonmatter_features # default: stress tensor and pe
        self.number_of_lstm_layer_ = number_of_lstm_layer
        self.param  = list()

        #Create neighbor roll-shift and roll-dim list
        system_dim_offset = 2
        self.neighbor_roll_dim_list = (0+system_dim_offset, 1+system_dim_offset, 2+system_dim_offset)
        self.neighbor_roll_shift_list = []
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    self.neighbor_roll_shift_list.append((-(x-1), -(y-1), -(z-1)))
        self.neighbor_roll_shift_list = tuple(self.neighbor_roll_shift_list)
        self.number_of_neighbor_block_ = len(self.neighbor_roll_shift_list)

        #Matter Encoder-Autoencoder
        self.matter_encoder_encoded_dim_ = 3*matter_dim
        self.matter_encoder_ae_ = ConvAutoencoder(number_of_matters, matter_encoder_conv_ae_kernal_size, matter_encoder_conv_ae_stride, 3*matter_dim)
        for index in range(len(matter_encoder_conv_ae_kernal_size)):
            self.matter_encoder_encoded_dim_ = (self.matter_encoder_encoded_dim_ - 1 - (matter_encoder_conv_ae_kernal_size[index]-1))//matter_encoder_conv_ae_stride[index] + 1
        self.matter_encoder_encoded_flatten_dim_ = self.number_of_matters_ * self.matter_encoder_encoded_dim_**3
        self.param += list(self.matter_encoder_ae_.parameters())

        #Matter Decoder-Autoencoder
        self.matter_decoder_encoded_dim_ = matter_dim
        self.matter_decoder_ae_ = ConvAutoencoder(number_of_matters, matter_decoder_conv_ae_kernal_size, matter_decoder_conv_ae_stride, matter_dim)
        for index in range(len(matter_decoder_conv_ae_kernal_size)):
            self.matter_decoder_encoded_dim_ = (self.matter_decoder_encoded_dim_ - 1 - (matter_decoder_conv_ae_kernal_size[index]-1))//matter_decoder_conv_ae_stride[index] + 1
        self.matter_decoder_encoded_flatten_dim_ = self.number_of_matters_ * self.matter_decoder_encoded_dim_**3
        self.param += list(self.matter_decoder_ae_.parameters())

        if self.number_of_nonmatter_features_ > 0:
            #Non-matter Encoder
            self.nonmatter_encoder_      = MLPLayer(self.number_of_neighbor_block_ * self.number_of_nonmatter_features_, nonmatter_encoded_size, nonmatter_encoder_hidden_size, )
            self.nonmatter_encoded_dim_  = nonmatter_encoded_size
            #Non-matter Decoder
            self.nonmatter_decoder_      = MLPLayer(self.nonmatter_encoded_dim_, self.number_of_nonmatter_features_, nonmatter_decoder_hidden_size)
            self.param += list(self.nonmatter_encoder_.parameters())
            self.param += list(self.nonmatter_decoder_.parameters())
        else:
            self.nonmatter_encoded_dim_ = 0

        self.lstm_out_dim_ = self.matter_decoder_encoded_flatten_dim_ + self.nonmatter_encoded_dim_
        self.lstm_in_dim_  = self.matter_encoder_encoded_flatten_dim_ + self.nonmatter_encoded_dim_
        self.lstm_ = nn.LSTM(self.lstm_in_dim_, self.lstm_out_dim_, self.number_of_lstm_layer_)
        self.param += list(self.lstm_.parameters())

        self.print_info()

    def to(self, device):
        self.matter_encoder_ae_.to(device)
        self.matter_decoder_ae_.to(device)
        self.lstm_.to(device)
        if self.nonmatter_encoded_dim_ > 0:
            self.nonmatter_encoder_.to(device)
            self.nonmatter_decoder_.to(device)

    def freeze_model(self):
        for a in self.param:
            a.requires_grad = False
    
    def unfreeze_model(self):
        for a in self.param:
            a.requires_grad = True

    def parameters(self):
        return self.param
    
    def print_info(self):
        print("%35s%d"%("Number of neighbor block: ", self.number_of_neighbor_block_))
        print("")
        print("%35s%d"%("Number of matter types: ", self.number_of_matters_))
        print("%35s%d, %d, %d, %d"%("matter dim: ", self.number_of_matters_, self.matter_dim_, self.matter_dim_, self.matter_dim_))
        print("%35s%d"%("matter encoded flat dim: ", self.matter_encoder_encoded_flatten_dim_))
        self.matter_encoder_ae_.print_info()
        self.matter_decoder_ae_.print_info()
        print("")
        if self.number_of_nonmatter_features_ > 0:
            print("%35s%d"%("Number of non-matter features: ", self.number_of_nonmatter_features_))
            print("%35s%d"%("Non-matter encoded dim: ", self.nonmatter_encoded_dim_))
            self.nonmatter_encoder_.print_info()
            print("%35s%d"%("Non-matter decoded dim: ", self.nonmatter_encoded_dim_))
            self.nonmatter_decoder_.print_info()
            print("")
        print("%35s%3d"%("Number of LSTM layer: ", self.number_of_lstm_layer_))
        print("%35s%3d <- (%d*%d^3+%d)*%d"%("LSTM input dim: ", self.lstm_in_dim_, self.number_of_matters_, self.matter_encoder_encoded_dim_, self.nonmatter_encoded_dim_, self.number_of_neighbor_block_))
        print("%35s%3d <- (%d*%d^3+%d)"%("LSTM output dim: ", self.lstm_out_dim_, self.number_of_matters_, self.matter_encoder_encoded_dim_, self.nonmatter_encoded_dim_))
        print("")

    def forward(self, matter_sequence, nonmatter_sequence, system_dim):
        #input
        #   matter_sequence:        (sequence_length-1, batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        #   nonmatter_sequence:     (sequence_length-1, batch_size * system_dim3, number_of_nonmatter_features)
        #   system_dim:             (batch_size, 3)
        #output
        #   predicted_matter:       (batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        #   predicted_nonmatter:    (batch_size * system_dim3, number_of_nonmatter_features)
        matter_sequence_neighbors    = self.get_neighbor_cell_matter(matter_sequence.diff(dim = 0), system_dim)
        if self.number_of_nonmatter_features_ > 0:
            nonmatter_sequence_neighbors = self.get_neighbor_cell_nonmatter(nonmatter_sequence.diff(dim = 0), system_dim)
        else:
            nonmatter_sequence_neighbors = None
        print(matter_sequence_neighbors.shape, nonmatter_sequence_neighbors)
        exit()
        lstm_in = self.encode(matter_sequence_neighbors, nonmatter_sequence_neighbors)
        out, (hn, cn) = self.advance(lstm_in)
        lstm_out = hn.sum(dim = 0) #(batch_size * system_dim3, lstm_out_dim)
        predicted_matter, predicted_nonmatter = self.decode(lstm_out)
        predicted_matter = self.matter_normalize(predicted_matter, system_dim)
        return predicted_matter, predicted_nonmatter
    
    def get_neighbor_cell_matter(self, matter_sequence, system_dim):
        #input
        #   matter_sequence:                  (sequence_length-2, batch_size * system_dim3, number_of_matters,   matter_dim,   matter_dim,   matter_dim)
        #output
        #   matter_sequence_neighbors:        (sequence_length-2, batch_size * system_dim3, number_of_matters, 3*matter_dim, 3*matter_dim, 3*matter_dim)
        matter_sequence_neighbors = matter_sequence.view(len(matter_sequence), -1, *system_dim, self.number_of_matters_, self.matter_dim_, self.matter_dim_, self.matter_dim_)
        matter_sequence_neighbors = self.get_neighbor_cell(matter_sequence_neighbors, system_dim)
        matter_sequence_neighbors = matter_sequence_neighbors.view(3, 3, 3, len(matter_sequence), -1, *system_dim, self.number_of_matters_, self.matter_dim_, self.matter_dim_, self.matter_dim_).permute(3, 4, 5, 6, 7, 8, 0, 9, 1, 10, 2, 11)
        matter_sequence_neighbors = matter_sequence_neighbors.flatten(10, 11).flatten(8, 9).flatten(6, 7).flatten(1, 4)
        return matter_sequence_neighbors

    def get_neighbor_cell_nonmatter(self, nonmatter_sequence, system_dim):
        #input
        #   nonmatter_sequence:               (sequence_length-2, batch_size * system_dim3,    number_of_nonmatter_features)
        #output
        #   nonmatter_sequence_neighbors:     (sequence_length-1, batch_size * system_dim3, 27*number_of_nonmatter_features)
        nonmatter_sequence_neighbors = nonmatter_sequence.view(len(nonmatter_sequence), -1, *system_dim, self.number_of_nonmatter_features)
        nonmatter_sequence_neighbors = self.get_neighbor_cell(nonmatter_sequence_neighbors, system_dim)
        nonmatter_sequence_neighbors = nonmatter_sequence_neighbors.view(3, 3, 3, len(matter_sequence), -1, *system_dim, self.number_of_nonmatter_features).permute(3, 4, 5, 6, 7, 0, 1, 2, 8)
        nonmatter_sequence_neighbors = nonmatter_sequence_neighbors.flatten(5, 8).flatten(1, 4)
        return nonmatter_sequence_neighbors

    def get_neighbor_cell(self, sequence, system_dim):
        #input
        #   sequence:                  (sequence_length-2, batch_size, system_dim3, *)
        #output
        #   sequence_neighbors:        (27, sequence_length-2, batch_size, system_dim3, *)
        sequence_neighbors = torch.stack((sequence.roll(shifts=self.neighbor_roll_shift_list[ 0], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[ 1], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[ 2], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[ 3], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[ 4], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[ 5], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[ 6], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[ 7], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[ 8], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[ 9], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[10], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[11], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[12], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[13], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[14], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[15], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[16], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[17], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[18], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[19], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[20], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[21], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[22], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[23], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[24], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[25], dims=self.neighbor_roll_dim_list),\
                                          sequence.roll(shifts=self.neighbor_roll_shift_list[26], dims=self.neighbor_roll_dim_list)))
        return sequence_neighbors
    
    # def get_neighbor_block_data(self, encoded_in, system_dim):
    #     #encoded_in:        (sequence_length-1, batch_size * system_dim3, lstm_out_dim)
    #     lstm_in = encoded_in.view(len(encoded_in), -1, *system_dim, self.lstm_out_dim_) #(sequence_length-1, batch_size, *system_dim, lstm_out_dim)
    #     lstm_in = torch.stack((lstm_in, \
    #                            lstm_in.roll(shifts= 1, dims=2), \
    #                            lstm_in.roll(shifts=-1, dims=2), \
    #                            lstm_in.roll(shifts= 1, dims=3), \
    #                            lstm_in.roll(shifts=-1, dims=3), \
    #                            lstm_in.roll(shifts= 1, dims=4), \
    #                            lstm_in.roll(shifts=-1, dims=4))).permute((1, 2, 3, 4, 5, 0, 6))  #(sequence_length-1, batch_size, *system_dim, number_of_neighbor_block, lstm_out_dim)
    #     lstm_in = lstm_in.flatten(start_dim = -2)             #(sequence_length-1, batch_size, *system_dim, lstm_in_dim)
    #     lstm_in = lstm_in.flatten(start_dim = 1, end_dim = 4) #(sequence_length-1, batch_size * system_dim3, lstm_in_dim)
    #     return lstm_in

    def advance(self, lstm_in):
        #lstm_in:                   (sequence_length-1, batch_size * system_dim3, lstm_in_dim)
        lstm_out, hidden = self.lstm_(lstm_in)
        return lstm_out, hidden

    def encode(self, matter_sequence, nonmatter_sequence):
        #matter_sequence:           (sequence_length-2, batch_size * system_dim3, number_of_matters, 3*matter_dim, 3*matter_dim, 3*matter_dim)
        #nonmatter_sequence:        (sequence_length-2, batch_size * system_dim3, 27*number_of_nonmatter_features)
        L = len(matter_sequence)
        matter_encoded    = matter_sequence.flatten(0, 1)                                          #(sequence_length-2 * batch_size * system_dim3, number_of_matters, 3*matter_dim, 3*matter_dim, 3*matter_dim)
        matter_encoded    = self.matter_encoder_ae_.encode(matter_encoded).flatten(start_dim = 1)  #(sequence_length-2 * batch_size * system_dim3, matter_encoder_encoded_flatten_dim)
        matter_encoded    = matter_encoded.view(L, -1, self.matter_encoder_encoded_flatten_dim_)   #(sequence_length-2,  batch_size * system_dim3, matter_encoder_encoded_flatten_dim)

        if self.number_of_nonmatter_features_ > 0:
            nonmatter_encoded = nonmatter_sequence.flatten(start_dim = 0, end_dim = 1)                 #(sequence_length-2, * batch_size * system_dim3, 27*number_of_nonmatter_features_)
            nonmatter_encoded = self.nonmatter_encoder_.encode(nonmatter_encoded)                           #(sequence_length-2, * batch_size * system_dim3, nonmatter_encoded_dim)
            nonmatter_encoded = nonmatter_encoded.view(L, -1, self.nonmatter_encoded_dim_)             #(sequence_length-2,   batch_size * system_dim3, nonmatter_encoded_dim)
            lstm_in = torch.cat((matter_encoded, nonmatter_encoded), dim = -1)                      #(sequence_length-2,  batch_size * system_dim3, lstm_in_dim)
        else:
            lstm_in = matter_encoded

        return lstm_in
    
    def decode(self, lstm_out):
        #lstm_out:                  (batch_size * system_dim3, lstm_out_dim)
        matter_decoded = lstm_out[:, :self.matter_decoder_encoded_flatten_dim_].view(-1, self.number_of_matters_, self.matter_decoder_encoded_dim_, self.matter_decoder_encoded_dim_, self.matter_decoder_encoded_dim_) #(batch_size * system_dim3, number_of_matters, matter_decoder_encoded_dim, matter_decoder_encoded_dim, matter_decoder_encoded_dim)
        matter_decoded = self.matter_decoder_ae_.decode(matter_decoded)                                                                                                                                                 #(batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        
        if self.number_of_nonmatter_features_ > 0:
            nonmatter_decoded = lstm_out[:, self.matter_decoder_encoded_flatten_dim_:]                                                                                                                                      #(batch_size * system_dim3, nonmatter_encoded_dim)
            nonmatter_decoded = self.nonmatter_decoder_.decode(nonmatter_decoded)                                                                                                                                                #(batch_size * system_dim3, number_of_nonmatter_features)
        
        return matter_decoded, nonmatter_decoded
    
    def matter_normalize(self, matter, system_dim):
        #matter:                    (batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        matter = matter.view(-1, system_dim[0]*system_dim[1]*system_dim[2], self.number_of_matters_, self.matter_dim_, self.matter_dim_, self.matter_dim_)
        matter_mean = matter.mean(dim = (1, 3, 4, 5), keepdim = True) #(batch_size, 1, number_of_matters, 1, 1, 1)
        matter -= matter_mean
        matter = matter.flatten(start_dim = 0, end_dim = 1)
        return matter


