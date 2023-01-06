import torch
import torch.nn as nn
import os
from Autoencoder import ConvAutoencoder, MLPAutoencoder, DummyAutoencoder
torch.set_default_tensor_type(torch.FloatTensor)
#torch.set_default_tensor_type(torch.DoubleTensor)

class DimensionalityException(Exception): pass
class TimestepException(Exception): pass
class NumberOfElementsException(Exception): pass

class MDSequenceData:
    def __init__(self, input_dir, system_dim, sequence_length, max_frame = -1, sequence_start_indices = [0], output_dir='./', select_matter = (), log_filename = 'log.lammps', moving_average_length = 1):
        self.output_dir_ = output_dir
        self.system_dim_ = system_dim
        self.system_dim3_ = system_dim[0]*system_dim[1]*system_dim[2]
        self.sequence_length_ = sequence_length
        if os.path.isdir(self.output_dir_) == False: os.system("mkdir -p %s"%(self.output_dir_))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # read pe: potential energy of each chunk. [unit: kcal/mole]
            print("Reading %s/pe.txt..."%(input_dir))
            keys, step, data = self.read_lammps_ave("%s/pe.txt"%(input_dir), max_frame = max_frame)
            self.data_step_ = step
            if len(keys) != 1: raise NumberOfElementsException
            if self.system_dim3_ != data.shape[1]: raise DimensionalityException
            pe_data = data
            
            # read stress, virial stress tensor of each chunk. [unit: atm]
            print("Reading %s/stress.txt..."%(input_dir))
            keys, step, data = self.read_lammps_ave("%s/stress.txt"%(input_dir), max_frame = max_frame)
            if len(self.data_step_) != len(step) or ((self.data_step_-step)**2).sum() > 0.0: raise TimestepException
            if len(keys) != 6:  raise NumberOfElementsException
            if self.system_dim3_ != data.shape[1]: raise DimensionalityException
            stress_data = data

            # read matter, amount of the matter of each chunk. [unit: #]
            print("Reading %s/matter.txt..."%(input_dir))
            keys, step, data = self.read_lammps_ave("%s/matter.txt"%(input_dir), max_frame = max_frame)
            if len(self.data_step_) != len(step) or ((self.data_step_-step)**2).sum() > 0.0: raise TimestepException
            if len(select_matter) == 0:
                self.number_of_matters_ = len(keys)
                data = data
            else:
                self.number_of_matters_ = len(select_matter)
                data = data[:, :, select_matter]
            self.matter_dim_ = int(round((data.shape[1]/self.system_dim3_)**(1/3), 0))
            if self.matter_dim_**3 * self.system_dim3_ != data.shape[1]: raise DimensionalityException
            matter_data = data

            # read matter sum
            print("Reading %s/matter_sum_list.txt..."%(input_dir))
            with open("%s/matter_sum_list.txt"%(input_dir), 'r') as fin:
                matter_sum_data = torch.tensor([float(i) for i in fin.readline().strip().split()])

        except DimensionalityException:
            print("Error: Dimensionality ")
            exit()
        except NumberOfElementsException:
            print("Error: Number of elements in the dataset is wrong.")
            exit()
        except TimestepException:
            print("Error: Time steps should be the same.")
            exit()

        matter_data     = matter_data.to(device)
        stress_data     = stress_data.to(device)
        pe_data         = pe_data.to(device)
        matter_sum_data = matter_sum_data.to(device)
        stress_sum_data = stress_data.sum(dim = 1)
        pe_sum_data     = pe_data.sum(dim = 1)

        matter_data_normalization_factor = matter_sum_data/matter_data.sum(dim = 1)
        matter_data *= matter_data_normalization_factor.unsqueeze(1)

        #convert S*V/V to S*V/v (deprecated)
        #stress_data *= self.system_dim3_
        #stress_sum_data *= self.system_dim3_

        #compute prefactor
        self.matter_prefactor_     = 1.0/self.compute_moving_average_std(matter_data, moving_average_length, dim = (0, 1))
        self.stress_prefactor_     = 1.0/self.compute_moving_average_std(stress_data, moving_average_length, dim = (0, 1))
        self.pe_prefactor_         = 1.0/self.compute_moving_average_std(pe_data, moving_average_length, dim = (0, 1))
        self.matter_sum_prefactor_ = 1.0/matter_sum_data
        self.stress_sum_prefactor_ = 1.0/self.compute_moving_average_std(stress_sum_data, moving_average_length, dim = 0)
        self.pe_sum_prefactor_     = 1.0/self.compute_moving_average_std(pe_sum_data, moving_average_length, dim = 0)

        #split data into sequences
        self.matter_sequence_data_ = self.split_data_sequence(matter_data,     self.sequence_length_+1, sequence_start_indices, moving_average_length, verbose = True)
        self.number_of_batches_    = len(self.matter_sequence_data_)
        ss = self.sequence_length_ + 1
        self.matter_sequence_data_ = self.matter_sequence_data_.view(self.number_of_batches_, ss, self.system_dim_[0], self.matter_dim_, self.system_dim_[1], self.matter_dim_, self.system_dim_[2], self.matter_dim_, self.number_of_matters_).permute(0, 1, 2, 4, 6, 8, 3, 5, 7)
        self.matter_sequence_data_ = self.matter_sequence_data_.flatten(start_dim = 2, end_dim = 4)
        self.stress_sequence_data_ = self.split_data_sequence(stress_data,     self.sequence_length_+1, sequence_start_indices, moving_average_length)
        self.pe_sequence_data_     = self.split_data_sequence(pe_data,         self.sequence_length_+1, sequence_start_indices, moving_average_length)
        self.stress_sum_data_      = self.split_data_sequence(stress_sum_data, self.sequence_length_+1, sequence_start_indices, moving_average_length)
        self.pe_sum_data_          = self.split_data_sequence(pe_sum_data,     self.sequence_length_+1, sequence_start_indices, moving_average_length)
        self.matter_sum_data_      = torch.tile(matter_sum_data, (self.number_of_batches_, 1))

        self.matter_prefactor_     = self.matter_prefactor_.cpu()
        self.stress_prefactor_     = self.stress_prefactor_.cpu()
        self.pe_prefactor_         = self.pe_prefactor_.cpu()
        self.matter_sum_prefactor_ = self.matter_sum_prefactor_.cpu()
        self.stress_sum_prefactor_ = self.stress_sum_prefactor_.cpu()
        self.pe_sum_prefactor_     = self.pe_sum_prefactor_.cpu()
        
        self.matter_sequence_data_ = self.matter_sequence_data_.cpu()
        self.stress_sequence_data_ = self.stress_sequence_data_.cpu()
        self.pe_sequence_data_ = self.pe_sequence_data_.cpu()
        self.matter_sum_data_ = self.matter_sum_data_.cpu()
        self.stress_sum_data_ = self.stress_sum_data_.cpu()
        self.pe_sum_data_ = self.pe_sum_data_.cpu()

        self.print_info()

    def print_info(self):
        print("")
        print("%30s%d"%("Number of batches: ",len(self.matter_sequence_data_)))
        print("%30s%d"%("Sequence length: ",self.sequence_length_))
        print("%30s%d %d %d"%("System dim: ", self.system_dim_[0], self.system_dim_[1], self.system_dim_[2]))
        print("%30s%d %d^3"%("Matter Image dim: ", self.number_of_matters_, self.matter_dim_))

    def get_data(self):
        return self.number_of_matters_, self.matter_dim_, self.system_dim_, self.sequence_length_, \
               self.matter_sequence_data_, self.stress_sequence_data_, self.pe_sequence_data_, \
               self.matter_sum_data_ ,     self.stress_sum_data_,      self.pe_sum_data_

    def get_prefactor(self):
        return self.matter_prefactor_,     self.stress_prefactor_,     self.pe_prefactor_,\
               self.matter_sum_prefactor_, self.stress_sum_prefactor_, self.pe_sum_prefactor_

    def compute_moving_average_std(self, x, moving_average_length, dim = 0):
        if moving_average_length == 1: return x.std(dim = dim, unbiased = False)
        x_ma_tmp = self.moving_average(x, moving_average_length)[moving_average_length-1::moving_average_length]
        x_tmp = x[moving_average_length:][moving_average_length-1::moving_average_length]
        x_ma_std = (x_ma_tmp - x_tmp).std(dim = dim, unbiased = False)
        return x_ma_std

    def moving_average(self, x, moving_average_length, dim = 0):
        if moving_average_length == 1: return x
        return (x.cumsum(dim = dim)[moving_average_length:] - x.cumsum(dim = dim)[:-moving_average_length])/moving_average_length
    
    def split_data_sequence(self, data, sequence_length, sequence_start_indices, moving_average_length, verbose = False):
        split_data = None
        for sequence_start_index in sequence_start_indices:
            data_ma = self.moving_average(data[sequence_start_index:], moving_average_length)[moving_average_length-1::moving_average_length]
            if sequence_length > 1:
                data_tmp = data_ma.split(sequence_length)
                if len(data_tmp[-1]) != sequence_length: data_tmp = data_tmp[:-1]
                data_tmp = torch.stack(data_tmp)
            elif sequence_length == 1:
                data_tmp = data_ma.unsqueeze(1)

            if split_data is not None: split_data = torch.cat((split_data, data_tmp))
            else: split_data = data_tmp

            if verbose:
                print("splitting data:")
                print("\tmoving average length = %d"%(moving_average_length))
                print("\tdata range = %d ~ %d"%(sequence_start_index, len(data)))
                print("\tbatch subtotal = %d"%(len(split_data)))
        return split_data

    def read_lammps_ave(self, filename, max_frame = -1):
        #keys: [total_number_of_matters]
        #step: (total_sequence_length)
        #data: (total_sequence_length, system_dim3*matter_dim3, total_number_of_matters) for matter
        #      (total_sequence_length, system_dim3,             6)                       for stress tensor
        #      (total_sequence_length, system_dim3,             1)                       for potential energy
        with open(filename, 'r') as fin:
            fin.readline()
            fin.readline()
            linelist = fin.readline().strip().split()
            keys = linelist[2:]
            data = []
            step = []
            for aline in fin:
                linelist = aline.strip().split()
                nline = int(linelist[1])
                data.append([])
                step.append(float(linelist[0]))
                for i in range(nline):
                    data[-1].append([float(i) for i in fin.readline().strip().split()[1:]])
                if max_frame > -1 and len(data) >= max_frame: break 
        data = torch.tensor(data)
        step = torch.tensor(step)
        return keys, step, data

class LatticeMDDataset(torch.utils.data.Dataset):
    #matter_sequence:           (sequence_length+1, batch_size*system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
    #nonmatter_sequence:        (sequence_length+1, batch_size*system_dim3, number_of_nonmatter_features)

    #matter_sequence_data:      (Nsamples, sequence_length+1, system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
    #nonmatter_sequence_data:   (Nsamples, sequence_length+1, system_dim3, number_of_nonmatter_features)

    #matter_sum_data:           (Nsamples, 1)
    #nonmatter_sum_data:        (Nsamples, sequence_length+1, number_of_nonmatter_features)

    #sequence <- sequence_data.transpose(0, 1).flatten(start_dim = 1, end_dim = 2)
    def __init__(self, matter_sequence_data, nonmatter_sequence_data, matter_sum_data, nonmatter_sum_data, device = 'cpu'):
        self.matter_sequence_data_ = matter_sequence_data.to(device)
        self.nonmatter_sequence_data_ = nonmatter_sequence_data.to(device)
        self.matter_sum_data_ = matter_sum_data.to(device)
        self.nonmatter_sum_data_ = nonmatter_sum_data.to(device)

    def __len__(self):
        return len(self.matter_sequence_data_)

    def __getitem__(self, index):
        return self.matter_sequence_data_[index], self.nonmatter_sequence_data_[index], self.matter_sum_data_[index], self.nonmatter_sum_data_[index]

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
    # matter_conv_layer_kernal_size: (int, ...)
    # matter_conv_layer_stride:      (int, ...)
    # matter_conv_layer_stride:      (int, ...)
    # matter_linear_encoded_size:    int
    # number_of_nonmatter_features:  int
    # nonmatter_decoder_layer_size:  (int, ...)
    #
    def __init__(self, number_of_matters, matter_dim, \
                 matter_conv_layer_kernal_size = (), matter_conv_layer_stride = (), \
                 matter_linear_layer_size = (16, 16), matter_linear_encoded_size = 16, \
                 number_of_nonmatter_features = 7, nonmatter_decoder_layer_size = (16, 16)):
        super(LatticeMD, self).__init__()
        if len(matter_conv_layer_kernal_size) != len(matter_conv_layer_stride):
            print("Error: matter_conv_layer_kernal_size and matter_conv_layer_stride should have the same sizes.")
            exit()
        self.number_of_matters_ = number_of_matters
        self.matter_dim_ = matter_dim
        self.matter_dim3_ = matter_dim**3
        self.number_of_neighbor_block_ = 6 + 1
        self.number_of_nonmatter_features_ = number_of_nonmatter_features # default: stress tensor and pe
        
        if len(matter_conv_layer_kernal_size) > 0:
            self.matter_ae_type_ = 0
            self.matter_encoded_dim_ = matter_dim
            self.matter_ae_ = ConvAutoencoder(number_of_matters, matter_conv_layer_kernal_size, matter_conv_layer_stride)
            for index in range(len(matter_conv_layer_kernal_size)): self.matter_encoded_dim_ = (self.matter_encoded_dim_ - 1 - (matter_conv_layer_kernal_size[index]-1))//matter_conv_layer_stride[index] + 1
            self.matter_encoded_flatten_dim_ = self.number_of_matters_ * self.matter_encoded_dim_**3
        elif len(matter_linear_layer_size) > 0:
            self.matter_ae_type_ = 1
            input_size = self.number_of_matters_ * self.matter_dim_**3
            self.matter_ae_ = MLPAutoencoder(input_size, matter_linear_encoded_size, list(matter_linear_layer_size))
            self.matter_encoded_flatten_dim_ = matter_linear_encoded_size
        else:
            self.matter_ae_type_ = 2
            self.matter_ae_ = DummyAutoencoder()
            self.matter_encoded_flatten_dim_ = self.number_of_matters_ * self.matter_dim_**3

        self.lstm_out_dim_ = self.matter_encoded_flatten_dim_ + self.number_of_nonmatter_features_
        self.lstm_in_dim_ = self.number_of_neighbor_block_ * self.lstm_out_dim_
        self.lstm_ = nn.LSTM(self.lstm_in_dim_, self.lstm_out_dim_)

        nonmatter_decoder_layer_size = [self.number_of_nonmatter_features_] + list(nonmatter_decoder_layer_size) + [self.number_of_nonmatter_features_]
        self.number_of_nonmatter_decoder_layer_ = len(nonmatter_decoder_layer_size) - 1
        self.nonmatter_decoder_ = nn.ModuleList()
        for index in range(self.number_of_nonmatter_decoder_layer_):
            self.nonmatter_decoder_.append(nn.Linear(nonmatter_decoder_layer_size[index], nonmatter_decoder_layer_size[index + 1]))
        self.nonmatter_decoder_actv_ = nn.Tanh()

        self.param  = list()
        self.param += list(self.matter_ae_.parameters())
        self.param += list(self.lstm_.parameters())
        self.param  += list(self.nonmatter_decoder_.parameters())

        self.print_info()

    def to(self, device):
        self.matter_ae_.to(device)
        self.lstm_.to(device)
        self.nonmatter_decoder_.to(device)

    def parameters(self):
        return self.param
    
    def print_info(self):
        print("%35s%d"%("Number of matter types: ", self.number_of_matters_))
        print("%35s%d"%("Number of Neighbor Block: ", self.number_of_neighbor_block_))
        print("%35s%d"%("Number of Non-matter features: ", self.number_of_nonmatter_features_))
        print("%35s%d, %d, %d, %d"%("Matter Image dim: ", self.number_of_matters_, self.matter_dim_, self.matter_dim_, self.matter_dim_))
        print("%35s%6s"%("Matter Autoencoder type: ", "Conv" if self.matter_ae_type_ == 0 else "MLP"))
        print("%35s%d"%("Matter Image Encoded flat dim: ", self.matter_encoded_flatten_dim_))
        print("%35s%3d <- (%d*%d^3+%d)*%d"%("LSTM input dim: ", self.lstm_in_dim_, self.number_of_matters_, self.matter_encoded_dim_, self.number_of_nonmatter_features_, self.number_of_neighbor_block_))
        print("%35s%3d <- (%d*%d^3+%d)"%("LSTM output dim: ", self.lstm_out_dim_, self.number_of_matters_, self.matter_encoded_dim_, self.number_of_nonmatter_features_))
        
    def forward(self, matter_sequence, nonmatter_sequence, system_dim, matter_normalization = False, total_matter = 1.0):
        #input
        #   matter_sequence:        (sequence_length, batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        #   nonmatter_sequence:     (sequence_length, batch_size * system_dim3, number_of_nonmatter_features)
        #   system_dim:             (batch_size, 3)
        #output
        #   predicted_matter:  (batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        #   predicted_nonmatter:    (batch_size * system_dim3, number_of_nonmatter_features)
        encoded_in = self.encode(matter_sequence, nonmatter_sequence)
        lstm_in = self.get_neighbor_block_data(encoded_in, system_dim)
        lstm_out, hidden = self.advance(lstm_in)
        lstm_out = lstm_out[-1] #(batch_size * system_dim3, lstm_out_dim)
        predicted_matter, predicted_nonmatter = self.decode(lstm_out)
        if matter_normalization: predicted_matter = self.matter_normalize(predicted_matter, system_dim, total_matter)
        return predicted_matter, predicted_nonmatter
    
    def get_neighbor_block_data(self, encoded_in, system_dim):
        #encoded_in:        (sequence_length, batch_size * system_dim3, lstm_out_dim)
        lstm_in = encoded_in.view(len(encoded_in), -1, system_dim[0], system_dim[1], system_dim[2], self.lstm_out_dim_) #(sequence_length, batch_size, system_dim[0], system_dim[1], system_dim[2], lstm_out_dim)
        lstm_in = torch.stack((lstm_in, \
                               lstm_in.roll(shifts= 1, dims=2), \
                               lstm_in.roll(shifts=-1, dims=2), \
                               lstm_in.roll(shifts= 1, dims=3), \
                               lstm_in.roll(shifts=-1, dims=3), \
                               lstm_in.roll(shifts= 1, dims=4), \
                               lstm_in.roll(shifts=-1, dims=4))).permute((1, 2, 3, 4, 5, 0, 6))  #(sequence_length, batch_size, system_dim[0], system_dim[1], system_dim[2], number_of_neighbor_block, lstm_out_dim)
        lstm_in = lstm_in.flatten(start_dim = -2)             #(sequence_length, batch_size, system_dim[0], system_dim[1], system_dim[2], lstm_in_dim)
        lstm_in = lstm_in.flatten(start_dim = 1, end_dim = 4) #(sequence_length, batch_size * system_dim3, lstm_in_dim)
        return lstm_in

    def advance(self, lstm_in):
        #lstm_in:                   (sequence_length, batch_size * system_dim3, lstm_in_dim)
        lstm_out, hidden = self.lstm_(lstm_in)
        return lstm_out, hidden

    def encode(self, matter_sequence, nonmatter_sequence):
        #matter_sequence:           (sequence_length, batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        #nonmatter_sequence:        (sequence_length, batch_size * system_dim3, number_of_nonmatter_features)
        matter = matter_sequence.flatten(start_dim = 0, end_dim = 1)                                            #(sequence_length * batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        if self.matter_ae_type_ == 1:
            matter_encoded = self.matter_ae_.encode(matter.flatten(start_dim = 1))                              #(sequence_length * batch_size * system_dim3, matter_encoded_flatten_dim)
        else:
            matter_encoded = self.matter_ae_.encode(matter).flatten(start_dim = 1)                              #(sequence_length * batch_size * system_dim3, number_of_matters, matter_encoded_dim, matter_encoded_dim, matter_encoded_dim)
        matter_encoded = matter_encoded.view(len(nonmatter_sequence), -1, self.matter_encoded_flatten_dim_) #(sequence_length,  batch_size * system_dim3, matter_encoded_flatten_dim)
        encoded_in = torch.cat((matter_encoded, nonmatter_sequence), dim = -1)             #(sequence_length,  batch_size * system_dim3, lstm_out_dim)
        return encoded_in
    
    def decode(self, lstm_out):
        #lstm_out:                  (batch_size * system_dim3, lstm_out_dim)
        if self.matter_ae_type_ == 1:
            matter_encoded = lstm_out[:, :self.matter_encoded_flatten_dim_]  #(batch_size * system_dim3, matter_encoded_flatten_dim)
            matter = self.matter_ae_.decode(matter_encoded) #(batch_size * system_dim3, number_of_matters * matter_dim * matter_dim * matter_dim)
            matter = matter.view(-1, self.number_of_matters_, self.matter_dim_, self.matter_dim_, self.matter_dim_) #(batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        else:
            matter_encoded = lstm_out[:, :self.matter_encoded_flatten_dim_].view(-1, self.number_of_matters_, self.matter_encoded_dim_, self.matter_encoded_dim_, self.matter_encoded_dim_) #(batch_size * system_dim3, number_of_matters, matter_encoded_dim, matter_encoded_dim, matter_encoded_dim)
            matter = self.matter_ae_.decode(matter_encoded) #(batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        nonmatter = lstm_out[:, self.matter_encoded_flatten_dim_:] #(batch_size * system_dim3, number_of_nonmatter_features)
        for index in range(self.number_of_nonmatter_decoder_layer_ - 1):
            nonmatter = self.nonmatter_decoder_[index](nonmatter)
            nonmatter = self.nonmatter_decoder_actv_(nonmatter)
        nonmatter = self.nonmatter_decoder_[-1](nonmatter)

        return matter, nonmatter
    
    def matter_normalize(self, matter, system_dim, total_matter):
        #matter:                    (batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        matter = matter.view(-1, system_dim[0]*system_dim[1]*system_dim[2], self.number_of_matters_, self.matter_dim_, self.matter_dim_, self.matter_dim_)
        matter_sum = matter.sum(dim = (1, 3, 4, 5), keepdim = True) #(batch_size, 1, number_of_matters, 1, 1, 1)
        matter = matter/matter_sum*total_matter
        matter = matter.flatten(start_dim = 0, end_dim = 1)
        return matter


