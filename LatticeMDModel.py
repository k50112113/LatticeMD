import torch
import torch.nn as nn
import os
#torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_tensor_type(torch.DoubleTensor)

class DimensionalityException(Exception): pass
class TimestepException(Exception): pass
class NumberOfElementsException(Exception): pass

class MDSequenceData:
    def __init__(self, input_dir, system_dim, sequence_length, sequence_start_indices = [0], output_dir='./', select_matter = ()):
        self.output_dir_ = output_dir
        self.system_dim_ = system_dim
        self.system_dim3_ = system_dim[0]*system_dim[1]*system_dim[2]
        self.sequence_length_ = sequence_length
        if os.path.isdir(self.output_dir_) == False: os.system("mkdir -p %s"%(self.output_dir_))
        
        try:
            # read pe
            keys, step, data = self.read_lammps_ave("%s/pe.txt"%(input_dir))
            self.data_step_ = step
            if len(keys) != 1: raise NumberOfElementsException
            if self.system_dim3_ != data.shape[1]: raise DimensionalityException
            pe_data = data

            # read stress
            keys, step, data = self.read_lammps_ave("%s/stress.txt"%(input_dir))
            if len(self.data_step_) != len(step) or ((self.data_step_-step)**2).sum() > 0.0: raise TimestepException
            if len(keys) != 6:  raise NumberOfElementsException
            if self.system_dim3_ != data.shape[1]: raise DimensionalityException
            stress_data = data
            
            # read matter
            keys, step, data = self.read_lammps_ave("%s/matter.txt"%(input_dir))
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
            with open("%s/matter_sum_list.txt"%(input_dir), 'r') as fin:
                matter_sum_data = torch.tensor([float(i) for i in fin.readline().strip().split()])
            
            # read stress pe sum
            pe_sum_data = []
            stress_sum_data = []
            step = []
            with open("%s/log.lammps"%(input_dir), 'r') as fin:
                for aline in fin:
                    if "Step PotEng Pxx Pyy Pzz Pxy Pxz Pyz" in aline: break
                fin.readline()
                for aline in fin:
                    if "Loop" in aline: break
                    linelist = aline.strip().split()
                    step.append(int(linelist[0]))
                    pe_sum_data.append(float(linelist[1]))
                    stress_sum_data.append([float(i) for i in linelist[2:]])
            step = torch.tensor(step)
            pe_sum_data = torch.tensor(pe_sum_data).unsqueeze(-1)
            stress_sum_data = torch.tensor(stress_sum_data)
            if len(self.data_step_) != len(step) or ((self.data_step_-step)**2).sum() > 0.0: raise TimestepException

        except DimensionalityException:
            print("Error: Dimensionality ")
            exit()
        except NumberOfElementsException:
            print("Error: Number of elements in the dataset is wrong.")
            exit()
        except TimestepException:
            print("Error: Time steps should be the same.")
            exit()
        
        matter_data_normalization_factor = matter_sum_data/matter_data.sum(dim = 1)
        matter_data *= matter_data_normalization_factor.unsqueeze(1)
        
        #compute prefactor
        self.matter_prefactor_ = 1.0/matter_data.mean(dim = (0, 1))
        self.stress_prefactor_ = 1.0/stress_data.std(dim = (0, 1), unbiased = False)
        self.pe_prefactor_ = 1.0/pe_data.std(dim = (0, 1), unbiased = False)
        self.matter_sum_prefactor_ = 1.0/matter_sum_data
        self.stress_sum_prefactor_ = 1.0/stress_sum_data.std(dim = 0, unbiased = False)
        self.pe_sum_prefactor_ = 1.0/pe_sum_data.std(dim = 0, unbiased = False)

        #split data into sequences
        self.matter_sequence_data_ = self.split_data_sequence(matter_data,     self.sequence_length_+1, sequence_start_indices)
        self.number_of_batches_    = len(self.matter_sequence_data_)
        self.matter_sequence_data_ = self.matter_sequence_data_.view(self.number_of_batches_, self.sequence_length_+1, self.system_dim3_, self.matter_dim_, self.matter_dim_, self.matter_dim_, self.number_of_matters_).permute(0,1,2,6,3,4,5)
        self.stress_sequence_data_     = self.split_data_sequence(stress_data,     self.sequence_length_+1, sequence_start_indices)
        self.pe_sequence_data_         = self.split_data_sequence(pe_data,         self.sequence_length_+1, sequence_start_indices)
        self.stress_sum_data_          = self.split_data_sequence(stress_sum_data, self.sequence_length_+1, sequence_start_indices)
        self.pe_sum_data_              = self.split_data_sequence(pe_sum_data,     self.sequence_length_+1, sequence_start_indices)
        self.matter_sum_data_          = torch.tile(matter_sum_data, (self.number_of_batches_, 1))
        
        self.print_info()

    def print_info(self):
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

    def split_data_sequence(self, data, sequence_length, sequence_start_indices):
        split_data = None
        for sequence_start_index in sequence_start_indices:
            data_tmp = data[sequence_start_index:].split(sequence_length)
            if len(data_tmp[-1]) != sequence_length: data_tmp = data_tmp[:-1]
            data_tmp = torch.stack(data_tmp)
            if split_data is not None: split_data = torch.cat((split_data, data_tmp))
            else: split_data = data_tmp
        return split_data

    def read_lammps_ave(self, filename):
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
        data = torch.tensor(data)
        step = torch.tensor(step)
        return keys, step, data

class LatticeMDDataset(torch.utils.data.Dataset):
    #matter_sequence:           (sequence_length+1, batch_size*system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
    #stress_pe_sequence:        (sequence_length+1, batch_size*system_dim3, 6 + 1)

    #matter_sequence_data:      (Nsamples, sequence_length+1, system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
    #stress_pe_sequence_data:   (Nsamples, sequence_length+1, system_dim3, 6 + 1)

    #matter_sum_data:           (Nsamples, 1)
    #stress_pe_sum_data:        (Nsamples, sequence_length+1, 6 + 1)

    #sequence <- sequence_data.transpose(0, 1).flatten(start_dim = 1, end_dim = 2)
    def __init__(self, matter_sequence_data, stress_pe_sequence_data, matter_sum_data, stress_pe_sum_data, device = 'cpu'):
        self.matter_sequence_data_ = matter_sequence_data.to(device)
        self.stress_pe_sequence_data_ = stress_pe_sequence_data.to(device)
        self.matter_sum_data_ = matter_sum_data.to(device)
        self.stress_pe_sum_data_ = stress_pe_sum_data.to(device)

    def __len__(self):
        return len(self.matter_sequence_data_)

    def __getitem__(self, index):
        return self.matter_sequence_data_[index], self.stress_pe_sequence_data_[index], self.matter_sum_data_[index], self.stress_pe_sum_data_[index]

class MatterImageAutoencoder(nn.Module):
    def __init__(self, number_of_matters, matter_conv_layer_kernal_size, matter_conv_layer_stride):
        super(MatterImageAutoencoder, self).__init__()
        if len(matter_conv_layer_kernal_size) != len(matter_conv_layer_stride):
            print("Error: matter_conv_layer_kernal_size and matter_conv_layer_stride should have the same sizes.")

        self.matter_conv_layer_ = nn.ModuleList()   # dim_out = (dim_in - 1 - (kernal_size-1))//stride + 1
        self.matter_deconv_layer_ = nn.ModuleList() # dim_out = (dim_in - 1                  ) *stride + 1 + (kernal_size-1)
        self.number_of_conv_layer_ = len(matter_conv_layer_kernal_size)

        for index in range(self.number_of_conv_layer_):
            self.matter_conv_layer_.append(nn.Conv3d(number_of_matters, number_of_matters, matter_conv_layer_kernal_size[index], stride = matter_conv_layer_stride[index]))
        for stride in range(self.number_of_conv_layer_):
            self.matter_deconv_layer_.append(nn.ConvTranspose3d(number_of_matters, number_of_matters, matter_conv_layer_kernal_size[index], stride = matter_conv_layer_stride[index]))

        self.actv_ = nn.ReLU()

    def encode(self, x):
        #x: (batch_size, number_of_matters, matter_dim, matter_dim, matter_dim)
        y = x
        for index in range(self.number_of_conv_layer_ - 1):
            y = self.matter_conv_layer_[index](y)
            y = self.actv_(y)
        y = self.matter_conv_layer_[-1](y)
        return y
    
    def decode(self, y):
        x = y
        for index in range(self.number_of_conv_layer_ - 1):
            x = self.matter_deconv_layer_[index](x)
            x = self.actv_(x)
        x = self.matter_deconv_layer_[-1](x)
        return x
    
    def forward(self, x):
        return self.decode(self.encode(x))

class DummyAutoencoder:
    def __init__(self):   pass
    def encode(self, x):  return x
    def decode(self, x):  return x
    def parameters(self): return list()
    def to(self, device): pass

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
    # matter_conv_layer_kernal_size: (int, ...)
    #
    def __init__(self, number_of_matters, matter_dim, matter_conv_layer_kernal_size = (), matter_conv_layer_stride = ()):
        super(LatticeMD, self).__init__()
        if len(matter_conv_layer_kernal_size) != len(matter_conv_layer_stride):
            print("Error: matter_conv_layer_kernal_size and matter_conv_layer_stride should have the same sizes.")
            exit()
        self.number_of_matters_ = number_of_matters
        self.matter_dim_ = matter_dim
        self.matter_dim3_ = matter_dim**3
        self.number_of_neighbor_block_ = 6 + 1
        
        self.matter_encoded_dim_ = matter_dim
        if len(matter_conv_layer_kernal_size) > 0:
            self.matter_ae_ = MatterImageAutoencoder(number_of_matters, matter_conv_layer_kernal_size, matter_conv_layer_stride)
            for index in range(len(matter_conv_layer_kernal_size)): self.matter_encoded_dim_ = (self.matter_encoded_dim_ - 1 - (matter_conv_layer_kernal_size[index]-1))//matter_conv_layer_stride[index] + 1
            self.matter_encoded_dim_ = matter_encoded_dim
        else:
            self.matter_ae_ = DummyAutoencoder()
        self.matter_encoded_flatten_dim_ = self.number_of_matters_ * self.matter_encoded_dim_**3

        self.lstm_out_dim_ = self.matter_encoded_flatten_dim_ + 6 + 1
        self.lstm_in_dim_ = self.number_of_neighbor_block_ * self.lstm_out_dim_
        self.lstm_ = nn.LSTM(self.lstm_in_dim_, self.lstm_out_dim_)

        self.param  = list()
        self.param += list(self.matter_ae_.parameters())
        self.param += list(self.lstm_.parameters())

        self.print_info()

    def to(self, device):
        self.matter_ae_.to(device)
        self.lstm_.to(device)

    def parameters(self):
        return self.param
    
    def print_info(self):
        print("%30s%d %d^3"%("Matter Image dim: ", self.number_of_matters_, self.matter_dim_))
        print("%30s%d %d^3"%("Matter Image Encoded dim: ", self.number_of_matters_, self.matter_encoded_dim_))
        print("%30s%d"%("Number of Neighbor Block: ", self.number_of_neighbor_block_))
        print("%30s%3d <- (%d*%d+6+1)*%d"%("LSTM input dim: ", self.lstm_in_dim_, self.number_of_matters_, self.matter_dim_, self.number_of_neighbor_block_))
        print("%30s%3d <- (%d*%d+6+1)"%("LSTM output dim: ", self.lstm_out_dim_, self.number_of_matters_, self.matter_dim_))

    def forward(self, matter_sequence, stress_pe_sequence, system_dim, matter_normalization = False):
        #input
        #   matter_sequence:        (sequence_length, batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        #   stress_pe_sequence:     (sequence_length, batch_size * system_dim3, 6 + 1)
        #   system_dim:             (batch_size, 3)
        #output
        #   predicted_matter:       (batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        #   predicted_stress_pe:    (batch_size * system_dim3, 6 + 1)
        encoded_in = self.encode(matter_sequence, stress_pe_sequence)
        lstm_in = self.get_neighbor_block_data(encoded_in, system_dim)
        lstm_out, hidden = self.advance(lstm_in)
        lstm_out = lstm_out[-1] #(batch_size * system_dim3, lstm_out_dim)
        predicted_matter, predicted_stress_pe = self.decode(lstm_out)
        if matter_normalization: predicted_matter = self.matter_normalize(predicted_matter)
        return predicted_matter, predicted_stress_pe
    
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

    def encode(self, matter_sequence, stress_pe_sequence):
        #matter_sequence:           (sequence_length, batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        #stress_pe_sequence: (sequence_length, batch_size * system_dim3, 6 + 1)
        matter = matter_sequence.flatten(start_dim = 0, end_dim = 1)                        #(sequence_length * batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        matter_encoded = self.matter_ae_.encode(matter).flatten(start_dim = 2)             #(sequence_length * batch_size * system_dim3, matter_encoded_flatten_dim)
        matter_encoded = matter_encoded.view(len(stress_pe_sequence), -1, self.matter_encoded_flatten_dim_) #(sequence_length,  batch_size * system_dim3, matter_encoded_flatten_dim)
        encoded_in = torch.cat((matter_encoded, stress_pe_sequence), dim = -1)      #(sequence_length,  batch_size * system_dim3, lstm_out_dim)
        return encoded_in
    
    def decode(self, lstm_out):
        #lstm_out:                  (batch_size * system_dim3, lstm_out_dim)
        matter_encoded = lstm_out[:, :self.matter_encoded_flatten_dim_].view(-1, self.number_of_matters_, self.matter_encoded_dim_, self.matter_encoded_dim_, self.matter_encoded_dim_) #(batch_size * system_dim3, number_of_matters, matter_encoded_dim, matter_encoded_dim, matter_encoded_dim)
        matter = self.matter_ae_.decode(matter_encoded) #(batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        stress_pe = lstm_out[:, self.matter_encoded_flatten_dim_:] #(batch_size * system_dim3, 6 + 1)
        return matter, stress_pe
    
    def matter_normalize(self, matter, system_dim):
        #matter:           (batch_size * system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        mattersum = matter.view(-1, system_dim[0]*system_dim[1]*system_dim[2], self.number_of_matters_, self.matter_dim_, self.matter_dim_, self.matter_dim_)
        mattersum = mattersum.sum(dim = (1, 3, 4, 5), keepdim = True) #(batch_size, 1, number_of_matters, 1, 1, 1)
        matter = matter/mattersum
        return matter

