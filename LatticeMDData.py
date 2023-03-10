import torch
import torch.nn as nn
import os

class DimensionalityException(Exception): pass
class TimestepException(Exception): pass
class NumberOfElementsException(Exception): pass

class MDSequenceData:
    def __init__(self, system_dim):
        self.system_dim_ = system_dim
        self.system_dim3_ = system_dim[0]*system_dim[1]*system_dim[2]
        self.number_of_matters_      = None
        self.matter_dim_             = None
        self.sequence_length_        = None
        self.matter_sequence_data_   = None #(N_batches, sequence_length, system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        self.momentum_sequence_data_ = None
        self.stress_sequence_data_   = None
        self.pe_sequence_data_       = None
        self.matter_sum_data_        = None
        self.stress_sum_data_        = None
        self.pe_sum_data_            = None

        self.matter_prefactor_       = None
        self.momentum_prefactor_     = None
        self.stress_prefactor_       = None
        self.pe_prefactor_           = None
        self.matter_sum_prefactor_   = None
        self.stress_sum_prefactor_   = None
        self.pe_sum_prefactor_       = None

    def load_data(self, input_dir, sequence_length, max_md_snapshot = -1, md_snapshot_start_indices = [0], output_dir='./', select_matter = (), moving_average_length = 1, md_snapshot_per_frame = 1,\
                  load_data = {"pe":True, "stress":True, "momentum":True}):
        #sequence_length = LSTM length
        #max_md_snapshot = maximum number of MD snapshots to load
        #md_snapshot_start_indices = number of MD snapshots to trim from start
        #moving_average_length = number of MD snapshots to average per LSTM unit
        #md_snapshot_per_frame = number of MD snapshots per LSTM unit
        self.output_dir_ = output_dir
        self.sequence_length_ = sequence_length
        if os.path.isdir(self.output_dir_) == False: os.system("mkdir -p %s"%(self.output_dir_))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            # read matter, amount of the matter of each chunk. [unit: g/mole*Ang/fs]
            print("Reading %s/matter.txt..."%(input_dir))
            keys, step, data = self.read_lammps_ave("%s/matter.txt"%(input_dir), max_md_snapshot = max_md_snapshot)
            self.data_step_ = step
            if len(select_matter) == 0:
                self.number_of_matters_ = len(keys)
                data = data
            else:
                self.number_of_matters_ = len(select_matter)
                data = data[:, :, select_matter]
            self.matter_dim_ = int(round((data.shape[1]/self.system_dim3_)**(1/3), 0))
            if self.matter_dim_**3 * self.system_dim3_ != data.shape[1]: raise DimensionalityException
            matter_data = data
            matter_data = matter_data.to(device)
            if load_data.get("pe"):
                # read pe: potential energy of each chunk. [unit: kcal/mole]
                print("Reading %s/pe.txt..."%(input_dir))
                keys, step, data = self.read_lammps_ave("%s/pe.txt"%(input_dir), max_md_snapshot = max_md_snapshot)
                if len(self.data_step_) != len(step) or ((self.data_step_-step)**2).sum() > 0.0: raise TimestepException
                if len(keys) != 1: raise NumberOfElementsException
                if self.system_dim3_ != data.shape[1]: raise DimensionalityException
                pe_data     = data
                pe_data     = pe_data.to(device)
                pe_sum_data = pe_data.sum(dim = 1)
            if load_data.get("stress"):
                # read stress, virial stress tensor of each chunk. [unit: atm]
                print("Reading %s/stress.txt..."%(input_dir))
                keys, step, data = self.read_lammps_ave("%s/stress.txt"%(input_dir), max_md_snapshot = max_md_snapshot)
                if len(self.data_step_) != len(step) or ((self.data_step_-step)**2).sum() > 0.0: raise TimestepException
                if len(keys) != 6:  raise NumberOfElementsException
                if self.system_dim3_ != data.shape[1]: raise DimensionalityException
                stress_data     = data
                stress_data     = stress_data.to(device)
                stress_sum_data = stress_data.sum(dim = 1)
            if load_data.get("momentum"):
                # read momentum, momentum of each chunk. [unit: ]
                print("Reading %s/momentum.txt..."%(input_dir))
                keys, step, data = self.read_lammps_ave("%s/momentum.txt"%(input_dir), max_md_snapshot = max_md_snapshot)
                if len(self.data_step_) != len(step) or ((self.data_step_-step)**2).sum() > 0.0: raise TimestepException
                if len(select_matter) == 0:
                    if self.number_of_matters_*3 != len(keys): raise NumberOfElementsException
                    data = data
                else:
                    select_momentum = []
                    for a_matter in select_matter: select_momentum += [a_matter*3, a_matter*3+1, a_matter*3+2]
                    select_momentum = tuple(select_momentum)
                    data = data[:, :, select_momentum]
                matter_dim_tmp = int(round((data.shape[1]/self.system_dim3_)**(1/3), 0))
                if self.matter_dim_ != matter_dim_tmp: raise DimensionalityException
                momentum_data = data
                momentum_data   = momentum_data.to(device)
            # read matter sum
            print("Reading %s/matter_sum_list.txt..."%(input_dir))
            with open("%s/matter_sum_list.txt"%(input_dir), 'r') as fin:
                matter_sum_data = torch.tensor([float(i) for i in fin.readline().strip().split()])
            matter_sum_data = matter_sum_data.to(device)

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

        #convert S*V/V to S*V/v (deprecated)
        #stress_data *= self.system_dim3_
        #stress_sum_data *= self.system_dim3_

        #compute prefactor
        self.matter_prefactor_     = 1.0/self.compute_moving_average_std(matter_data,     moving_average_length, md_snapshot_per_frame, dim = (0, 1))
        self.matter_sum_prefactor_ = 1.0/matter_sum_data
        if load_data.get("pe"):
            self.pe_prefactor_         = 1.0/self.compute_moving_average_std(pe_data,         moving_average_length, md_snapshot_per_frame, dim = (0, 1))
            self.pe_sum_prefactor_     = 1.0/self.compute_moving_average_std(pe_sum_data,     moving_average_length, md_snapshot_per_frame, dim = 0)
        if load_data.get("stress"):
            self.stress_prefactor_     = 1.0/self.compute_moving_average_std(stress_data,     moving_average_length, md_snapshot_per_frame, dim = (0, 1))
            self.stress_sum_prefactor_ = 1.0/self.compute_moving_average_std(stress_sum_data, moving_average_length, md_snapshot_per_frame, dim = 0)
        if load_data.get("momentum"):
            self.momentum_prefactor_   = 1.0/self.compute_moving_average_std(momentum_data,   moving_average_length, md_snapshot_per_frame, dim = (0, 1))

        #split data into sequences
        self.matter_sequence_data_   = self.split_data_sequence(matter_data,     self.sequence_length_+1, md_snapshot_start_indices, moving_average_length, md_snapshot_per_frame, verbose = True)
        self.number_of_batches_      = len(self.matter_sequence_data_)
        ss = self.sequence_length_ + 1
        self.matter_sequence_data_   = self.matter_sequence_data_  .view(self.number_of_batches_, ss, self.system_dim_[0], self.matter_dim_, self.system_dim_[1], self.matter_dim_, self.system_dim_[2], self.matter_dim_, self.number_of_matters_)   .permute(0, 1, 2, 4, 6, 8, 3, 5, 7)
        self.matter_sequence_data_   = self.matter_sequence_data_  .flatten(start_dim = 2, end_dim = 4)
        if load_data.get("pe"):
            self.pe_sequence_data_       = self.split_data_sequence(pe_data,         self.sequence_length_+1, md_snapshot_start_indices, moving_average_length, md_snapshot_per_frame)
            self.pe_sum_data_            = self.split_data_sequence(pe_sum_data,     self.sequence_length_+1, md_snapshot_start_indices, moving_average_length, md_snapshot_per_frame)
        if load_data.get("stress"):
            self.stress_sequence_data_   = self.split_data_sequence(stress_data,     self.sequence_length_+1, md_snapshot_start_indices, moving_average_length, md_snapshot_per_frame)
            self.stress_sum_data_        = self.split_data_sequence(stress_sum_data, self.sequence_length_+1, md_snapshot_start_indices, moving_average_length, md_snapshot_per_frame)
        if load_data.get("momentum"):
            self.momentum_sequence_data_ = self.split_data_sequence(momentum_data,   self.sequence_length_+1, md_snapshot_start_indices, moving_average_length, md_snapshot_per_frame)        
            self.momentum_sequence_data_ = self.momentum_sequence_data_.view(self.number_of_batches_, ss, self.system_dim_[0], self.matter_dim_, self.system_dim_[1], self.matter_dim_, self.system_dim_[2], self.matter_dim_, self.number_of_matters_, 3).permute(0, 1, 2, 4, 6, 8, 9, 3, 5, 7)
            self.momentum_sequence_data_ = self.momentum_sequence_data_.flatten(start_dim = 2, end_dim = 4)
        
        self.matter_sum_data_      = torch.tile(matter_sum_data, (self.number_of_batches_, 1))
        self.matter_prefactor_     = self.matter_prefactor_.cpu()
        self.matter_sum_prefactor_ = self.matter_sum_prefactor_.cpu()
        self.matter_sequence_data_   = self.matter_sequence_data_.cpu()
        self.matter_sum_data_        = self.matter_sum_data_.cpu()
        if load_data.get("pe"):
            self.pe_prefactor_         = self.pe_prefactor_.cpu()
            self.pe_sum_prefactor_     = self.pe_sum_prefactor_.cpu()
            self.pe_sequence_data_       = self.pe_sequence_data_.cpu()
            self.pe_sum_data_            = self.pe_sum_data_.cpu()
        if load_data.get("stress"):
            self.stress_prefactor_     = self.stress_prefactor_.cpu()
            self.stress_sum_prefactor_ = self.stress_sum_prefactor_.cpu()
            self.stress_sum_data_        = self.stress_sum_data_.cpu()
            self.stress_sequence_data_   = self.stress_sequence_data_.cpu()
        if load_data.get("momentum"):
            self.momentum_prefactor_   = self.momentum_prefactor_.cpu()
            self.momentum_sequence_data_ = self.momentum_sequence_data_.cpu()
        self.print_info()

    def print_info(self):
        print("")
        print("%30s%d"%("Number of batches: ",len(self.matter_sequence_data_)))
        print("%30s%d"%("Sequence length: ",self.sequence_length_))
        print("%30s%d %d %d"%("System dim: ", self.system_dim_[0], self.system_dim_[1], self.system_dim_[2]))
        print("%30s%d %d^3"%("Matter Image dim: ", self.number_of_matters_, self.matter_dim_))

    def get_basic(self):
        return self.number_of_matters_, self.matter_dim_, self.system_dim_, self.sequence_length_
    
    def get_matter(self):
        return self.matter_sequence_data_, self.matter_prefactor_
    
    def get_momentum(self):
        return self.momentum_sequence_data_, self.momentum_prefactor_
    
    def get_stress(self):
        return self.stress_sequence_data_, self.stress_sum_data_, self.stress_prefactor_, self.stress_sum_prefactor_

    def get_pe(self):
        return self.pe_sequence_data_, self.pe_sum_data_, self.pe_prefactor_, self.pe_sum_prefactor_

    def get_data(self):
        return self.number_of_matters_, self.matter_dim_, self.system_dim_, self.sequence_length_, \
               self.matter_sequence_data_, self.momentum_sequence_data_, self.stress_sequence_data_, self.pe_sequence_data_, \
               self.matter_sum_data_,                                    self.stress_sum_data_,      self.pe_sum_data_

    def get_prefactor(self):
        return self.matter_prefactor_,     self.momentum_prefactor_,    self.stress_prefactor_,     self.pe_prefactor_,\
               self.matter_sum_prefactor_,                              self.stress_sum_prefactor_, self.pe_sum_prefactor_

    def compute_moving_average_std(self, x, moving_average_length, md_snapshot_per_frame, dim = 0):
        if moving_average_length == 1: return x.std(dim = dim, unbiased = False)
        x_ma_tmp = self.moving_average(x, moving_average_length)[moving_average_length-1::md_snapshot_per_frame]
        x_tmp = x[moving_average_length:][moving_average_length-1::md_snapshot_per_frame]
        x_ma_std = (x_ma_tmp - x_tmp).std(dim = dim, unbiased = False)
        return x_ma_std

    def moving_average(self, x, moving_average_length, dim = 0):
        if moving_average_length == 1: return x
        return (x.cumsum(dim = dim)[moving_average_length:] - x.cumsum(dim = dim)[:-moving_average_length])/moving_average_length
    
    def split_data_sequence(self, data, sequence_length, md_snapshot_start_indices, moving_average_length, md_snapshot_per_frame, verbose = False):
        split_data = None
        for sequence_start_index in md_snapshot_start_indices:
            data_ma = self.moving_average(data[sequence_start_index:], moving_average_length)[moving_average_length-1::md_snapshot_per_frame]
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
                print("\tnumber of MD snapshots per frame = %d"%(md_snapshot_per_frame))
                print("\tdata range = %d ~ %d"%(sequence_start_index, len(data)))
                print("\tbatch subtotal = %d"%(len(split_data)))
        return split_data

    def read_lammps_ave(self, filename, max_md_snapshot = -1):
        #keys: [total_number_of_matters]
        #step: (total_sequence_length)
        #data: (total_sequence_length, system_dim3*matter_dim3, total_number_of_matters)   for matter
        #      (total_sequence_length, system_dim3*matter_dim3, total_number_of_matters*3) for momentum
        #      (total_sequence_length, system_dim3,             6)                         for stress tensor
        #      (total_sequence_length, system_dim3,             1)                         for potential energy
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
                if max_md_snapshot > -1 and len(data) >= max_md_snapshot: break 
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
