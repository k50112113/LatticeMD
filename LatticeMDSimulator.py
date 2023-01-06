import torch
import torch.nn as nn
import os
import sys
from LatticeMDModel import LatticeMD, LatticeMDDataset
import dill
from Clock import Clock

class LatticeMDSimulator:
    def __init__(self, model_path):
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("")
        print("Loading model from %s..."%(model_path))
        fin = open(model_path,"rb")
        self.lattice_md_ = dill.load(fin, fout)
        fin.close()

        self.lattice_md_.print_info()
        self.lattice_md_.to(self.device_)
    
    def initialize_box(self, data_path):
        print("\tReading %s..."%(data_path))
        fin = open("%s"%(data_path),"rb")
        sd = dill.load(fin)
        number_of_matters_tmp, matter_dim_tmp, system_dim_tmp, sequence_length_tmp, \
        matter_sequence_data_tmp, stress_sequence_data_tmp, pe_sequence_data_tmp, \
        matter_sum_data_tmp,      stress_sum_data_tmp,      pe_sum_data_tmp = sd.get_data()
        matter_prefactor,     stress_prefactor,     pe_prefactor,\
        matter_sum_prefactor, stress_sum_prefactor, pe_sum_prefactor = sd.get_prefactor()

        fin.close()
        nonmatter_sequence_data_tmp = torch.cat((stress_sequence_data_tmp, pe_sequence_data_tmp), dim = -1)
        print(matter_sequence_data_tmp.shape, nonmatter_sequence_data_tmp.shape)
        self.system_dim_ = system_dim_tmp
        if matter_dim_tmp != self.lattice_md_.matter_dim_:
            print("Error: Unmatched matter dim.")
            exit()
        if number_of_matters_tmp != self.lattice_md_.number_of_matters_:
            print("Error: Unmatched number of matters.")
            exit()
        self.current_matter_sequence = matter_sequence_data_tmp[0][:-1]
        self.current_nonmatter_sequence = nonmatter_sequence_data_tmp[0][:-1]
        self.total_matter_ = 1.0/matter_sum_prefactor

    def run(self):
        next_matter, next_nonmatter = self.lattice_md_(self.current_matter_sequence, self.current_nonmatter_sequence, system_dim, matter_normalization = True, total_matter = self.total_matter_)
        self.current_matter_sequence = torch.cat((self.current_matter_sequence[1:], next_matter.unsqueeze(0)), dim = 0)
        self.current_nonmatter_sequence = torch.cat((self.current_nonmatter_sequence[1:], next_nonmatter.unsqueeze(0)), dim = 0)