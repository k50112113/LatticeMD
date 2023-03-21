import torch
import os
import sys
from LatticeMDData import MDSequenceData, LatticeMDDataset
from LatticeMDModel import LatticeMD
import dill
from Clock import Clock

class LatticeMDSimulator:
    def __init__(self, model_path):
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("")
        print("Loading model from %s..."%(model_path))
        fin = open(model_path,"rb")
        self.lattice_md_ = dill.load(fin, fin)
        fin.close()

        self.lattice_md_.print_info()
        self.lattice_md_.to(self.device_)
    
    def initialize_box(self, data_path):
        print("Reading %s..."%(data_path))
        fin = open("%s"%(data_path),"rb")
        sd = dill.load(fin)
        number_of_matters_tmp, matter_dim_tmp, system_dim_tmp, sequence_length_tmp, \
        matter_sequence_data_tmp, momentum_sequence_data, stress_sequence_data_tmp, pe_sequence_data_tmp, \
        matter_sum_data_tmp,                              stress_sum_data_tmp,      pe_sum_data_tmp = sd.get_data()
        matter_prefactor,         momentum_prefactor, stress_prefactor,     pe_prefactor,\
        matter_sum_prefactor,                         stress_sum_prefactor, pe_sum_prefactor = sd.get_prefactor()
        fin.close()
        nonmatter_sequence_data_tmp = torch.cat((stress_sequence_data_tmp, pe_sequence_data_tmp), dim = -1)
        self.system_dim_ = system_dim_tmp
        if matter_dim_tmp != self.lattice_md_.matter_dim_:
            print("Error: Unmatched matter dim.")
            exit()
        if number_of_matters_tmp != self.lattice_md_.number_of_matters_:
            print("Error: Unmatched number of matters.")
            exit()
        self.number_of_matters_ = self.lattice_md_.number_of_matters_
        self.matter_dim_ = self.lattice_md_.matter_dim_
        nth_batch = 0
        self.current_matter_sequence = matter_sequence_data_tmp[nth_batch][:-1]
        self.current_nonmatter_sequence = nonmatter_sequence_data_tmp[nth_batch][:-1]
        self.total_matter_ = 1.0/matter_sum_prefactor

        self.current_matter_sequence = self.current_matter_sequence.to(self.device_)
        self.current_nonmatter_sequence = self.current_nonmatter_sequence.to(self.device_)
        self.total_matter_ = self.total_matter_.to(self.device_)

    def run(self, steps, output_filename):
        output_data = MDSequenceData(system_dim = self.system_dim_)
        output_data.number_of_matters_ = self.number_of_matters_
        output_data.matter_dim_ = self.matter_dim_
        output_data.matter_sequence_data_ = self.current_matter_sequence.detach().clone().cpu()
        output_data.nonmatter_sequence_data_ = self.current_nonmatter_sequence.detach().clone().cpu()
        for i_step in range(steps):
            next_matter, next_nonmatter = self.lattice_md_(self.current_matter_sequence, self.current_nonmatter_sequence, self.system_dim_)
            max_dmatter = next_matter.max().item()

            next_matter    += self.current_matter_sequence[-1]
            self.current_matter_sequence = torch.cat((self.current_matter_sequence[1:], next_matter.unsqueeze(0)), dim = 0)
            output_data.matter_sequence_data_ = torch.cat((output_data.matter_sequence_data_, next_matter.unsqueeze(0).detach().clone().cpu()), dim = 0)
            
            if next_nonmatter:
                next_nonmatter += self.current_nonmatter_sequence[-1]   
                self.current_nonmatter_sequence = torch.cat((self.current_nonmatter_sequence[1:], next_nonmatter.unsqueeze(0)), dim = 0)
                output_data.nonmatter_sequence_data_ = torch.cat((output_data.nonmatter_sequence_data_, next_nonmatter.unsqueeze(0).detach().clone().cpu()), dim = 0)
            
            print("%d %.2f %.2f %f"%(i_step, next_matter.sum().item(), next_matter.max().item(), max_dmatter))
        
        output_data.matter_sequence_data_ = output_data.matter_sequence_data_.unsqueeze(1)
        output_data.nonmatter_sequence_data_ = output_data.nonmatter_sequence_data_.unsqueeze(1)
        fout = open(output_filename,"wb")
        dill.dump(output_data, fout)
        fout.close()