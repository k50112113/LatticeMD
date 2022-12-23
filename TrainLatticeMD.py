import torch
import torch.nn as nn
import os
import sys
from LatticeMDModel import LatticeMD, LatticeMDDataset
import dill
from Clock import Clock

class NumberOfMattersException(Exception): pass
class MatterDimException(Exception): pass
class SequenceLengthException(Exception): pass

class TrainLatticeMD:
    def __init__(self, input_filename):
        self.input_filename_ = input_filename
        self.read_input_file()

        ###################################### Load basic information ######################################
        print("%s - double precision."%(self.settings_["mode"]))
        self.output_dir = self.settings_["output_dir"]
        self.output_log = self.settings_["output_log"]
        if os.path.isdir(self.output_dir) == False: os.mkdir(self.output_dir)
        self.original_stdout = sys.stdout 
        if self.output_log == "stdout": self.logfile = self.original_stdout
        else:                           self.logfile = open(self.output_dir+"/"+self.output_log, "w")
        sys.stdout = self.logfile
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.random_split_seed_              = 42  if self.settings_.get("random_split_seed") is None else int(self.settings_["random_split_seed"])
        self.batch_size_                     = 1   if self.settings_.get("batch_size")        is None else int(self.settings_["batch_size"])
        self.test_ratio_                     = 0.1 if self.settings_.get("test_ratio")         is None else float(self.settings_["test_ratio"])
        
        ###################################### Load basic information ######################################

        if self.settings_["mode"] == "train": self.init_training()
        elif self.settings_["mode"] == "test": self.init_test()

        ###################################### Load MD sequence data ######################################
        print("Loading MD sequence data...")
        self.training_data_dir_list_ = self.settings_["training_data_dir_list"].strip().split()
        self.load_training_data()
        ###################################### Load MD sequence data ######################################

        ###################################### Create LatticeMD model ######################################
        print("Creating LatticeMD Models...")
        self.create_model()
        ###################################### Create LatticeMD model ######################################

        self.apply_loss()

        self.train()
        sys.stdout = self.original_stdout
        
    def create_model(self):
        self.lattice_md_ = LatticeMD(number_of_matters = self.number_of_matters_, matter_dim = self.matter_dim_, matter_conv_layer_kernal_size = (), matter_conv_layer_stride = ())
        self.lattice_md_.to(self.device_)
        # batch_size = 10
        # matter_sequence           = self.batched_data_sequence_to_model_input(self.matter_sequence_data_[:batch_size])
        # stress_pe_sequence = self.batched_data_sequence_to_model_input(self.stress_pe_sequence_data_[:batch_size])
        # print(matter_sequence.shape, stress_pe_sequence.shape)
        # predicted_matter, predicted_stress_pe = lmd(matter_sequence, stress_pe_sequence, system_dim, matter_normalization = False)
        # print(predicted_matter.shape, predicted_stress_pe.shape)

    def batched_data_sequence_to_model_input(self, x):
        #input:
        #   x:      (Nbatch, sequence_length+1, system_dim3, ...)
        #output:
        #   return: (sequence_length+1, batch_size*system_dim3, ...)
        return x.transpose(0, 1).flatten(1, 2)

    def apply_loss(self):
        self.loss_function_ = nn.MSELoss(reduction='mean')

    def apply_optimize(self, method='adam'):
        if method == 'adam':
            self.optimization = torch.optim.Adam(self.lattice_md_.parameters(), lr=self.learning_rate_start_, weight_decay = self.weight_decay_)
        elif method == 'sgd':
            self.optimization = torch.optim.SGD(self.lattice_md_.parameters(), lr=self.learning_rate_start_, momentum = 0.3 ,dampening = 0.01, weight_decay = self.weight_decay_)
        self.learning_rate_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimization, lr_lambda=lambda epoch: self.learning_rate_lambda_table_[epoch])

    def compute_loss(self, predicted_matter, predicted_stress_pe, labeled_matter, labeled_stress_pe):
        matter_loss = self.loss_function_(predicted_matter, labeled_matter)
        stress_pe_loss = self.loss_function_(predicted_stress_pe, labeled_stress_pe)
        return matter_loss, stress_pe_loss

    def run_epoch(self, ith_epoch, test_fg = False):
        if test_fg: loader = self.test_loader_
        else: loader = self.train_loader_
        loss_avg = torch.tensor(0.)

        total_n_frames = 0
        # self.optimization.zero_grad()
        
        for training_index, training_data_dir in enumerate(self.training_data_dir_list_):
            number_of_samples = self.dataset_number_of_samples_[training_index]
            system_dim = self.dataset_system_dim_[training_index]
            system_dim3 = system_dim[0]*system_dim[1]*system_dim[2]
            total_n_frames += 0
            for matter_sequence_data, stress_pe_sequence_data, matter_sum_data, stress_pe_sum_data in loader[training_index]:
                
                if not test_fg: self.optimization.zero_grad()
                
                batch_size = len(matter_sequence_data)
                total_n_frames += batch_size

                #reshape input from batch-first to sequence-first
                #flatten batch and system_dim dimensions
                matter_sequence    = self.batched_data_sequence_to_model_input(matter_sequence_data)
                stress_pe_sequence = self.batched_data_sequence_to_model_input(stress_pe_sequence_data)
                
                #split sequence into input (t = 0~sequence_length-1) and output (t = last element)
                matter_sequence_in = matter_sequence[:-1]
                stress_pe_sequence_in = stress_pe_sequence[:-1]
                labeled_matter = matter_sequence[-1]
                labeled_stress_pe = stress_pe_sequence[-1]

                predicted_matter, predicted_stress_pe = self.lattice_md_(matter_sequence_in, stress_pe_sequence_in, system_dim, matter_normalization = False)
                
                matter_loss, stress_pe_loss = self.compute_loss(predicted_matter.permute(0, 2, 3, 4, 1)*self.dataset_matter_prefactor_[training_index], \
                                                                predicted_stress_pe*self.dataset_stress_pe_prefactor_[training_index], \
                                                                labeled_matter*self.dataset_matter_prefactor_[training_index], \
                                                                labeled_stress_pe*self.dataset_stress_pe_prefactor_[training_index])
                
                predicted_matter_sum = predicted_matter.view(batch_size, system_dim3, self.number_of_matters_, self.matter_dim_, self.matter_dim_, self.matter_dim_).sum(dim = (1, 3, 4, 5))
                predicted_stress_pe_sum = predicted_stress_pe.view(batch_size, system_dim3, 7).sum(dim = 1)
                
                matter_sum_loss, stress_pe_sum_loss = self.compute_loss(predicted_matter_sum*self.dataset_matter_sum_prefactor_[training_index], \
                                                                        predicted_stress_pe_sum*self.dataset_stress_pe_sum_prefactor_[training_index], \
                                                                        matter_sum_data*self.dataset_matter_sum_prefactor_[training_index], \
                                                                        stress_pe_sum_data[:, -1, :]*self.dataset_stress_pe_sum_prefactor_[training_index])

                #print("%.4f %.4f %.4f %.4f"%(matter_sum_loss.item(), stress_pe_loss.item(), matter_sum_loss.item(), stress_pe_sum_loss.item()))
                loss = matter_sum_loss + stress_pe_loss + matter_sum_loss + stress_pe_sum_loss
                if not test_fg:
                    loss.backward()
                    self.optimization.step()

                loss_avg += loss.detach().cpu()*batch_size
        
        return loss_avg/total_n_frames

    def train(self):
        sys.stdout = self.logfile
        self.apply_optimize()

        timer = Clock()
        print("")
        print("%5s%12s%12s%12s%12s\n"%("epoch", "l2", "l2_t", "lr", "time"),end="")
        
        timer.get_dt()
        for ith_epoch in range(self.number_of_epochs_):
            loss_train = self.run_epoch(ith_epoch)
            with torch.no_grad():
                loss_test = self.run_epoch(ith_epoch, test_fg = True)

            print("%5d%12.3e%12.3e%12.3e"%(ith_epoch+1, \
                                           loss_train.item(), loss_test.item(), \
                                           self.optimization.param_groups[0]["lr"].item()), end="")
            print("%12.3f"%(timer.get_dt()))
            self.learning_rate_schedule.step()

    def init_test(self):
        self.number_of_epochs_            = None
        self.learning_rate_start_          = None
        self.learning_rate_end_            = None
        print("Starting a NNFF testing on device %s"%(self.device_))
        print("")
        
    def init_training(self):
        self.number_of_epochs_   = int(self.settings_["number_of_epochs"])
        self.learning_rate_start_ = torch.tensor(float(self.settings_["learning_rate_start"]))
        if self.settings_.get("learning_rate_end"): self.learning_rate_end_ = torch.tensor(float(self.settings_["learning_rate_end"]))
        else:                                       self.learning_rate_end_ = torch.tensor(float(self.settings_["learning_rate_start"]))
        self.learning_rate_lambda_table_ = (torch.cat((torch.logspace(torch.log10(self.learning_rate_start_),torch.log10(self.learning_rate_end_),self.number_of_epochs_),torch.tensor([0.])))/self.learning_rate_start_).to(self.device_)
        self.weight_decay_                = 0.0 if self.settings_.get("weight_decay") is None else float(self.settings_["weight_decay"])

        print("Starting a NNFF training on device %s"%(self.device_))
        print("\n\t Number of epochs = %s"%(self.number_of_epochs_))
        print("\t lr = %s ~ %s"%(self.learning_rate_start_.item(),self.learning_rate_end_.item()))
        print("\t random split seed = %s"%(self.random_split_seed_))
        print("\t batch size = %s"%(self.batch_size_))
        print("\t test ratio = %s"%(self.test_ratio_))
        print("\t weight decay = %s"%(self.weight_decay_))
        print("")

    def load_training_data(self):
        try:
            self.dataset_system_dim_ = []
            self.dataset_number_of_samples_ = []
            
            self.dataset_matter_prefactor_ = []
            self.dataset_stress_pe_prefactor_ = []
            self.dataset_matter_sum_prefactor_ = []
            self.dataset_stress_pe_sum_prefactor_ = []

            self.train_loader_ = []
            self.test_loader_ = []

            for training_index, training_data_dir in enumerate(self.training_data_dir_list_):
                print("\tReading %s/MDSD.save..."%(training_data_dir))
                fin = open("%s/MDSD.save"%(training_data_dir),"rb")
                self.sd_ = dill.load(fin)
                fin.close()
                number_of_matters_tmp, matter_dim_tmp, system_dim_tmp, sequence_length_tmp, \
                matter_sequence_data_tmp, stress_sequence_data_tmp, pe_sequence_data_tmp, \
                matter_sum_data_tmp,      stress_sum_data_tmp,      pe_sum_data_tmp = self.sd_.get_data()

                matter_prefactor,     stress_prefactor,     pe_prefactor,\
                matter_sum_prefactor, stress_sum_prefactor, pe_sum_prefactor = self.sd_.get_prefactor()

                if training_index > 0:
                    if self.number_of_matters_ != number_of_matters_tmp: raise NumberOfMattersException
                    if self.matter_dim_        != matter_dim_tmp:        raise MatterDimException
                    if self.sequence_length_   != sequence_length_tmp:   raise SequenceLengthException
                else:
                    self.number_of_matters_ = number_of_matters_tmp
                    self.matter_dim_        = matter_dim_tmp
                    self.sequence_length_   = sequence_length_tmp

                stress_pe_sequence_data_tmp = torch.cat((stress_sequence_data_tmp, pe_sequence_data_tmp), dim = -1)
                stress_pe_sum_data_tmp = torch.cat((stress_sum_data_tmp, pe_sum_data_tmp), dim = -1)
                
                self.dataset_matter_prefactor_.append(matter_prefactor.to(self.device_))
                self.dataset_stress_pe_prefactor_.append(torch.cat((stress_prefactor,pe_prefactor)).to(self.device_))
                self.dataset_matter_sum_prefactor_.append(matter_sum_prefactor.to(self.device_))
                self.dataset_stress_pe_sum_prefactor_.append(torch.cat((stress_sum_prefactor,pe_sum_prefactor)).to(self.device_))

                self.dataset_number_of_samples_.append(len(matter_sequence_data_tmp))
                self.dataset_system_dim_.append(system_dim_tmp)
                dataset = LatticeMDDataset(matter_sequence_data_tmp, stress_pe_sequence_data_tmp, matter_sum_data_tmp, stress_pe_sum_data_tmp, device = self.device_)
                
                total_n_test_sample        = int(self.test_ratio_*self.dataset_number_of_samples_[-1])
                total_n_train_sample       = self.dataset_number_of_samples_[-1] - total_n_test_sample
                print("\t\tNumber of training samples = %s"%(total_n_train_sample))
                print("\t\tNumber of testing samples  = %s"%(total_n_test_sample))
                train_dataset, test_dataset = torch.utils.data.random_split(dataset, [total_n_train_sample, total_n_test_sample], generator=torch.Generator().manual_seed(self.random_split_seed_))
                self.train_loader_.append(torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size_,shuffle=True))
                self.test_loader_.append(torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_,shuffle=False))

                print("\t\tMatter prefactor        = %s"%(matter_prefactor.numpy()))
                print("\t\tStress prefactor        = %s"%(stress_prefactor.numpy()))
                print("\t\tPe     prefactor        = %s"%(pe_prefactor.numpy()))
                print("\t\tMatter sum prefactor    = %s"%(matter_sum_prefactor.numpy()))
                print("\t\tStress sum prefactor    = %s"%(stress_sum_prefactor.numpy()))
                print("\t\tPe     sum prefactor    = %s"%(pe_sum_prefactor.numpy()))

        except NumberOfMattersException:
            print("Number of matters should be the same for all training data.")
            exit()
        except MatterDimException:
            print("Matter Image Dimension should be the same for all training data.")
            exit()
        except SequenceLengthException:
            print("The length of sequences should be the same for all training data.")
            exit()

    def read_input_file(self):
        self.settings_ = {}
        with open(self.input_filename_, "r") as fin:
            for aline in fin:
                if "#" in aline: aline = aline[:aline.index("#")]
                if aline.strip() == "": continue
                linelist = aline.strip().split("=")
                if len(linelist) > 1: self.settings_[linelist[0].strip()] = linelist[1].strip()
