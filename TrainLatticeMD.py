import torch
import torch.nn as nn
import os
import sys
from LatticeMDData import LatticeMDDataset
from LatticeMDModel import LatticeMD
from OctahedralGroup import OctahedralGroup
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
        print("%s"%(self.settings_["mode"]))
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
        self.test_ratio_                     = 0.1 if self.settings_.get("test_ratio")        is None else float(self.settings_["test_ratio"])
        
        ###################################### Load basic information ######################################

        if self.settings_["mode"]   == "train": self.init_training()
        elif self.settings_["mode"] == "test":  self.init_test()

        ###################################### Create Octahedral Group Symmetry operation ######################################
        print("Creating Octahedral Group Symmetry operation...")
        self.number_of_random_symmetry_group_ = 0 if self.settings_.get("number_of_random_symmetry_group") is None else int(self.settings_["number_of_random_symmetry_group"])
        self.create_octahedral_group_operation()
        ###################################### Create Octahedral Group Symmetry operation ######################################

        ###################################### Load MD sequence data ######################################
        print("Loading MD sequence data...")
        self.training_data_dir_list_ = self.settings_["training_data_dir_list"].strip().split()
        self.include_stress_fg     = False if self.settings_.get("stress") is None else self.settings_.get("stress").lower() == 'true'
        self.include_pe_fg         = False if self.settings_.get("pe")     is None else self.settings_.get("pe").lower() == 'true'
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
        if self.settings_.get("model_dir") is not None:
            print("Path to existing model is specified, loading model...")
            self.model_dir = self.settings_["model_dir"]
            self.load_model()    
        else:
            print("Create new model")
            matter_encoder_conv_ae_kernal_size  = ( 3,  3) if self.settings_.get("matter_encoder_conv_ae_kernal_size")    is None else tuple([int(i) for i in self.settings_.get("matter_encoder_conv_ae_kernal_size").strip().split()])
            matter_encoder_conv_ae_stride       = ( 2,  2) if self.settings_.get("matter_encoder_conv_ae_stride")         is None else tuple([int(i) for i in self.settings_.get("matter_encoder_conv_ae_stride").strip().split()])
            matter_encoder_activation           = 'silu'   if self.settings_.get("matter_encoder_activation")             is None else self.settings_.get("matter_encoder_activation")
            
            matter_decoder_conv_ae_kernal_size  = ( 2,  2) if self.settings_.get("matter_decoder_conv_ae_kernal_size")    is None else tuple([int(i) for i in self.settings_.get("matter_decoder_conv_ae_kernal_size").strip().split()])
            matter_decoder_conv_ae_stride       = ( 1,  1) if self.settings_.get("matter_decoder_conv_ae_stride")         is None else tuple([int(i) for i in self.settings_.get("matter_decoder_conv_ae_stride").strip().split()])
            matter_decoder_activation           = 'silu'   if self.settings_.get("matter_decoder_activation")             is None else self.settings_.get("matter_decoder_activation")

            nonmatter_encoder_hidden_size       = (32, 16) if self.settings_.get("nonmatter_encoder_hidden_size")         is None else tuple([int(i) for i in self.settings_.get("nonmatter_encoder_hidden_size").strip().split()])
            nonmatter_encoder_activation        = 'tanh'   if self.settings_.get("nonmatter_encoder_activation")          is None else self.settings_.get("nonmatter_encoder_activation")
            nonmatter_encoded_size              = 8        if self.settings_.get("nonmatter_encoded_size")                is None else int(self.settings_.get("nonmatter_encoded_size"))
            
            nonmatter_decoder_hidden_size       = (32, 16) if self.settings_.get("nonmatter_decoder_hidden_size")         is None else tuple([int(i) for i in self.settings_.get("nonmatter_decoder_hidden_size").strip().split()])
            nonmatter_decoder_activation        = 'tanh'   if self.settings_.get("nonmatter_decoder_activation")          is None else self.settings_.get("nonmatter_decoder_activation")

            number_of_lstm_layer                = 1        if self.settings_.get("number_of_lstm_layer") is None else int(self.settings_.get("number_of_lstm_layer"))
            number_of_nonmatter_features        = 0
            if self.include_stress_fg:        number_of_nonmatter_features += 6
            if self.include_pe_fg:            number_of_nonmatter_features += 1
            
            self.lattice_md_ = LatticeMD(number_of_matters = self.number_of_matters_, matter_dim = self.matter_dim_, \
                                         matter_encoder_conv_ae_kernal_size = matter_encoder_conv_ae_kernal_size, matter_encoder_conv_ae_stride = matter_encoder_conv_ae_stride, matter_encoder_activation = matter_encoder_activation, \
                                         matter_decoder_conv_ae_kernal_size = matter_decoder_conv_ae_kernal_size, matter_decoder_conv_ae_stride = matter_decoder_conv_ae_stride, matter_decoder_activation = matter_decoder_activation, \
                                         nonmatter_encoder_hidden_size = nonmatter_encoder_hidden_size,           nonmatter_encoder_activation = nonmatter_encoder_activation,   nonmatter_encoded_size = nonmatter_encoded_size, \
                                         nonmatter_decoder_hidden_size = nonmatter_decoder_hidden_size,           nonmatter_decoder_activation = nonmatter_decoder_activation, \
                                         number_of_nonmatter_features = number_of_nonmatter_features,             number_of_lstm_layer = number_of_lstm_layer)
            self.lattice_md_.to(self.device_)

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

    # def compute_loss(self, predicted_matter, predicted_nonmatter, labeled_matter, labeled_nonmatter):
    #     matter_loss = self.loss_function_(predicted_matter, labeled_matter)
    #     nonmatter_loss = self.loss_function_(predicted_nonmatter, labeled_nonmatter)
    #     return matter_loss, nonmatter_loss
    def compute_loss(self, predicted, labeled):
        return self.loss_function_(predicted, labeled)

    def run_epoch(self, ith_epoch, test_fg = False):
        if test_fg: loader = self.test_loader_
        else: loader = self.train_loader_
        loss_avg                    = 0.
        matter_loss_avg             = 0.
        nonmatter_loss_avg          = 0.
        encoder_ae_reconst_loss_avg = 0.

        total_n_frames = 0

        for training_index, training_data_dir in enumerate(self.training_data_dir_list_):
            number_of_samples = self.dataset_number_of_samples_[training_index]
            system_dim = self.dataset_system_dim_[training_index]
            system_dim3 = system_dim[0]*system_dim[1]*system_dim[2]
            total_n_frames = 0
            for matter_sequence_data_untransformed, nonmatter_sequence_data_untransformed, matter_sum_data, nonmatter_sum_data_untransformed in loader[training_index]:
                
                batch_size = len(matter_sequence_data_untransformed)
                total_n_frames += batch_size*self.number_of_octahedral_group_operations_
                random_group_index_list = torch.randint(0, self.number_of_octahedral_group_operations_, (self.number_of_random_symmetry_group_,)).numpy() if self.number_of_random_symmetry_group_ > 0 else [0]

                for group_index in random_group_index_list:

                    matter_sequence_data, nonmatter_sequence_data, nonmatter_sum_data = self.octahedral_group_operation(matter_sequence_data_untransformed, nonmatter_sequence_data_untransformed, nonmatter_sum_data_untransformed, batch_size, system_dim, group_index)
                    
                    #reshape input from batch-first to sequence-first, flatten batch and system_dim dimensions
                    #split sequence into input (t = 0~sequence_length-1) and output (t = last element)
                    matter_sequence    = self.batched_data_sequence_to_model_input(matter_sequence_data)
                    matter_sequence_in = matter_sequence[:-1]
                    if self.lattice_md_.number_of_nonmatter_features_ > 0:
                        nonmatter_sequence = self.batched_data_sequence_to_model_input(nonmatter_sequence_data)
                        nonmatter_sequence_in = nonmatter_sequence[:-1]
                    else:
                        nonmatter_sequence_in = None
                    
                    # test with batch_size = 1
                    # print(nonmatter_sequence[-1].sum(dim = 0)[ -1]/nonmatter_sum_data[0, -1, -1])
                    # print(nonmatter_sequence[-1].sum(dim = 0)[:-1]/nonmatter_sum_data[0, -1, :-1])
                    # print(nonmatter_sum_data[0, -1, :-1])

                    ##############################################################forward pass
                    #predicted_matter_tmp, predicted_nonmatter_tmp = self.lattice_md_(matter_sequence_in, nonmatter_sequence_in, system_dim)
                    matter_sequence_neighbors    = self.lattice_md_.get_neighbor_cell_matter(matter_sequence, system_dim)
                    nonmatter_sequence_neighbors = None
                    if self.lattice_md_.number_of_nonmatter_features_ > 0:
                        nonmatter_sequence_neighbors = self.lattice_md_.get_neighbor_cell_nonmatter(nonmatter_sequence, system_dim)
                    
                    ##############################################################encode
                    L = len(matter_sequence_neighbors)
                    matter_encoded    = self.lattice_md_.matter_encoder_ae_encode(matter_sequence_neighbors.flatten(0, 1))
                    matter_sequence_neighbors_reconst = self.lattice_md_.matter_encoder_ae_decode(matter_encoded).view(L, -1, self.lattice_md_.number_of_matters_, 3*self.lattice_md_.matter_dim_, 3*self.lattice_md_.matter_dim_, 3*self.lattice_md_.matter_dim_)
                    matter_encoded    = matter_encoded.flatten(start_dim = 1)
                    matter_encoded    = matter_encoded.view(L, -1, self.lattice_md_.matter_encoder_encoded_flatten_dim_)
                    
                    lstm_in           = matter_encoded   #(sequence_length-2,  batch_size * system_dim3, matter_encoder_encoded_flatten_dim)
                    if self.lattice_md_.number_of_nonmatter_features_ > 0:
                        nonmatter_encoded = nonmatter_sequence_neighbors.flatten(start_dim = 0, end_dim = 1)                 #(sequence_length-2, * batch_size * system_dim3, 27*number_of_nonmatter_features_)
                        nonmatter_encoded = self.lattice_md_.nonmatter_encoder_.encode(nonmatter_encoded)                           #(sequence_length-2, * batch_size * system_dim3, nonmatter_encoded_dim)
                        nonmatter_encoded = nonmatter_encoded.view(L, -1, self.lattice_md_.nonmatter_encoded_dim_)             #(sequence_length-2,   batch_size * system_dim3, nonmatter_encoded_dim)
                        lstm_in = torch.cat((lstm_in, nonmatter_encoded), dim = -1)                      #(sequence_length-2,  batch_size * system_dim3, lstm_in_dim)
                    ##############################################################encode

                    lstm_in = lstm_in.diff(dim = 0)
                    out, (hn, cn) = self.lattice_md_.advance(lstm_in)
                    lstm_out = hn.sum(dim = 0) #(batch_size * system_dim3, lstm_out_dim)
                    
                    predicted_matter_tmp, predicted_nonmatter_tmp = self.lattice_md_.decode(lstm_out)
                    predicted_matter_tmp = self.lattice_md_.matter_normalize(predicted_matter_tmp, system_dim)

                    # print(predicted_matter_tmp.shape)
                    # print(predicted_nonmatter_tmp)
                    # print(matter_sequence_neighbors.shape)
                    # print(matter_sequence_neighbors_reconst.shape)
                    ##############################################################forward pass

                    #compute matter loss
                    # print((matter_sequence[-1] - matter_sequence_in[-1]).abs().max().item(), predicted_matter_tmp.abs().max().item())
                    predicted_matter        = predicted_matter_tmp# * self.dataset_matter_prefactor_[training_index]
                    labeled_matter          = (matter_sequence[-1] - matter_sequence_in[-1])# * self.dataset_matter_prefactor_[training_index]
                    matter_loss             = self.compute_loss(predicted_matter, labeled_matter)
                    matter_loss_avg        += matter_loss.detach().item()*batch_size

                    #compute nonmatter loss
                    nonmatter_loss = 0.
                    nonmatter_sum_loss = 0.
                    if self.lattice_md_.number_of_nonmatter_features_ > 0:
                        predicted_nonmatter      = predicted_nonmatter_tmp * self.dataset_nonmatter_prefactor_[training_index]                    
                        labeled_nonmatter        = (nonmatter_sequence[-1] - nonmatter_sequence_in[-1]) * self.dataset_nonmatter_prefactor_[training_index]
                        nonmatter_loss           = self.compute_loss(predicted_nonmatter, labeled_nonmatter)
                        nonmatter_loss_avg      += nonmatter_loss.detach().item()*batch_size
                        
                    encoder_ae_reconst_loss = self.compute_loss(matter_sequence_neighbors_reconst, matter_sequence_neighbors)
                    encoder_ae_reconst_loss_avg += encoder_ae_reconst_loss.detach().item()*batch_size
                    # print(matter_sequence_neighbors.shape)
                    
                    loss = matter_loss + nonmatter_loss + nonmatter_sum_loss + encoder_ae_reconst_loss
                    if not test_fg:
                        self.optimization.zero_grad()
                        loss.backward()
                        self.optimization.step()
                    loss_avg += loss.detach().item()*batch_size

        loss_avg               /= total_n_frames
        matter_loss_avg        /= total_n_frames
        nonmatter_loss_avg     /= total_n_frames
        encoder_ae_reconst_loss_avg /= total_n_frames

        return loss_avg, matter_loss_avg, nonmatter_loss_avg, encoder_ae_reconst_loss_avg

    def train(self):
        sys.stdout = self.logfile
        self.apply_optimize()

        timer = Clock()
        print("")
        print("%5s%24s%24s%24s | %10s%10s\n"%("epoch", "mat2", "non-mat2", "encoder_reconst", "lr", "time"),end="")

        self.lattice_md_.unfreeze_model()
        timer.get_dt()
        for ith_epoch in range(self.number_of_epochs_):
            loss_train, matter_loss_train, nonmatter_loss_train, encoder_ae_reconst_loss_train = self.run_epoch(ith_epoch)
            with torch.no_grad():
                loss_test, matter_loss_test, nonmatter_loss_test, encoder_ae_reconst_loss_val = self.run_epoch(ith_epoch, test_fg = True)

            print("%5d%12.3e%12.3e%12.3e%12.3e%12.3e%12.3e | %10.3e"%(ith_epoch+1, \
                                           matter_loss_train       , matter_loss_test, \
                                           nonmatter_loss_train    , nonmatter_loss_test, \
                                           encoder_ae_reconst_loss_train, encoder_ae_reconst_loss_val, \
                                           self.optimization.param_groups[0]["lr"].item()), end="")
            print("%10.2f"%(timer.get_dt()))
            self.learning_rate_schedule.step()

        print("")
        self.save_model()
        sys.stdout = self.original_stdout

    def init_test(self):
        self.number_of_epochs_    = None
        self.learning_rate_start_ = None
        self.learning_rate_end_   = None
        print("Starting a NNFF testing on device %s"%(self.device_))
        print("")
        
    def init_training(self):
        self.number_of_epochs_    = int(self.settings_["number_of_epochs"])
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

    def create_octahedral_group_operation(self):
        #Create a list of, permutations, permutaion of dims, negations, and flipping of dims for each quantity for Octahedral Group Symmetry
        system_dim_offset = 2
        matter_dim_offset = 6
        batch_and_sequence_dim = [0, 1]
        number_of_matters_dim  = [5]
        stress_dim             = [5]
        og = OctahedralGroup(self.device_)
        og.print_octahedral_group()
        print("")
        self.number_of_octahedral_group_operations_ = og.size
        self.matter_sequence_octahedral_group_permutation_ = []
        self.matter_sequence_octahedral_group_flip_ = []
        self.nonmatter_sequence_system_dim_octahedral_group_permutation_ = []
        self.nonmatter_sequence_stress_dim_octahedral_group_permutation_ = []
        self.nonmatter_sequence_octahedral_group_negation_ = []
        for group_index in range(self.number_of_octahedral_group_operations_):
            matter_permute_dim_tmp        =  og.octahedral_group_permutation[group_index].cpu().numpy()
            matter_flip_dim_tmp           = (og.octahedral_group_negation[group_index]<0).nonzero().squeeze(1).cpu().numpy()
            self.matter_sequence_octahedral_group_permutation_.append(tuple(batch_and_sequence_dim + list(system_dim_offset + matter_permute_dim_tmp) + number_of_matters_dim + list(matter_dim_offset + matter_permute_dim_tmp)))
            self.matter_sequence_octahedral_group_flip_       .append(tuple(list(system_dim_offset + matter_flip_dim_tmp)+list(matter_dim_offset + matter_flip_dim_tmp)))

            normal_stress_permute_tmp  = og.octahedral_group_permutation[group_index]
            shear_stress_permute_tmp   = og.shear_stress_octahedral_group_permutation[group_index] + 3
            pe_permute_tmp = torch.tensor([6]).to(self.device_)
            normal_stress_negation_tmp = torch.ones(3).to(self.device_)
            shear_stress_negation_tmp  = og.shear_stress_octahedral_group_negation[group_index]
            pe_negation_tmp = torch.ones(1).to(self.device_)
            self.nonmatter_sequence_system_dim_octahedral_group_permutation_.append(tuple(batch_and_sequence_dim + list(system_dim_offset + matter_permute_dim_tmp) + stress_dim))
            self.nonmatter_sequence_stress_dim_octahedral_group_permutation_.append(torch.cat((normal_stress_permute_tmp,  shear_stress_permute_tmp,  pe_permute_tmp)))
            self.nonmatter_sequence_octahedral_group_negation_              .append(torch.cat((normal_stress_negation_tmp, shear_stress_negation_tmp, pe_negation_tmp)))

    def octahedral_group_operation(self, matter_sequence_data_untransformed, nonmatter_sequence_data_untransformed, nonmatter_sum_data_untransformed, batch_size, system_dim, group_index):
        #matter_sequence_data:        (batch_size, sequence_length+1, system_dim3, number_of_matters, matter_dim, matter_dim, matter_dim)
        #nonmatter_sequence_data:     (batch_size, sequence_length+1, system_dim3, number_of_nonmatter_features)
        #nonmatter_sum_data:          (batch_size, sequence_length+1, number_of_nonmatter_features)
        matter_sequence_data_transformed    = matter_sequence_data_untransformed   .view(batch_size, self.sequence_length_+1, *system_dim, self.number_of_matters_, self.matter_dim_, self.matter_dim_, self.matter_dim_)
        matter_sequence_data_transformed    = matter_sequence_data_transformed   .permute(dims = self.matter_sequence_octahedral_group_permutation_[group_index]).flip(dims = self.matter_sequence_octahedral_group_flip_[group_index])
        matter_sequence_data_transformed    = matter_sequence_data_transformed.flatten(start_dim = 2, end_dim = 4)
        
        if self.lattice_md_.number_of_nonmatter_features_ > 0:
            nonmatter_sequence_data_transformed = nonmatter_sequence_data_untransformed.view(batch_size, self.sequence_length_+1, *system_dim, nonmatter_sequence_data_untransformed.shape[-1])
            nonmatter_sequence_data_transformed = nonmatter_sequence_data_transformed.permute(dims = self.nonmatter_sequence_system_dim_octahedral_group_permutation_[group_index])[...,self.nonmatter_sequence_stress_dim_octahedral_group_permutation_[group_index]]*self.nonmatter_sequence_octahedral_group_negation_[group_index]
            nonmatter_sum_data_transformed = nonmatter_sum_data_untransformed[...,self.nonmatter_sequence_stress_dim_octahedral_group_permutation_[group_index]]*self.nonmatter_sequence_octahedral_group_negation_[group_index]
            nonmatter_sequence_data_transformed = nonmatter_sequence_data_transformed.flatten(start_dim = 2, end_dim = 4)
        else:
            nonmatter_sequence_data_transformed = None
            nonmatter_sum_data_transformed      = None

        return matter_sequence_data_transformed, nonmatter_sequence_data_transformed, nonmatter_sum_data_transformed

    def load_training_data(self):
        try:
            self.dataset_system_dim_ = []
            self.dataset_number_of_samples_ = []
            
            self.dataset_matter_prefactor_ = []
            self.dataset_nonmatter_prefactor_ = []
            self.dataset_matter_sum_prefactor_ = []
            self.dataset_nonmatter_sum_prefactor_ = []

            self.train_loader_ = []
            self.test_loader_ = []

            for training_index, training_data_dir in enumerate(self.training_data_dir_list_):
                print("\tReading %s..."%(training_data_dir))
                fin = open("%s"%(training_data_dir),"rb")
                self.sd_ = dill.load(fin)
                fin.close()
                number_of_matters, matter_dim, system_dim, sequence_length, \
                matter_sequence_data, momentum_sequence_data, stress_sequence_data, pe_sequence_data, \
                matter_sum_data,                                  stress_sum_data,  pe_sum_data = self.sd_.get_data()

                matter_prefactor,     momentum_prefactor, stress_prefactor,     pe_prefactor,\
                matter_sum_prefactor,                     stress_sum_prefactor, pe_sum_prefactor = self.sd_.get_prefactor()

                if training_index > 0:
                    if self.number_of_matters_ != number_of_matters: raise NumberOfMattersException
                    if self.matter_dim_        != matter_dim:        raise MatterDimException
                    if self.sequence_length_   != sequence_length:   raise SequenceLengthException
                else:
                    self.number_of_matters_ = number_of_matters
                    self.matter_dim_        = matter_dim
                    self.sequence_length_   = sequence_length

                nonmatter_sequence_data = None
                nonmatter_sum_data      = None
                if self.include_stress_fg:
                    nonmatter_sequence_data = stress_sequence_data
                    nonmatter_sum_data      = stress_sum_data
                if self.include_pe_fg:
                    if nonmatter_sequence_data is not None:
                        nonmatter_sequence_data = torch.cat((nonmatter_sequence_data, pe_sequence_data), dim = -1)
                        nonmatter_sum_data = torch.cat((nonmatter_sum_data, pe_sum_data), dim = -1)
                    else:
                        nonmatter_sequence_data = pe_sequence_data
                        nonmatter_sum_data      = pe_sum_data
                if nonmatter_sum_data is None:
                    nonmatter_sequence_data = torch.zeros(len(pe_sequence_data), 1)
                    nonmatter_sum_data = torch.zeros(len(pe_sequence_data), 1)

                self.dataset_matter_prefactor_.append(matter_prefactor.to(self.device_))
                stress_prefactor_avg = torch.cat((torch.ones(3)*stress_prefactor[:3].mean(), torch.ones(3)*stress_prefactor[3:].mean()))
                self.dataset_nonmatter_prefactor_.append(torch.cat((stress_prefactor_avg, pe_prefactor)).to(self.device_))
                self.dataset_matter_sum_prefactor_.append(matter_sum_prefactor.to(self.device_))
                stress_sum_prefactor_avg = torch.cat((torch.ones(3)*stress_sum_prefactor[:3].mean(), torch.ones(3)*stress_sum_prefactor[3:].mean()))
                self.dataset_nonmatter_sum_prefactor_.append(torch.cat((stress_sum_prefactor_avg, pe_sum_prefactor)).to(self.device_))

                self.dataset_number_of_samples_.append(len(matter_sequence_data))
                self.dataset_system_dim_.append(system_dim)
                dataset = LatticeMDDataset(matter_sequence_data, nonmatter_sequence_data, matter_sum_data, nonmatter_sum_data, device = self.device_)
                
                total_n_test_sample        = int(self.test_ratio_*self.dataset_number_of_samples_[-1])
                total_n_train_sample       = self.dataset_number_of_samples_[-1] - total_n_test_sample
                print("\t\tNumber of training samples = %s"%(total_n_train_sample))
                print("\t\tNumber of testing samples  = %s"%(total_n_test_sample))
                train_dataset, test_dataset = torch.utils.data.random_split(dataset, [total_n_train_sample, total_n_test_sample], generator=torch.Generator().manual_seed(self.random_split_seed_))
                self.train_loader_.append(torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size_,shuffle=True))
                self.test_loader_.append(torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_,shuffle=False))

                print("\t\tMatter prefactor        = %s"%(matter_prefactor.numpy()))
                if self.include_stress_fg:
                    print("\t\tStress prefactor        = %s"%(stress_prefactor_avg.numpy()))
                if self.include_pe_fg:
                    print("\t\tPe     prefactor        = %s"%(pe_prefactor.numpy()))
                print("\t\tMatter sum prefactor    = %s"%(matter_sum_prefactor.numpy()))
                if self.include_stress_fg:
                    print("\t\tStress sum prefactor    = %s"%(stress_sum_prefactor_avg.numpy()))
                if self.include_pe_fg:
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

    def save_model(self):
        model_filename = "save.lmd"
        print("")
        print("Saving model to %s..."%(model_filename))
        self.lattice_md_.to('cpu')
        self.lattice_md_.freeze_model()
        fout = open(self.output_dir+"/"+model_filename,"wb")
        dill.dump(self.lattice_md_, fout)
        fout.close()
        self.lattice_md_.to(self.device_)
        self.lattice_md_.unfreeze_model()
    
    def load_model(self):
        model_filename = "save.lmd"
        print("")
        print("Loading model from %s..."%(model_filename))
        fin = open(model_filename,"rb")
        self.lattice_md_ = dill.load(fin, fout)
        fin.close()
        self.lattice_md_.to(self.device_)
    
        