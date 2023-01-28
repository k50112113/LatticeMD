import torch
import numpy as np

class OctahedralGroup:
    #Octohedral Group Symmetry (Rotation/Reflection), self.size=48
    #octahedral_group:               x = 1, y = 2, z = 3
    #shear_stress_octahedral_group:  xy= 1, yz= 2, xz= 3
    #_negation:    if an axis is flipped or not
    #_flip         if an axis is flipped or not
    #_permutation: a permutation of the axes
    #
    #coordinate_transform    : for coordinates for 3-D point cloud data
    #normal_stress_transform : for normal stress of a box (xx, yy, zz)
    #shear_stress_transform  : for shear stress of a box (xy, yz, xz)
    #grid_transform          : for 3-D grid data

    def __init__(self, device = 'cpu'):
        self.octahedral_group = torch.tensor([[ 1, 2, 3],\
                                              [-1, 3, 2],[ 3, -2,1],[ 2, 1,-3],\
                                              [ 1,-2,-3],[-1, 2,-3],[-2,-1, 3],\
                                              [-1,-2,-3]]).to(device)                  
        #Reflection Operation
        self.octahedral_group = torch.vstack((self.octahedral_group, self.octahedral_group[:,torch.LongTensor([0, 2, 1])]))
        #Rotation along (1,1,1)
        self.octahedral_group = torch.vstack((self.octahedral_group, \
                                                            torch.roll(self.octahedral_group, shifts = 1, dims = -1), \
                                                            torch.roll(self.octahedral_group, shifts = 2, dims = -1)))
        
        self.octahedral_group_negation = torch.where(self.octahedral_group > 0, 1., -1.)
        self.octahedral_group_permutation = self.octahedral_group.abs()-1
        self.size = len(self.octahedral_group)
        self.octahedral_group_flip = []
        for i in range(self.size): self.octahedral_group_flip.append(tuple((self.octahedral_group_negation[i]<0).nonzero().squeeze(1).cpu().numpy()))
        
        self.shear_stress_octahedral_group = torch.stack((self.octahedral_group[:,0]*self.octahedral_group[:,1],self.octahedral_group[:,1]*self.octahedral_group[:,2],self.octahedral_group[:,0]*self.octahedral_group[:,2])).transpose(0, 1)
        self.shear_stress_octahedral_group = torch.where(self.shear_stress_octahedral_group==2, 1, self.shear_stress_octahedral_group)
        self.shear_stress_octahedral_group = torch.where(self.shear_stress_octahedral_group==6, 2, self.shear_stress_octahedral_group)
        self.shear_stress_octahedral_group = torch.where(self.shear_stress_octahedral_group==3, 3, self.shear_stress_octahedral_group)
        self.shear_stress_octahedral_group = torch.where(self.shear_stress_octahedral_group==-2, -1, self.shear_stress_octahedral_group)
        self.shear_stress_octahedral_group = torch.where(self.shear_stress_octahedral_group==-6, -2, self.shear_stress_octahedral_group)
        self.shear_stress_octahedral_group = torch.where(self.shear_stress_octahedral_group==-3, -3, self.shear_stress_octahedral_group)
        self.shear_stress_octahedral_group_negation = torch.where(self.shear_stress_octahedral_group>0, 1., -1.)
        self.shear_stress_octahedral_group_permutation = self.shear_stress_octahedral_group.abs()-1

        # pairwise_distance = ((self.octahedral_group.unsqueeze(1)-self.octahedral_group.unsqueeze(0))**2).sum(dim=-1)
        # i,  j  = torch.where(pairwise_distance==0)
        # print("There is no duplicate in the Group:",all(i==j))

        self.normal_stress_str_list = self.create_str_list(self.octahedral_group, {1:"x",2:"y",3:"z"})
        self.shear_stress_str_list  = self.create_str_list(self.shear_stress_octahedral_group,  {1:"xy",2:"yz",3:"xz"})

    def create_str_list(self, g, m):
        str_list = []
        for i in g:
            str_list.append("")
            for j in i:
                pre = "-" if j.item()<0 else ""
                str_list[-1] += "%4s"%(pre+m[abs(j.item())])
        return str_list
    
    def print_octahedral_group(self):
        print("Octohedral Group Symmetry (Rotation/Reflection)")
        for i in range(len(self.normal_stress_str_list)):
            print("%4d | %s | %s"%(i, self.normal_stress_str_list[i], self.shear_stress_str_list[i]))
    
    def coordinate_transform(self, m, index = -1):
        #index  > -1: shape (Nx3) ->   (Nx3)
        #index == -1: shape (Nx3) -> (GxNx3)
        if index > -1: return  m[:,self.octahedral_group_permutation[index]]*self.octahedral_group_negation[index]
        else:          return (m[:,self.octahedral_group_permutation       ]*self.octahedral_group_negation       ).transpose(0, 1)

    def normal_stress_transform(self, m, index = -1):
        #index  > -1: shape (3) ->   (3)
        #index == -1: shape (3) -> (Gx3)
        if index > -1: return m[self.octahedral_group_permutation[index]]
        else:          return m[self.octahedral_group_permutation       ]
    
    def shear_stress_transform(self, m, index = -1):
        #index  > -1: shape (3) ->   (3)
        #index == -1: shape (3) -> (Gx3)
        if index > -1: return m[self.shear_stress_octahedral_group_permutation[index]]*self.shear_stress_octahedral_group_negation[index]
        else:          return m[self.shear_stress_octahedral_group_permutation       ]*self.shear_stress_octahedral_group_negation

    def grid_transform(self, m, index):
        #Currently, grid transform does not support batched operation
        #shape (NxNxN) -> (NxNxN)
        return m.permute(dims = tuple(self.octahedral_group_permutation[index].numpy())).flip(dims = self.octahedral_group_flip[index])
