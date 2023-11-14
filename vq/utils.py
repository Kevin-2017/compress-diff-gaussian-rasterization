import os, math, torch
import numpy as np
from plyfile import PlyData, PlyElement


def read_ply_data(input_file):
    
        ply_data = PlyData.read(input_file)
        i = 0
        vertex = ply_data['vertex']
        for prop in vertex._property_lookup:       
            tmp = vertex.data[prop].reshape(-1,1)
            if i == 0:
                data = tmp
                i += 1
            else:
                data = np.concatenate((data, tmp), axis=1)
            
            # print(prop, vertex.data[prop].shape, vertex.data[prop], '\n')
        return data


def write_ply_data(feats, save_ply_path, sh_dim):
    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(sh_dim-3):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l
    
    path= save_ply_path+'/dequantized.ply'
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]      # f4单精度float32，f2
    elements = np.empty(feats.shape[0], dtype=dtype_full)
    # attributes = np.concatenate((xyz, normals, fdcs, frest, opacities, scales, rots), axis=1)#.astype(np.float16)
    # print(attributes.dtype)
    # elements[:] = list(map(tuple, attributes))
    elements[:] = list(map(tuple, feats))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)



def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)



def load_vqdvgo(path, device='cuda'):
    def load_f(name, allow_pickle=False,array_name='arr_0'):
        return np.load(os.path.join(path,name),allow_pickle=allow_pickle)[array_name]

    metadata = load_f('metadata.npz',allow_pickle=True,array_name='metadata')
    metadata = metadata.item()

    ## load basic info
    codebook_size = metadata['codebook_size']   
    codebook_dim = metadata['codebook_dim']   
    bit_length = int(math.log2(codebook_size))                              # log_2_K 
    input_pc_num = metadata['input_pc_num']                                       # 特征第0维，的number
    input_pc_dim  = metadata['input_pc_dim']                                         # 特征第1维，的number  

    # ===================================================== load vq_SH ============================================
    ## loading the two masks  
    non_vq_mask = load_f('non_vq_mask.npz')
    non_vq_mask = np.unpackbits(non_vq_mask)
    non_vq_mask = non_vq_mask[:input_pc_num] 
    non_vq_mask = torch.from_numpy(non_vq_mask).bool().to(device)               # keep_mask表示，不需要vq的feats索引
    all_one_mask = torch.ones_like(non_vq_mask).bool().to(device)             # all_one_mask表示全1的数组，没啥意义，只是用来求vq_mask的

    ## loading codebook and vq indexes
    codebook = load_f('codebook.npz')
    codebook = torch.from_numpy(codebook).float().to(device)
    vq_mask = torch.logical_xor(non_vq_mask, all_one_mask)                    # vq_mask表示，需要vq的feats索引，对应的bool_mask
    vq_elements = vq_mask.sum()

    vq_indexs = load_f('vq_indexs.npz')
    vq_indexs = np.unpackbits(vq_indexs)
    vq_indexs = vq_indexs[:vq_elements*bit_length].reshape(vq_elements,bit_length)
    vq_indexs = torch.from_numpy(vq_indexs).float()
    vq_indexs = bin2dec(vq_indexs, bits=bit_length)
    vq_indexs = vq_indexs.long().to(device)                                 # vq_indexs表示，需要vq的feats索引

    # ===================================================== load non_vq_SH ==========================================
    non_vq_feats = load_f('non_vq_feats.npz')
    non_vq_feats = torch.from_numpy(non_vq_feats).float().to(device)

    # =========================================== load xyz & other attr(opacity + 3*scale + 4*rot)  ===============
    other_attribute = load_f('other_attribute.npz')
    other_attribute = torch.from_numpy(other_attribute).float().to(device)

    xyz = load_f('xyz.npz')
    xyz = torch.from_numpy(xyz).float().to(device)



    # =========================================== build full features  =============================================
    full_feats = torch.zeros(input_pc_num, input_pc_dim).to(device)
    # --- xyz & other attr---
    full_feats[:, 0:3] = xyz
    full_feats[:, -8:] = other_attribute

    # --- nx==ny==nz==0

    # --- vq_SH ---
    full_feats[vq_mask, 6:6+codebook_dim] = codebook[vq_indexs]

    # --- non_vq_SH ---
    # non_vq_mask = torch.logical_xor(vq_mask, all_one_mask)   
    full_feats[non_vq_mask, 6:6+codebook_dim] = non_vq_feats




    # full_density = torch.zeros(input_pc_num, 1).to(device) #- 99999
    # full_density[non_prune_mask,:] = true_density

    # mdoel_state_dict =  metadata['model_state_dict']
    # rgbnet_npz = load_f('rgbnet.npz',allow_pickle=True)
    # for k,v in rgbnet_npz.item().items():
    #     mdoel_state_dict['rgbnet.'+k] =v.to(device)
    # mdoel_state_dict['k0.grid'] = full_grid.T.reshape(1,input_pc_dim,*world_size)
    # mdoel_state_dict['density.grid'] = full_density.reshape(1,1,*world_size)
    # return model_kwargs, mdoel_state_dict

    return full_feats