import os, torch, argparse
import numpy as np
from tqdm import tqdm, trange

import math
from copy import deepcopy

from vq import VectorQuantize
from utils import read_ply_data, write_ply_data, load_vqdvgo


def parse_args():
    parser = argparse.ArgumentParser(description="codebook based quantization")
    # parser.add_argument("--important_score_npz_path", type=str, default='../data/distill')
    # parser.add_argument("--input_path", type=str, default='../data/distill/iteration_55000/point_cloud.ply')
    parser.add_argument("--important_score_npz_path", type=str, default='room')
    parser.add_argument("--input_path", type=str, default='room/iteration_55000/point_cloud.ply')
    
    parser.add_argument("--save_path", type=str, default='../codebook/logs/room-0.6')    
    parser.add_argument("--ply_path", type=str, default='../codebook/logs/room-0.6')    
    # parser.add_argument("--save_path", type=str, default='../codebook/logs/kit-48000-13-1000-vq70-SH')    
    # parser.add_argument("--ply_path", type=str, default='../codebook/logs/kit-48000-13-1000-vq70-SH')    
    # parser.add_argument("--save_path", type=str, default='../codebook/logs/room_vq80%')
    # parser.add_argument("--ply_path", type=str, default='../codebook/logs/room_vq80%')    
    parser.add_argument("--vq_ratio", type=float, default=0.6)
    parser.add_argument("--wage_vq_ratio", type=int, default=2**16-1)
    parser.add_argument("--iteration_num", type=float, default=1000)

    parser.add_argument("--codebook_size", type=int, default=2**13)
    parser.add_argument("--importance_prune", type=float, default=1.0)
    parser.add_argument("--importance_include", type=float, default=0.0)
    parser.add_argument("--no_IS", type=bool, default=False)
    parser.add_argument("--no_load_data", type=bool, default=False)
    parser.add_argument("--no_save_ply", type=bool, default=False)

    parser.add_argument("--sh_degree", type=int, default=2)
    parser.add_argument("--vq_way", type=str, default='none') # wage

    opt = parser.parse_args() 
    return opt
    

class Quantization():
    def __init__(self, opt):
        
        # ----- load ply data -----
        if opt.sh_degree == 3:
            self.sh_dim = 3+45
        elif opt.sh_degree == 2:
            self.sh_dim = 3+24
        

        self.feats = read_ply_data(opt.input_path)
        self.feats = torch.tensor(self.feats)
        self.feats_bak = self.feats.clone()
        self.feats_bak2 = self.feats.clone()

        self.feats = self.feats[:, 6:6+self.sh_dim]

        # ----- define model -----
        self.model_vq = VectorQuantize(
                    dim = self.feats.shape[1],              # 特征维度
                    codebook_size = opt.codebook_size,
                    decay = 0.8,                            # specify number of quantizersse， 对应公式(9)的 λ_d
                    commitment_weight = 1.0,                # codebook size
                    use_cosine_sim = False,
                    threshold_ema_dead_code=0,
                ).to(device)
        
        # ----- other -----
        self.save_path = opt.save_path
        self.ply_path = opt.ply_path
        self.imp_path = opt.important_score_npz_path
        self.high = None
        self.VQ_CHUNK = 80000
        self.k_expire = 10        
        self.vq_ratio = opt.vq_ratio
        self.wage_vq_ratio = opt.wage_vq_ratio

        self.no_IS = opt.no_IS
        self.no_load_data = opt.no_load_data
        self.no_save_ply = opt.no_save_ply
   
        self.codebook_size = opt.codebook_size
        self.importance_prune = opt.importance_prune
        self.importance_include = opt.importance_include
        self.iteration_num = opt.iteration_num

        self.vq_way = opt.vq_way

        # ----- print info -----
        print("=========================================")
        print("input_feats_shape: ", self.feats_bak.shape)
        print("vq_feats_shape: ", self.feats.shape)
        print("SH_degree: ", opt.sh_degree)
        print("Quantization_ratio: ", opt.vq_ratio)
        print("Add_important_score: ", opt.no_IS==False)
        print("Codebook_size: ", opt.codebook_size)
        print("=========================================")

    @torch.no_grad()
    def calc_vector_quantized_feature(self):
        """
        apply vector quantize on feature grid and return vq indexes
        """
        print("caculate vq features")
        CHUNK = 8192
        feat_list = []
        indice_list = []
        self.model_vq.eval()
        # self.model_vq._codebook.embed.half().float()   # ?
        for i in tqdm(range(0, self.feats.shape[0], CHUNK)):
            feat, indices, commit = self.model_vq(self.feats[i:i+CHUNK,:].unsqueeze(0).to(device))
            indice_list.append(indices[0])
            feat_list.append(feat[0])
        self.model_vq.train()
        # all_feat = torch.cat(feat_list).half().float()  # [num_elements, k0_dim]
        all_feat = torch.cat(feat_list)  # [num_elements, k0_dim]
        all_indice = torch.cat(indice_list)             # [num_elements, 1]
        return all_feat, all_indice


    @torch.no_grad()
    def fully_vq_reformat(self):  

        print("start fully vector quantize")
        
        all_feat, all_indice = self.calc_vector_quantized_feature()

        # print("start cdf three split")
        # self.init_cdf_mask(thres_mid=self.importance_include, thres_high=self.importance_prune)

        #=============== 对于important score高的point-cloud进行VQ
        # new_feats = torch.zeros_like(all_feat)
        # new_feats[self.all_one_mask,:] = all_feat[self.all_one_mask,:]
        # non_vq_feats = self.feats                                                        # feats[keep_mask,:]                                  
        # non_vq_feats = torch.quantize_per_tensor(non_vq_feats, scale=non_vq_feats.std()/15, zero_point=torch.round(non_vq_feats.mean()), dtype=torch.qint8)
        # new_feats = non_vq_feats.dequantize()                                      # new_feats[keep_mask,:] =  non_vq_feats.dequantize() 
        
        ##### To ease the implementation of codebook finetuneing, we add indexs of non-vq-voxels to all_indice.
        ##### note that these part of indexs will not be saved
        # all_indice[self.non_vq_mask] = torch.arange(self.non_vq_mask.sum()) + self.codebook_size


        if self.save_path is not None:
            save_path = self.save_path
            
            os.makedirs(f'{save_path}/extreme_saving', exist_ok=True)
            # np.savez_compressed(f'{save_path}/extreme_saving/non_prune_density.npz',non_prune_density.int_repr().cpu().numpy())
            # np.savez_compressed(f'{save_path}/extreme_saving/non_vq_grid.npz', non_vq_grid.int_repr().cpu().numpy())


            
            # ----- save basic info -----
            metadata = dict()
            metadata['input_pc_num'] = self.feats_bak.shape[0]  
            metadata['input_pc_dim'] = self.feats_bak.shape[1]  
            metadata['codebook_size'] = self.codebook_size
            metadata['codebook_dim'] = self.sh_dim
            np.savez_compressed(f'{save_path}/extreme_saving/metadata.npz', metadata=metadata)

            # ===================================================== save vq_SH =============================================
            # ----- save mapping_index (vq_index) -----
            def dec2bin(x, bits):
                mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
                return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()    
            # vq indice was saved in according to the bit length
            self.codebook_vq_index = all_indice[torch.logical_xor(self.all_one_mask,self.non_vq_mask)]                               # vq_index，       其中的每个值都小于codebooksize
            bin_indices = dec2bin(self.codebook_vq_index, int(math.log2(self.codebook_size))).bool().cpu().numpy()                 # mapping_index，  二进制版的vq_index
            np.savez_compressed(f'{save_path}/extreme_saving/vq_indexs.npz',np.packbits(bin_indices.reshape(-1)))               
            
            # ----- save codebook -----

            self.tmp_codebook = self.model_vq._codebook.embed.squeeze(0)
            # codebook = self.model_vq._codebook.embed.cpu().half().numpy()                                                       
            codebook = self.model_vq._codebook.embed.cpu().numpy().squeeze(0)                                                       
            np.savez_compressed(f'{save_path}/extreme_saving/codebook.npz', codebook)

            

            # ----- save keep mask (non_vq_feats_index)-----
            np.savez_compressed(f'{save_path}/extreme_saving/non_vq_mask.npz',np.packbits(self.non_vq_mask.reshape(-1).cpu().numpy()))

            # ===================================================== save non_vq_SH ============================================
            # non_vq_feats = self.feats_bak[self.non_vq_index, 6:6+self.sh_dim]     # bad
            non_vq_feats = self.feats_bak[self.non_vq_mask, 6:6+self.sh_dim]        # good
            wage_non_vq_feats = self.wage_vq(non_vq_feats)
            np.savez_compressed(f'{save_path}/extreme_saving/non_vq_feats.npz', wage_non_vq_feats) 

            
            
            # =========================================== save xyz &f other attr(opacity + 3*scale + 4*rot) ====================================
            other_attribute = self.feats_bak[:, -8:]
            wage_other_attribute = self.wage_vq(other_attribute)
            np.savez_compressed(f'{save_path}/extreme_saving/other_attribute.npz', wage_other_attribute)

            xyz = self.feats_bak[:, 0:3]
            np.savez_compressed(f'{save_path}/extreme_saving/xyz.npz', xyz)


            # zip everything together to get final size
            os.system(f"zip -r {save_path}/extreme_saving.zip {save_path}/extreme_saving")

            size = os.path.getsize(f'{save_path}/extreme_saving.zip')
            size_MB = size / 1024.0 / 1024.0
        print("size = ", size_MB, " MB")
        print("finish fully vector quantize")
        return all_feat, all_indice
    
    def load_f(self, path, name, allow_pickle=False,array_name='arr_0'):
        return np.load(os.path.join(path, name),allow_pickle=allow_pickle)[array_name]

    def wage_vq(self, feats):

        # half
        if self.vq_way == 'half':
            return feats.half()
        
        # Wage
        elif self.vq_way == 'wage':
            quantization = self.wage_vq_ratio
            for i in range(feats.shape[0]):
                max = torch.max(feats[i], dim=0).values
                min = torch.min(feats[i], dim=0).values
                normValue = torch.max(torch.abs(max), torch.abs(min))        # WAGE -- max abs
                # normValue = max - min     # WAGE -- substract abs
                feats[i] = torch.round(feats[i] * quantization / normValue) / quantization * normValue
            return feats.half()
        
        else:
            return feats





    #     quantDict = {}
    #     for prop in ply_data['vertex']._property_lookup:
    #         quantDict[prop] = {}
    #         quantDict[prop]['max'] = np.max(ply_data['vertex'].data[prop])
    #         quantDict[prop]['min'] = np.min(ply_data['vertex'].data[prop])
    #     print(quantDict)

    #     vertex = ply_data['vertex']
    #     for prop in vertex._property_lookup:
    #         if prop == 'nx' or prop == 'ny' or prop == 'nz':
    #             continue
    #         # normValue = max(np.abs(quantDict[prop]['max']), np.abs(quantDict[prop]['min']))          # WAGE -- max abs
    #         normValue = quantDict[prop]['max'] - quantDict[prop]['min']                                # WAGE -- substract

    #         print(prop,
    #             vertex.data[prop].shape, '\n',
    #             vertex.data[prop], '\n',
    #             vertex.data[prop] * quantization / normValue, '\n',
    #             np.round(vertex.data[prop] * quantization / normValue), '\n',
    #             quantization * normValue)

    #         vertex.data[prop] = np.round(vertex.data[prop] * quantization / normValue) / quantization * normValue
    #         print(vertex.data[prop], '\n')
    
    def quantize(self):
        if self.no_IS:                                                      #  no important score
            importance = np.ones((self.feats.shape[0]))                     #  全都置为1，表示每个点云的important-score相同
        else:
            importance = self.load_f(self.imp_path, 'imp_score.npz')

        ###################################################
        only_vq_some_vector = True
        if only_vq_some_vector:
            tensor_importance = torch.tensor(importance)
            large_val, large_index = torch.topk(tensor_importance, k=int(tensor_importance.shape[0] * (1-self.vq_ratio)), largest=True)   # large_index 表示IS分高的index，里面是具体的index数字
            self.all_one_mask = torch.ones_like(tensor_importance).bool()     
            self.non_vq_mask = torch.zeros_like(tensor_importance).bool()         
            self.non_vq_mask[large_index] = True                                                                                 # self.non_vq_mask表示IS分高的index，里面是bool值
        self.non_vq_index = large_index

        # 计算重要性分数前X%的点，占总分数的百分之多少
        IS_non_vq_point = large_val.sum()
        IS_all_point = tensor_importance.sum()
        IS_percent = IS_non_vq_point/IS_all_point
        print("IS_percent: ", IS_percent)


        #=================== Codebook initialization ====================

        self.model_vq.train()
        with torch.no_grad():
            self.vq_mask = torch.logical_xor(self.all_one_mask, self.non_vq_mask)                  # 需要进行vq的feats，的索引 
            feats_needs_vq = self.feats[self.vq_mask].clone()                                       # 需要进行vq的feats
            imp = tensor_importance[self.vq_mask].float()                                        # 需要进行vq的feats，对应的important-score 
            k = self.k_expire                                                               # 每次更新codebook中，importance最低的k个code 
            if k > self.model_vq.codebook_size:
                k = 0            
            for i in trange(self.iteration_num):
                indexes = torch.randint(low=0, high=feats_needs_vq.shape[0], size=[self.VQ_CHUNK])         # 随机生成self.VQ_CHUNK个不同的整数，即索引
                vq_weight = imp[indexes].to(device)
                vq_feature = feats_needs_vq[indexes,:].to(device)
                quantize, embed, loss = self.model_vq(vq_feature.unsqueeze(0), weight=vq_weight.reshape(1,-1,1))

                replace_val, replace_index = torch.topk(self.model_vq._codebook.cluster_size, k=k, largest=False)      # 找出k个importance最低的vector，return index，准备替换
                _, most_important_index = torch.topk(vq_weight, k=k, largest=True)
                self.model_vq._codebook.embed[:,replace_index,:] = vq_feature[most_important_index,:]

        #=================== Apply voxel pruning and vector quantization ====================
        all_feat, all_indices = self.fully_vq_reformat()
        print('\n')
        print('\n')
        print('\n')
        self.tmp = all_feat #after_codebook 
        print("output_feats: ", all_feat.shape)        
        print("quantized succcessfully!")



    def dequantize(self):

        if self.no_load_data:
            # 111
            # self.feats_bak[:, 6:6+self.sh_dim] = self.tmp        
            # self.feats_bak[self.non_vq_index, 6:6+self.sh_dim] = self.feats_bak2[self.non_vq_index, 6:6+self.sh_dim]
            # dequantized_feats = self.feats_bak

            # 222 -- 原始的codebook实现
            # dequantized_feats=torch.zeros(self.feats_bak.shape[0], self.feats_bak.shape[1]).to(device)
            # dequantized_feats=torch.zeros(self.feats_bak.shape[0], self.feats_bak.shape[1]).to(device)
            # dequantized_feats[:, 0:3] = self.feats_bak[:, 0:3]
            # dequantized_feats[:, -8:] = self.feats_bak[:, -8:]
            # dequantized_feats[self.vq_mask.cpu(), 6:6+self.sh_dim] = self.tmp_codebook[self.vq_index.cpu()]            
            # dequantized_feats[self.non_vq_mask.cpu(), 6:6+self.sh_dim] = self.feats_bak[self.non_vq_index.cpu(), 6:6+self.sh_dim].cuda()

            # 333 -- 证明不是opacity的问题
            # dequantized_feats=torch.zeros(self.non_vq_mask.sum(), self.feats_bak.shape[1]).to(device)
            # dequantized_feats[:, 0:3] = self.feats_bak[self.non_vq_index.cpu(), 0:3]
            # dequantized_feats[:, -8:] = self.feats_bak[self.non_vq_index.cpu(), -8:]
            # dequantized_feats[:, 6:6+self.sh_dim] = self.feats_bak[self.non_vq_index.cpu(), 6:6+self.sh_dim].cuda()

            # 444 -- 如果render质量好，证明不是index和mask的问题
            # dequantized_feats=torch.zeros(self.feats_bak.shape[0], self.feats_bak.shape[1]).to(device)
            # dequantized_feats=torch.zeros(self.feats_bak.shape[0], self.feats_bak.shape[1]).to(device)
            # dequantized_feats[:, 0:3] = self.feats_bak[:, 0:3]
            # dequantized_feats[:, -8:] = self.feats_bak[:, -8:]
            # dequantized_feats[self.vq_mask.cpu(), 6:6+self.sh_dim] = self.feats_bak[self.vq_index.cpu(), 6:6+self.sh_dim].cuda()         
            # dequantized_feats[self.non_vq_index.cpu(), 6:6+self.sh_dim] = self.feats_bak[self.non_vq_index.cpu(), 6:6+self.sh_dim].cuda()
            # dequantized_feats[self.non_vq_mask.cpu(), 6:6+self.sh_dim] = self.feats_bak[self.non_vq_index.cpu(), 6:6+self.sh_dim].cuda()


            # 555 -- 相比于4，add了codebook __________________________nice!!!    : 最后一行使用self.non_vq_index.cpu()
            # dequantized_feats=torch.zeros(self.feats_bak.shape[0], self.feats_bak.shape[1]).to(device)
            # dequantized_feats=torch.zeros(self.feats_bak.shape[0], self.feats_bak.shape[1]).to(device)
            # dequantized_feats[:, 0:3] = self.feats_bak[:, 0:3]
            # dequantized_feats[:, -8:] = self.feats_bak[:, -8:]
            # dequantized_feats[self.vq_mask.cpu(), 6:6+self.sh_dim] = self.tmp_codebook[self.vq_index.cpu()]            
            # dequantized_feats[self.non_vq_index.cpu(), 6:6+self.sh_dim] = self.feats_bak[self.non_vq_index.cpu(), 6:6+self.sh_dim].cuda()    
 
            # 666  : 最后一行使用self.non_vq_mask
            dequantized_feats=torch.zeros(self.feats_bak.shape[0], self.feats_bak.shape[1]).to(device)
            dequantized_feats[:, 0:3] = self.feats_bak[:, 0:3]
            dequantized_feats[:, -8:] = self.feats_bak[:, -8:]
            dequantized_feats[self.vq_mask.cpu(), 6:6+self.sh_dim] = self.tmp_codebook[self.codebook_vq_index.cpu()]            
            dequantized_feats[self.non_vq_mask.cpu(), 6:6+self.sh_dim] = self.feats_bak[self.non_vq_mask.cpu(), 6:6+self.sh_dim].cuda()    


            # aaa = torch.logical_xor(self.vq_mask, self.non_vq_mask)

            # a = self.non_vq_mask.sum()
            # print(a)
            # print(a)
            # print(a)

            dequantized_feats_2 = load_vqdvgo(os.path.join(self.save_path,'extreme_saving'), device=device)       
            a = torch.equal(dequantized_feats, dequantized_feats_2)
            # a = torch.equal(self.non_vq_mask.cuda(), nnnon_vq_mask.cuda())
            # a = torch.equal(dequantized_feats[self.non_vq_mask.cpu(), 6:6+self.sh_dim], bbb)
            print("Equal or not: ", a)
            print("Equal or not: ", a)
            print("Equal or not: ", a)
            print("Equal or not: ", a)
            print("Equal or not: ", a)
            print("Equal or not: ", a)

        else:
            print("Load saved data:")
            dequantized_feats = load_vqdvgo(os.path.join(self.save_path,'extreme_saving'), device=device)

        if self.no_save_ply == False:
            os.makedirs(f'{self.ply_path}/', exist_ok=True)
            write_ply_data(dequantized_feats.cpu().numpy(), self.ply_path, self.sh_dim)

        print("dequantized_feats: ", dequantized_feats.shape)
        print("dequantized succcessfully!")




if __name__=='__main__':
    opt = parse_args()
    device = torch.device('cuda')
    vq = Quantization(opt)

    vq.quantize()
    vq.dequantize()
    
    print("all done!!!")

