import torch
from copy import deepcopy
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sr_flag(epoch,sr):
     return sr

def scale_gammas(sr_flag, alpha, model, prune_idx, scale_down=True):
    #'''放缩bn层大小，加快稀疏'''
    if sr_flag:
        alpha_ = 1/alpha

        if not scale_down:
            # after training we want to scale back up so need to invert alpha
            alpha_ = alpha
            alpha = 1/alpha
        nnlist = model.module_list
        for idx in prune_idx:
            nnlist[idx][0].weight.data.mul(alpha_)
            nnlist[idx][1].weight.data.mul(alpha)
        


def updateBN(sr_flag, module_list, s, prune_idx):
    if sr_flag:
        for idx in prune_idx:
            module_list[idx][1].weight.grad.data.add_(s * torch.sign(module_list[idx][1].weight.data)) 


def gather_bn_weights(module_list, prune_idx):
    bn_weights = torch.Tensor().to(device)#'cuda'
    for idx in prune_idx:
        bn_weights = torch.cat([bn_weights, module_list[idx][1].weight.data])
        #print(bn_weights.size())
    return bn_weights


def parse_module_defs(module_defs):
    Conv_idx = []
    CBL_idx = []
    for idx in range(len(module_defs)):
        if module_defs[idx]['type'] == 'convolutional':
            Conv_idx.append(idx)
            if module_defs[idx]['batch_normalize'] == '1':
                CBL_idx.append(idx)
    prune_idx = [0,\
          2,\
          6, 9,\
          13, 16, 19, 22, 25, 28, 31, 34,\
          38, 41, 44, 47, 50, 53, 56, 59,\
          63, 66, 69, 72,\
          75, 76, 77, 78, 79,     80,\
         #84,
          87, 88, 89, 90, 91,     92,\
         #96,
          99,100,101,102,103,    104]#liumm
    return CBL_idx, Conv_idx, prune_idx

def obtain_bn_mask(bn_module, thre):
    weight_copy = bn_module.weight.data.abs().clone()
    mask = weight_copy.gt(thre).float().cuda()
    if int(torch.sum(mask)) == 0:
        mask[int(torch.argmax(weight_copy))] = 1.0
        print("mask is  0")
    return mask

def prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask):
    pruned_model = deepcopy(model)
    for idx in CBL_idx:
        #print('idx=', idx)
        bn_module = pruned_model.module_list[idx][1]
        if idx in prune_idx:
            mask = torch.from_numpy(CBLidx2mask[idx]).to(device)
            bn_module.weight.data.mul_(mask) # mask mul BN weight
            nonmask = torch.ones_like(mask)-mask
            nonmask = nonmask*bn_module.bias.data
            if idx+1 in CBL_idx:
                act = torch.mm(F.relu(nonmask).view(1,-1),pruned_model.module_list[idx+1][0].weight.data.sum(dim=[2,3]).transpose(1,0).contiguous())
                next_mean = pruned_model.module_list[idx+1][1].running_mean - act
                next_mean = next_mean.view(-1)
                pruned_model.module_list[idx+1][1].running_mean = next_mean
            else:
                #print(pruned_model.module_list[idx+1][0].bias.size())
                act = torch.mm(F.relu(nonmask).view(1,-1),pruned_model.module_list[idx+1][0].weight.data.sum(dim=[2,3]).transpose(1,0).contiguous())
                next_bias = pruned_model.module_list[idx+1][0].bias.data + act
                next_bias = next_bias.view(-1)
                pruned_model.module_list[idx+1][0].bias.data = next_bias
    return pruned_model

def init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask):
    for idx in Conv_idx:
        #print('idx=',idx)
        module0 = pruned_model.module_list[idx]
        module1 = compact_model.module_list[idx]
        conv_layer0 = module0[0]
        conv_layer1 = module1[0]
        if idx in CBL_idx:
            bn_layer0 = module0[1]
            bn_layer1 = module1[1]
            ind = CBLidx2mask[idx]
            ind_choose = [ i for i in range(len(ind)) if ind[i]==1.0 ]            
            bn_layer1.bias.data.copy_(bn_layer0.bias.data[ind_choose])
            bn_layer1.weight.data.copy_(bn_layer0.weight.data[ind_choose])
            bn_layer1.running_mean.data.copy_(bn_layer0.running_mean.data[ind_choose])
            bn_layer1.running_var.data.copy_(bn_layer0.running_var.data[ind_choose])
            if idx-1 in Conv_idx and idx-1 in CBL_idx:
                ind_pre = CBLidx2mask[idx-1]
                ind_choose_pre = [ i for i in range(len(ind_pre)) if ind_pre[i]==1.0 ]
                tmp = conv_layer0.weight.data[:,ind_choose_pre,...]
                conv_layer1.weight.data.copy_(tmp[ind_choose,...])
            elif idx == 84 or idx == 96 and (idx-5 in Conv_idx and idx-5 in CBL_idx):#83 is route to get 79
                    #print(conv_layer1.weight.data.size(),conv_layer0.weight.data.size())
                    ind_pre = CBLidx2mask[idx-5]
                    ind_choose_pre = [ i for i in range(len(ind_pre)) if ind_pre[i]==1.0 ]
                    tmp = conv_layer0.weight.data[:,ind_choose_pre,...]
                    conv_layer1.weight.data.copy_(tmp[ind_choose,...])
            else:
                conv_layer1.weight.data.copy_(conv_layer0.weight.data[ind_choose,...])
        else:
            conv_layer1.bias.data.copy_(conv_layer0.bias.data)
            if idx-1 in Conv_idx and idx-1 in CBL_idx:
                ind_pre = CBLidx2mask[idx-1]
                ind_choose_pre = [ i for i in range(len(ind_pre)) if ind_pre[i]==1.0 ]
                conv_layer1.weight.data.copy_(conv_layer0.weight.data[:,ind_choose_pre,...])
                #tmp = conv_layer0.weight.data[:,ind_choose_pre,...]
                #conv_layer1.weight.data.copy_(tmp[ind_choose,...])
            else:
                conv_layer1.weight.data.copy_(conv_layer0.weight.data)



def write_cfg(cfg_name, hyper):
    L = len(hyper)
    with open(cfg_name, 'w') as f:
        for i in range(L):
            if i != 0:
                f.write('['+hyper[i]['type']+']\n')
                for key in hyper[i]:
                    if key != 'type':
                        f.write(key+'='+str(hyper[i][key]))
                    f.write('\n')
            else:
                f.write('['+hyper[i]['type']+']')
                for key in hyper[i]:
                    if key != 'type':
                        f.write(key+'='+hyper[i][key])
                    f.write('\n')
    pruned_cfg_file = cfg_name 
    return pruned_cfg_file
