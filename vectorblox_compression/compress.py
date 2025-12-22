from abc import ABC, abstractmethod
import json
import torch
import numpy as np


def get_module_weights(model):
    parameters = _getParameters(model)
    avg_weight = {}
    for (module, name, label) in parameters:
        t = getattr(module, name).detach()
        avg_weight[label] = t.abs().mean().item()
    return avg_weight


def get_module_text(model):
    parameters = _getParameters(model)
    text = "\"{}\": {}".format('module label','mean abs weight')
    for (module, name, label) in parameters:
        t = getattr(module, name).detach()
        t_weight = t.abs().mean()
        text += ",\n\"{}\": {}".format(label,t_weight)
    return text


def compress_vbx3(model, weight_thresh=None, json_file=None, compression=None):
    parameters = _getParameters(model)
    if len(parameters)==0:
        print("compress.py: Warning: compress_vbx() did not find any modules to compress")
        return
    
    if not json_file is None:
        with open(json_file, 'r') as file:
            comp_instructions = json.load(file)
        for (module, name, label) in parameters:
            if not label in comp_instructions:
                raise ValueError(f"compress.py: compress_vbx() did not find module {label} in json_file")
    elif not weight_thresh is None:
        comp_instructions = {}
        for (module, name, label) in parameters:
            t = getattr(module, name).detach()
            t_weight = t.abs().mean()
            is1x1 = t.shape[2]==1 and t.shape[3]==1
            if t_weight<weight_thresh and not is1x1:
                comp_instructions[label] = 4
            else:
                comp_instructions[label] = 2
    elif not compression is None:
        if compression<0.5 or compression>=0.75:
            print("compress.py: Warning: compress_vbx() compression parameter operates between 0.5 and 0.75")
        comp_instructions = {}
        N = len(parameters)
        avg_weight = np.zeros(N)
        num_weight = np.zeros(N,dtype=int)
        is_1x1 = np.zeros(N)
        for n,(module, name, label) in enumerate(parameters):
            t = getattr(module, name).detach()
            avg_weight[n] = t.abs().mean()
            num_weight[n] = t.numel()
            is_1x1[n] = t.shape[2]==1 and t.shape[3]==1
        ind = np.argsort(avg_weight*(1+is_1x1)) # sort modules; err towards lower compression for 1x1s
        possible_total_weight = np.zeros(N+1,dtype=int)
        for n in range(N+1):    # 4x compression for n modules
            possible_total_weight[n] = num_weight[ind[:n]].sum()//4 + num_weight[ind[n:]].sum()//2
        possible_compression = 1-possible_total_weight/num_weight.sum()
        n_select = np.abs(compression-possible_compression).argmin()
        for n,(module, name, label) in enumerate(parameters):
            if n in ind[:n_select]:
                comp_instructions[label] = 4
            else:
                comp_instructions[label] = 2
    else:
        raise ValueError(f"compress.py: compress_vbx() needs a parameter value for either json_file, weight_thresh, or compression")
    
    device = getattr(parameters[0][0], parameters[0][1]).device
    totalWeights = 0
    totalWeightsComp = 0
    for (module, name, label) in parameters:
        ratio = comp_instructions[label]
        if ratio==1:
            pass
        elif ratio==2:
            block_type = '4:8_4L'
        elif ratio==4:
            block_type = '2:8_4L'
        else:
            raise ValueError(f"compress.py: compress_vbx() received an unexpected compression ratio {ratio} for module {label}.")
        print("compress.py: compressing {}x, module '{}', {}, device '{}'".format(ratio,label,module.weight.shape,device))
        _CompressVbx3.apply(module, name, device, block_type)
        totalWeights += np.prod(module.weight.shape)
        totalWeightsComp += np.prod(module.weight.shape)//ratio
    print("compress.py: total compression for available modules: {:.4}".format((1-totalWeightsComp/totalWeights)))


def compress_vbx3_unstructured(model, compression):
    if compression<0.5 or compression>=1.0:
        raise ValueError(f"compress.py: compress_vbx_unstructured() needs compression>=0.5 and compression<1.0")
    sparsity = 1 - 2*(1-compression)

    #prune and pair globally
    parameters = _getParameters(model)
    global_pairs = None
    for (module, name, _) in parameters:
        t = getattr(module, name).detach()
        t_abs = torch.abs(t)
        t0_abs_sort = torch.sort(t_abs[:,0::2,:,:],dim=1)[0]
        t1_abs_sort = torch.sort(t_abs[:,1::2,:,:],dim=1)[0]
        t_abs_pairs = t0_abs_sort.abs()
        t_abs_pairs[:,:t1_abs_sort.shape[1],:,:] += t1_abs_sort.abs()
        t_abs_pairs.flatten()
        if global_pairs==None:
            global_pairs = torch.tensor([],device=t_abs_pairs.device)
        global_pairs = torch.cat((global_pairs,t_abs_pairs.flatten()))
    global_nPrune = round(sparsity * global_pairs.numel())
    global_pairs_prune = torch.zeros_like(global_pairs)
    pairs_sortind = torch.argsort(global_pairs)
    global_pairs_prune[pairs_sortind[0:global_nPrune]] = 1

    #apply mask to each layer
    pointer = 0
    for module, name, label in parameters:
        param = getattr(module, name)
        n = param.numel()
        num_pairs = -(-n//2)    # round up
        nPrune = int(global_pairs_prune[pointer : pointer + num_pairs].sum().item()) * 2
        nPrune = min(nPrune,n)
        nPair = (n-nPrune)//2
        print("compress.py: unstructured compressing {:5.2f}%, module '{}', {}".format(100*(nPrune+nPair)/n,label,module.weight.shape))
        _CompressVbx3U.apply(module, name, nPrune=nPrune, nPair=nPair)
        pointer += num_pairs    # Increment the pointer to continue slicing the global_pairs_mask


def is_compressed(module):
    for _, submodule in module.named_modules():
        for hook in submodule._forward_pre_hooks.values():
            if isinstance(hook, _BaseCompressMethod):
                return True
    return False


def finalize_compression(model):
    parameters = _getParameters(model)
    for (module, name, label) in parameters:
        _remove(module, name)


def _remove(module, name):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, _BaseCompressMethod) and hook._tensor_name == name:
            hook._remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError(
        f"Parameter '{name}' of module {module} has to be pruned before pruning can be removed"
    )


def _getParameters(model):
    parameters_to_compress = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if module.weight.shape[1]>=8 and module.weight.numel()>=32: # skip tiny convolutions
                if not (module.in_channels==module.groups):             # skip depthwise
                    parameters_to_compress.append((module, "weight", module_name))
    return parameters_to_compress


class _BaseCompressMethod(ABC):
    _tensor_name: str

    def __call__(self, module, inputs):
        setattr(module, self._tensor_name, self.apply_mask(module))

    @abstractmethod
    def compute_mask(self, t):
        pass

    def apply_mask(self, module):
        assert (
            self._tensor_name is not None
        ), f"Module {module} has to be pruned"  # this gets set in apply()
        ind = getattr(module, self._tensor_name + "_ind")
        mask = getattr(module, self._tensor_name + "_mask")
        val = getattr(module, self._tensor_name + "_val")
        compressed_tensor = mask.to(dtype=val.dtype) * val.gather(1,ind)
        return compressed_tensor

    @classmethod
    def apply(cls, module, name, *args, **kwargs):

        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, _BaseCompressMethod) and hook._tensor_name == name:
                raise RuntimeError('compress.py: Trying to apply compression to a module that is already compressed.')
        
        method = cls(*args, **kwargs)   # new compression
        method._tensor_name = name      # have the pruning method remember what tensor it's been applied to

        orig = getattr(module, name)

        # move the original tensor to a new parameter called name + '_val' and and deleting the original parameter
        module.register_parameter(name + "_val", orig)
        del module._parameters[name]
            
        try:
            mask,ind,val = method.compute_mask(orig)
            module.register_buffer(name + "_mask", mask)
            module.register_buffer(name + "_ind", ind)
            with torch.no_grad():
                orig.data = val.data
            setattr(module, name, method.apply_mask(module))
            module.register_forward_pre_hook(method)
        except Exception as e:
            orig = getattr(module, name + "_val")
            module.register_parameter(name, orig)
            del module._parameters[name + "_val"]
            raise e

        return method

    def prune(self, t, default_mask=None):
        return t * self.compute_mask(t)

    def _remove(self, module):
        assert (
            self._tensor_name is not None
        ), f"Module {module} has to be pruned before pruning can be removed"  # this gets set in apply()

        weight = self.apply_mask(module)  # compressed weights

        # delete and reset
        if hasattr(module, self._tensor_name):
            delattr(module, self._tensor_name)
        orig = module._parameters[self._tensor_name + "_val"]
        orig.data = weight.data
        del module._parameters[self._tensor_name + "_val"]
        del module._buffers[self._tensor_name + "_mask"]
        del module._buffers[self._tensor_name + "_ind"]
        setattr(module, self._tensor_name, orig)


class _CompressVbx3U(_BaseCompressMethod):

    def __init__(self, nPrune, nPair):
        self.nPrune = nPrune
        self.nPair = nPair

    def compute_mask(self, t):

        orig = t.detach()
        val = orig.clone()
        mask = torch.ones_like(orig)
        ind = torch.arange(0,orig.shape[1],device=orig.device,dtype=torch.int64).unsqueeze(1).unsqueeze(1).repeat(orig.shape[0],1,orig.shape[2],orig.shape[3])

        N = orig.numel()
        NO = orig.shape[0]
        NI = orig.shape[1]
        NIpair = NI//2
        nPrune = self.nPrune
        nPair = self.nPair
        if nPrune > N:
            raise ValueError(f"Value for nPrune: {self.nPrune} is too high for the size.")
        if (nPair*2 + nPrune) > N:
            raise ValueError(f"Values for nPrune: {self.nPrune} and nPair: {self.nPair} are too high together.")

        if nPair==0:    
            mask = mask.flatten()
            orig = orig.flatten()
            t_abs = torch.abs(orig)
            t_sortind = torch.argsort(t_abs)

            #pruning
            if nPrune>0:
                mask[t_sortind[0:nPrune]] = 0
            mask = mask.reshape(t.shape)
        else:
            MIN_OUTPUT_ACCUMULATIONS = 12   # don't prune so much that we're limited by the quantizer; note this currently assumes 100% pairing
            output_accumulations = int(np.ceil(val.shape[1:].numel()/2))*torch.ones((val.shape[0]),dtype=int) # assume all pairs
            nPrunePairs = nPrune//2
            t_abs = torch.abs(orig)
            t0_sortind = 2*torch.argsort(t_abs[:,0::2,:,:],dim=1,descending=True)
            t1_sortind = 1+2*torch.argsort(t_abs[:,1::2,:,:],dim=1,descending=True)
            t0_sort = orig.gather(1,t0_sortind)
            t1_sort = orig.gather(1,t1_sortind)
            # pair_abs = t0_sort[:,:NI//2,:,:].abs()+t1_sort.abs()
            pair_abs = t0_sort.abs()
            pair_abs[:,:NIpair,:,:] += t1_sort.abs()
            pair_sortind = torch.argsort(pair_abs.flatten())

            for n in range(N//2):
                pair_ind = torch.unravel_index(pair_sortind[n], pair_abs.shape)
                i0 = (pair_ind[0],t0_sortind[pair_ind],pair_ind[2],pair_ind[3])
                if pair_ind[1]<t1_sortind.shape[1]:
                    i1 = (pair_ind[0],t1_sortind[pair_ind],pair_ind[2],pair_ind[3])
                else:
                    i1 = None
                if n<nPrunePairs and output_accumulations[pair_ind[0]]>MIN_OUTPUT_ACCUMULATIONS:
                    # prune
                    mask[i0] = 0
                    val[i0] = 0
                    output_accumulations[pair_ind[0]] -= 1
                    if i1 is not None:
                        mask[i1] = 0
                        val[i1] = 0
                elif n<nPrunePairs+nPair and (i1 is not None):
                    # pair
                    val0 = orig[i0]
                    val1 = orig[i1]
                    sign1 = torch.sign(val0*val1)
                    mask[i1] = sign1
                    ind[i1] = i0[1]
                    valAvg = 0.5*val0 + 0.5*sign1*val1
                    val[i0] = valAvg
                    val[i1] = sign1*valAvg

        return mask,ind,val

    @classmethod
    def apply(cls, module, name, nPrune, nPair):
        return super().apply(module, name, nPrune=nPrune, nPair=nPair)


class _CompressVbx3(_BaseCompressMethod):
    def _possible2_4L():
        possible = np.zeros((0,5),dtype=int)    # [B0, D0, B1, D1, sign]
        for selB0 in [-1,0,1]:
            for selD0 in [-1,2,3]:
                if (selB0==-1 and selD0==-1):
                    continue
                for selB1 in [-1,4,5]:
                    for selD1 in [-1,6,7]:    
                        if (selB1==-1 and selD1==-1):
                            continue
                        for signM in [-1,1]:
                            if signM==-1 and (selB0==-1 or selD0==-1) and (selB1==-1 or selD1==-1):
                                continue
                            sel = np.array([selB0,selD0,selB1,selD1])
                            sign = np.array([signM])
                            control = np.concatenate((sel,sign))
                            if np.any(np.bincount(sel[sel>=0])>1):
                                continue
                            duplicate = False
                            for p in possible:
                                duplicate = (control==p).all()
                                if duplicate:
                                    break
                            if not duplicate:
                                possible = np.concatenate((possible,control.reshape((1,5))))
        return possible
    
    def _possible4_4L():
        possible = np.zeros((0,10),dtype=int)    # [B0, D0, B0, D0, B1, D1, B1, D1, sign, sign]
        for selB0a in [-1,0,1]:
            for selD0a in [-1,2,3]:
                if (selB0a==-1 and selD0a==-1):
                    continue
                for selB1a in [-1,4,5]:
                    for selD1a in [-1,6,7]:    
                        if (selB1a==-1 and selD1a==-1):
                            continue
                        for signMa in [-1,1]:
                            if signMa==-1 and (selB0a==-1 or selD0a==-1) and (selB1a==-1 or selD1a==-1):
                                continue
                            for selB0b in [-1,0,1]:
                                for selD0b in [-1,2,3]:
                                    if (selB0b==-1 and selD0b==-1):
                                        continue
                                    for selB1b in [-1,4,5]:
                                        for selD1b in [-1,6,7]:    
                                            if (selB1b==-1 and selD1b==-1):
                                                continue
                                            for signMb in [-1,1]:
                                                if signMb==-1 and (selB0b==-1 or selD0b==-1) and (selB1b==-1 or selD1b==-1):
                                                    continue
                                                sel = np.array([selB0a,selD0a,selB0b,selD0b,selB1a,selD1a,selB1b,selD1b])
                                                sign = np.array([signMa, signMb])
                                                control = np.concatenate((sel,sign))
                                                if np.any(np.bincount(sel[sel>=0])>1):
                                                    continue
                                                duplicate = False
                                                for p in possible:
                                                    duplicate = (control==p).all()
                                                    if duplicate:
                                                        break
                                                if not duplicate:
                                                    possible = np.concatenate((possible,control.reshape((1,10))))
        return possible

    def _genMatrices(possible, block_weights):
        BW = block_weights     # number of math block weights per block
        K = np.zeros((0,BW,8),dtype=float)  # matmul K[n,:,:]@(uncompressed weights) to get (math block weights)
        W = np.zeros((0,8,BW),dtype=float)  # matmul W[n,:,:]@(math block weights) to get (compressed weights)
        WK = np.zeros((0,8,8),dtype=float)  # matmul WK[n,:,:]@(uncompressed weights) to get (compressed weights)
        for p in possible:
            k = np.zeros((BW,8),dtype=float)
            w = np.zeros((BW,8),dtype=float)
            for n in range(BW):
                if BW==2:
                    sign = p[4]
                else:   # BW==4
                    sign = p[8+(n%2)]
                p0 = p[n*2]
                p1 = p[n*2+1]
                if p0>=0 and p1<0:
                    k[n,p0] = 1
                    w[n,p0] = 1
                elif p0<0 and p1>=0:
                    k[n,p1] = sign
                    w[n,p1] = sign
                elif p0>=0 and p1>=0:
                    k[n,p0] = 0.5
                    k[n,p1] = sign*0.5
                    w[n,p0] = 1
                    w[n,p1] = sign
                else:
                    assert 0
            w = w.transpose()
            wk = np.matmul(w,k)
            duplicate = False
            for kk in K:
                duplicate = (k==kk).all()
                if duplicate:
                    break
            if not duplicate:
                K = np.concatenate((K,k.reshape((1,BW,8))))
                W = np.concatenate((W,w.reshape((1,8,BW))))
                WK = np.concatenate((WK,wk.reshape((1,8,8))))
        return K,W,WK
    
    possible2 = _possible2_4L()
    possible4 = _possible4_4L()
    K2,W2,WK2 = _genMatrices(possible2,2)
    K4,W4,WK4 = _genMatrices(possible4,4)
    block_in = 8

    def __init__(self, device, type):
        self.groups = 1
        if '2:8_4L' in type:
            self.block_weights = 2
            self.K = torch.tensor(self.K2,dtype=torch.float32,device=device)
            self.W = torch.tensor(self.W2,dtype=torch.float32,device=device)
            self.WK = torch.tensor(self.WK2,dtype=torch.float32,device=device)
        elif '4:8_4L' in type:
            self.block_weights = 4
            self.K = torch.tensor(self.K4,dtype=torch.float32,device=device)
            self.W = torch.tensor(self.W4,dtype=torch.float32,device=device)
            self.WK = torch.tensor(self.WK4,dtype=torch.float32,device=device)
    
    def compressBestBlock(self, t):
        WKt = torch.matmul(self.WK,t)
        best = (WKt-t).square().sum(dim=1,keepdim=True).argmin()
        return best

    def compute_mask(self, t):
        orig = t.detach()
        mask = torch.ones_like(orig)
        ind = torch.arange(0,orig.shape[1],device=orig.device,dtype=torch.int64).unsqueeze(1).unsqueeze(1).repeat(orig.shape[0],1,orig.shape[2],orig.shape[3])
        
        #mask = default_mask.clone(memory_format=torch.contiguous_format)
        #ind = default_ind.clone(memory_format=torch.contiguous_format)

        NO = orig.shape[0]
        NI = orig.shape[1]
        BI = self.block_in
        BW = self.block_weights
        NBI = -(-NI//BI)    # divide rounding up
        if NI<8: raise ValueError(f"input channels: {NI} must be >=8.")
        val = torch.zeros_like(orig)

        for no in range(NO):
            for nbi in range(NBI):
                for nkh in range(t.shape[2]):
                    for nkw in range(t.shape[3]):
                        t_block = orig[no,nbi*BI:(nbi+1)*BI,nkh,nkw]
                        tn = t_block.numel()    # number uncompressed elements in this block
                        vn = min(tn,BW)         # number of compressed values in this block
                        t_pad = torch.nn.functional.pad(t_block,(0,self.block_in-tn))
                        b = self.compressBestBlock(t_pad)
                        t_val = torch.matmul(self.K[b,:,:],t_pad)
                        #t_comp = torch.matmul(self.W[b,:,:],k)
                        t_mask = self.W[b,:,:].sum(dim=1)
                        t_ind = torch.zeros((8),device=ind.device,dtype=ind.dtype)
                        for n in range(BW):
                            t_ind[self.W[b,:,n]!=0] = n
                        
                        mask[no,nbi*BI:nbi*BI+tn,nkh,nkw] = t_mask[:tn]
                        ind[no,nbi*BI:nbi*BI+tn,nkh,nkw] = t_ind[:tn] + nbi*BI
                        val[no,nbi*BI:nbi*BI+vn,nkh,nkw] = t_val[:vn]
                        #mask[no,nbi*BI:(nbi+1)*BI,nkh,nkw] = t_mask[:tn]
                        #ind[no,nbi*BI:(nbi+1)*BI,nkh,nkw] = t_ind[:tn] + nbi*BI
                        #val[no,nbi*BI:nbi*BI+BW,nkh,nkw] = t_val[:tn]
        assert ind.max()<NI
        return mask,ind,val

    @classmethod
    def apply(cls, module, name, device, type):
        return super().apply(module, name, device, type)
    