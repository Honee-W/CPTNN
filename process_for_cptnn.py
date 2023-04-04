import torch 
import torch as th 
import torch.nn as nn
import torch.nn.functional as tnf 
import concurrent.futures 
import math 

def seg_and_add(wav: th.Tensor, frame_len: int=320, hop_size: int=160) -> th.Tensor:
    '''
    wav: [length]
    return: [1, F, L]
    '''
    if len(wav.shape) == 2:
        wav = wav.squeeze(0)
    assert len(wav.shape) <= 2 
    segs = []
    length = wav.shape[0]
    F = int(math.floor((length - frame_len) / (frame_len - hop_size) + 1) + 1)
    offset = 0
    for i in range(F):
        seg = wav[offset : offset + frame_len]
        if seg.shape[0] < frame_len:
            seg = tnf.pad(seg, (0, frame_len - seg.shape[0])) # padding at the end       
        segs.append(seg.unsqueeze(0))
        offset += hop_size
    
    return th.stack(segs, dim=1)



# ThreadExecutor.map() --> multithread processing while keeping the original order

def seg_and_add_by_batch(wav: th.Tensor, frame_len: int=320, hop_size: int=160) -> th.Tensor:
    '''
    inputs: [Batch, Length]
    return: [Batch, 1,  F, frame_len]
    '''
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(0)
    assert len(wav.shape) <= 2
    B, _ = wav.shape
    wav = th.chunk(wav, chunks=B, dim=0) 
    ret = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for segs in executor.map(seg_and_add, wav):
            ret.append(segs)
    ret = th.stack(ret, dim=0)
    return ret


def restore_to_wav(segs: th.Tensor, frame_len: int=320, hop_size: int=160) -> th.Tensor:
    '''
    segs: [F, L]
    return: [length]
    '''
    if len(segs.shape) == 3:
        segs = segs.squeeze(0)
    assert len(segs.shape) == 2
    
    F, L = segs.shape 
    wav = segs[0, ...]
    offset = frame_len - hop_size
    for i in range(1, F):
        wav = th.cat((wav, segs[i,offset:]))
    return wav 


def restore_to_wav_by_batch(segs: th.Tensor, frame_len: int=320, hop_size: int=160) -> th.Tensor:
    '''
    inputs: [Batch, F, frame_len]
    return: [Batch, Length]    
    '''
    if len(segs.shape) == 4:
        segs = segs.squeeze(1)
    assert len(segs.shape) == 3 
    B, _, _ = segs.shape 
    segs = th.chunk(segs, chunks=B, dim=0)
    ret = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for wav in executor.map(restore_to_wav, segs):
            ret.append(wav)
    ret = th.stack(ret, dim=0)
    return ret

if __name__=="__main__":
    wav = th.rand([4, 16000*4])
    out = seg_and_add_by_batch(wav)
    restored = restore_to_wav_by_batch(out)[...,:16000*4]
    idx = (wav==restored)
    print(idx)
    print(th.equal(wav, restored))