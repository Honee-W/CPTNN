import torch 
import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 
import random 
import concurrent.futures
import sys, os 
sys.path.append(os.path.dirname(__file__))



def random_mask(x: th.Tensor) -> th.Tensor:
    '''
    x: [channels, num_frames, time_frames]
    '''
    C, F, T = x.shape 

    masked = random.randint(0, C // 8)
    for i in range(masked):
        idx = random.randint(0, C - 1) 
        x[idx,...] = 0.0 
    return x 

def random_mask_by_batch(x: th.Tensor) -> th.Tensor:
    '''
    x: [batch, channels, num_frames, time_frames]
    '''
    B, _, _, _ = x.shape  
    mask = th.ones_like(x)
    ret = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_mask = {executor.submit(random_mask, mask[i,...]): i for i in range(B)}
        for future in concurrent.futures.as_completed(future_to_mask):
            m =  future_to_mask[future]
            try:
                data = future.result()
                ret.append(data)
            except Exception as exc:
                print('%r generated an exception: %s' % (m, exc))

    return th.stack(ret, dim=0) * x             


def test(x):
    print("after", random_mask_by_batch(x))

if __name__=="__main__":
    x = th.rand([2, 8, 1, 1])
    print("before", x)
    test(x)
