import torch 
import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 


class Transformer_SA(nn.Module):
    '''
    transformer with  self-attention    
    '''
    def __init__(self, embed_dim, hidden_size, num_heads, bidirectional=True):
        super(Transformer_SA, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, bidirectional=bidirectional)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size, embed_dim)
        self.dropout = nn.Dropout()
        self.ln3 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        '''
        x: [length, batch, dimension]
        return: [length, batch, dimension]
        '''        
        y = self.ln1(x)
        y, _ = self.mha(y, y, y)
        y += x 
        z = self.ln2(y)
        z, _ = self.gru(z)
        z = self.gelu(z)
        z = self.fc(z)
        z = self.dropout(z)
        z += y
        z = self.ln3(z)
        return z

class Transformer_CA(nn.Module):
    '''
    transformer with  cross-attention    
    '''
    def __init__(self, embed_dim, hidden_size, num_heads, bidirectional=True):
        super(Transformer_CA, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, bidirectional=bidirectional)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size, embed_dim)
        self.dropout = nn.Dropout()
        self.ln3 = nn.LayerNorm(embed_dim)

    def forward(self, x0, x1):
        '''
        x0, x1: [length, batch, dimension]
        return: [length, batch, dimension]
        '''        
        y0 = self.ln1(x0)
        y1 = self.ln1(x1)
        y, _ = self.mha(y0, y1, y1)
        y += x0 
        z = self.ln2(y)
        z, _ = self.gru(z)
        z = self.gelu(z)
        z = self.fc(z)
        z = self.dropout(z)
        z += y
        z = self.ln3(z)
        return z    


class CPTB(nn.Module):
    '''
    cross-parallel transformer block (CPTB)
    '''
    def __init__(self, embed_dim, hidden_size, num_heads, num_groups):
        super(CPTB, self).__init__()
        self.local_transformer = Transformer_SA(embed_dim, hidden_size, num_heads)
        self.local_norm = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim)
       
        self.global_transformer = Transformer_SA(embed_dim, hidden_size, num_heads)
        self.global_norm = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim)

        self.fusion_transformer = Transformer_CA(embed_dim, hidden_size, num_heads)
        self.fusion_norm = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim)

    def forward(self, x):
        '''
        x:  [batch, channels, num_frames, time_frames]
        return: [batch, channels, num_frames, time_frames]
        '''
        B, C, F, L = x.shape 
        local_feat = th.reshape(x.permute(0, 2, 3, 1), (-1, L, C))
        global_feat = th.reshape(x.permute(0, 3, 2, 1), (-1, F, C))

        local_feat = self.local_transformer(local_feat)
        local_feat = self.local_norm(local_feat.transpose(-1,-2)).transpose(-1,-2)

        global_feat = self.global_transformer(global_feat)
        global_feat = self.global_norm(global_feat.transpose(-1,-2)).transpose(-1,-2)

        fusion_feat = self.fusion_transformer(local_feat, global_feat)
        fusion_feat = self.fusion_norm(fusion_feat.transpose(-1,-2)).transpose(-1,-2)

        fusion_feat = th.reshape(fusion_feat, [B, C, F, L])
        return fusion_feat


if __name__=="__main__":
    inputs = th.rand([4, 64, 400, 320])
    net = CPTB(embed_dim=64, hidden_size=128, num_heads=4, num_groups=4) 
    print(net(inputs).shape)
