import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder,boundary


def shift(x,amount):
    if amount>0:return F.pad(x[...,:-amount],(amount,0))
    if amount<0:return F.pad(x[...,-amount:],(0,-amount))
    return x


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',24);self.encoder=Encoder(2,w)
        self.decoder=nn.Sequential(nn.Conv2d(2*w+1,w,3,padding=1),nn.SiLU(),nn.Conv2d(w,1,1))
    def forward(self,b):
        x=b['x'].float();mask=b['mask'].float();h=self.encoder(torch.cat([x,mask],1));valid=1-mask
        profile=(x*valid).sum(2)/valid.sum(2).clamp_min(1);profile=profile-profile.mean(-1,keepdim=True)
        periods=list(range(self.c.get('min_period',3),min(self.c.get('max_period',15),x.shape[-1]-1)+1))
        correlation=torch.stack([(profile[...,p:]*profile[...,:-p]).sum((1,2))/(profile[...,p:].norm(dim=(1,2))*profile[...,:-p].norm(dim=(1,2))+1e-8) for p in periods],1)
        selected=correlation.argmax(1);references=[];pitches=[]
        for i in range(len(x)):
            period=periods[int(selected[i])];pitches.append(period);candidates=[];scores=[]
            for sign in [-1,1]:
                offset=sign*period;reference=shift(h[i:i+1],offset);overlap=valid[i:i+1]*shift(valid[i:i+1],offset)
                scores.append((h[i:i+1]*reference*overlap).sum()/(torch.norm(h[i:i+1]*overlap)*torch.norm(reference*overlap)+1e-8));candidates.append(reference)
            weight=torch.stack(scores).softmax(0);references.append(sum(weight[j]*candidates[j] for j in range(2)))
        reference=torch.cat(references)
        if self.c['variant']=='baseline':reference=torch.zeros_like(reference)
        restored=self.decoder(torch.cat([h,reference,mask],1)).sigmoid()
        if self.c['variant']=='no_mask':mask=torch.ones_like(mask)
        return {'prediction':(1-mask)*x+mask*restored,'period':torch.tensor(pitches,device=x.device).float()[:,None]}
    def objective(self,b,p):
        mask=b['mask'].float();ring=boundary(mask,2);error=abs(p['prediction']-b['y'])
        loss=(error*(mask+.2*ring)).sum()/(mask+.2*ring).sum().clamp_min(1)
        loss=loss+.2*F.l1_loss(p['prediction'][...,1:]-p['prediction'][...,:-1],b['y'][...,1:]-b['y'][...,:-1])
        if self.c['variant']!='no_period_loss':
            for lag in [3,5,7]:loss=loss+.05*F.l1_loss(p['prediction'][...,lag:]*p['prediction'][...,:-lag],b['y'][...,lag:]*b['y'][...,:-lag])
        return loss


def demo(seed,n,c):
    rng=np.random.default_rng(seed);pitch=rng.integers(4,10,n);y=np.array([np.tile((np.arange(32)%p<p/2).astype(np.float32),(32,1))[None] for p in pitch])
    mask=np.zeros_like(y);mask[:,:,10:20,12:19]=1;x=(1-mask)*y+mask*rng.uniform(size=y.shape)
    return {'x':x.astype('float32'),'y':y,'mask':mask,'group':np.arange(n),'true_period':pitch[:,None]}


def measure(p,d,c):
    outside=d['mask']==0;result={'exterior_change':float(np.mean(abs(p['prediction'][outside]-d['x'][outside])))}
    if 'true_period' in d:result['period_mae']=float(np.mean(abs(p['period']-d['true_period'])))
    return result
