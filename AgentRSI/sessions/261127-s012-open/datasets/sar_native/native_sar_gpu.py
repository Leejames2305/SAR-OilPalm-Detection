"""Label-free native-SAR denoising and paired-date self-supervision, GPU only.
No ground truth, tree identifiers, coordinates or spatial split assignments enter.
"""
import os,time,json,hashlib,random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(4)
device='cuda'
raw=np.load('/content/native_sar.npz')
data=torch.tensor(raw['patches'],dtype=torch.float32,device=device)
N=len(data);B=256
class Encoder(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(c,48,3,padding=1),nn.GroupNorm(6,48),nn.GELU(),
                               nn.Conv2d(48,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.GELU(),
                               nn.Conv2d(64,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.GELU())
        self.head=nn.Linear(64*3*3,64)
    def forward(self,x):
        h=self.net(x)
        return self.head(h.flatten(1))
class AE(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder(22)
        self.decode=nn.Sequential(nn.Linear(64,64*3*3),nn.GELU(),nn.Unflatten(1,(64,3,3)),
                                  nn.ConvTranspose2d(64,64,4,2,1),nn.GELU(),
                                  nn.ConvTranspose2d(64,48,4,2,1),nn.GELU(),nn.Conv2d(48,22,3,padding=1))
    def forward(self,x):return self.decode(self.encoder(x))

def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=False
def augment(x):
    # Same geometric augmentation across spectral/date channels.
    k=int(torch.randint(4,(1,)).item());x=torch.rot90(x,k,(-2,-1))
    if torch.rand(())<.5:x=x.flip(-1)
    return x
def encode(model,x):
    model.eval();z=[]
    with torch.no_grad():
        for batch in x.split(256):z.append(model(batch).cpu().numpy())
    return np.concatenate(z)

outputs={};logs=[]
for s in [21,42]:
    seed(s);model=AE().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100)
    start=time.time()
    for epoch in range(100):
        model.train();losses=[]
        for ix in torch.randperm(N,device=device).split(B):
            x=augment(data[ix]);mask=(torch.rand(len(ix),1,6,6,device=device)>.35).float()
            mask=F.interpolate(mask,size=(12,12),mode='nearest')
            cm=(torch.rand(len(ix),22,1,1,device=device)>.1).float()
            noisy=x*mask*cm+.10*torch.randn_like(x)
            pred=model(noisy)
            loss=(F.smooth_l1_loss(pred,x,reduction='none')*(1.5-mask*.5)).mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        scheduler.step()
        if epoch%10==0 or epoch==99:
            row={'seed':s,'epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.time()-start};logs.append(row);print(row,flush=True)
    outputs[f'mae{s}']=encode(model.encoder,data)
    torch.save(model.state_dict(),f'/content/mae{s}.pt')

# Stable temporal morphology: VICReg paired-date objective, no disease labels.
seed(73);encoder=Encoder(11).to(device)
projector=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,128)).to(device)
opt=torch.optim.AdamW(list(encoder.parameters())+list(projector.parameters()),lr=.0005,weight_decay=.0001)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100)
start=time.time()
for epoch in range(100):
    encoder.train();projector.train();losses=[]
    for ix in torch.randperm(N,device=device).split(B):
        x=augment(data[ix]);xa=x[:,:11]+.05*torch.randn_like(x[:,:11]);xb=x[:,11:]+.05*torch.randn_like(x[:,11:])
        za,zb=projector(encoder(xa)),projector(encoder(xb))
        inv=F.mse_loss(za,zb)
        va=F.relu(1-torch.sqrt(za.var(0)+.0001)).mean();vb=F.relu(1-torch.sqrt(zb.var(0)+.0001)).mean()
        ac=za-za.mean(0);bc=zb-zb.mean(0)
        ca=ac.T@ac/(len(ix)-1);cb=bc.T@bc/(len(ix)-1)
        eye=torch.eye(128,device=device,dtype=torch.bool)
        cov=(ca[~eye].square().sum()+cb[~eye].square().sum())/128
        loss=25*inv+25*(va+vb)+cov
        opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
    scheduler.step()
    if epoch%10==0 or epoch==99:
        row={'objective':'vicreg','epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.time()-start};logs.append(row);print(row,flush=True)
outputs['vicjun']=encode(encoder,data[:,:11]);outputs['vicoct']=encode(encoder,data[:,11:])
outputs['vicdelta']=outputs['vicoct']-outputs['vicjun']
torch.save(encoder.state_dict(),'/content/vicreg.pt')
np.savez_compressed('/content/native_embeddings.npz',**outputs)
with open('/content/native_training.json','w') as f:json.dump({'logs':logs,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'input_sha256':hashlib.sha256(open('/content/native_sar.npz','rb').read()).hexdigest()},f,indent=2)
print('DONE',[(k,v.shape) for k,v in outputs.items()],flush=True)
