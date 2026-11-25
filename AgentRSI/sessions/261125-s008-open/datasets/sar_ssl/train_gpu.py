import os,time,random,json
os.environ['OMP_NUM_THREADS']='4'
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import PCA
SEED=20261125
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.set_num_threads(4)
A=np.load('/content/sar_patches.npz');X=torch.tensor(A['patches'].astype('float32'),device='cuda');N=len(X)
print('input',X.shape,'GPU',torch.cuda.get_device_name(0),flush=True)
class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.conv=nn.Sequential(nn.Conv2d(22,32,3,padding=1),nn.GroupNorm(4,32),nn.GELU(),nn.Conv2d(32,48,3,stride=2,padding=1),nn.GroupNorm(6,48),nn.GELU(),nn.Conv2d(48,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.GELU());self.project=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,64))
 def forward(self,x,project=False):
  z=self.conv(x);z=torch.cat([z.mean((2,3)),z[:,:,1:3,1:3].mean((2,3))],dim=1)
  return self.project(z) if project else z
class AE(nn.Module):
 def __init__(self):
  super().__init__();self.enc=Encoder();self.dec=nn.Sequential(nn.Linear(128,64*4*4),nn.GELU(),nn.Unflatten(1,(64,4,4)),nn.ConvTranspose2d(64,48,4,stride=2,padding=1),nn.GELU(),nn.ConvTranspose2d(48,22,4,stride=2,padding=1))
 def forward(self,x):return self.dec(self.enc(x))[:,:,:15,:15]
def aug(x):
 x=x.clone();k=random.randrange(4);x=torch.rot90(x,k,(2,3))
 if random.random()<.5:x=x.flip(3)
 x=x+torch.randn_like(x)*.07
 # Consistent channels are retained, with weak radiometric and speckle perturbations.
 return x*torch.exp(torch.randn(len(x),22,1,1,device=x.device)*.025)
def cov(z):
 z=z-z.mean(0);c=z.T@z/(len(z)-1);return (c.square().sum()-c.diag().square().sum())/z.shape[1]
def vic(a,b):
 invariance=F.mse_loss(a,b);std=.5*(F.relu(1-torch.sqrt(a.var(0)+1e-4)).mean()+F.relu(1-torch.sqrt(b.var(0)+1e-4)).mean());return 25*invariance+25*std+cov(a)+cov(b)
start=time.time();enc=Encoder().cuda();opt=torch.optim.AdamW(enc.parameters(),lr=7e-4,weight_decay=1e-4)
for epoch in range(121):
 order=torch.randperm(N,device='cuda');total=0
 for ix in order.split(256):
  x=X[ix];opt.zero_grad(set_to_none=True);loss=vic(enc(aug(x),True),enc(aug(x),True));loss.backward();opt.step();total+=loss.item()
 if epoch%20==0:print('VICReg',epoch,total/len(order.split(256)),'sec',time.time()-start,flush=True)
 if time.time()-start>600:break
enc.eval()
def embed(model):
 feats=[]
 with torch.no_grad():
  for x in X.split(256):
   feats.append(torch.stack([model(torch.rot90(x,k,(2,3))) for k in range(4)]).mean(0).cpu().numpy())
 return np.concatenate(feats)
v=embed(enc);torch.save(enc.cpu().state_dict(),'/content/vicreg.pt');del enc,opt;torch.cuda.empty_cache()
ae=AE().cuda();opt=torch.optim.AdamW(ae.parameters(),lr=8e-4,weight_decay=1e-4);st=time.time()
for epoch in range(101):
 total=0
 for ix in torch.randperm(N,device='cuda').split(256):
  target=aug(X[ix]);mask=(torch.rand(len(ix),1,15,15,device='cuda')>.3).float();inp=target*mask+torch.randn_like(target)*.1
  opt.zero_grad(set_to_none=True);pred=ae(inp);loss=F.smooth_l1_loss(pred,target);loss.backward();opt.step();total+=loss.item()
 if epoch%20==0:print('AE',epoch,total,'sec',time.time()-st,flush=True)
 if time.time()-st>480:break
ae.eval();w=embed(ae.enc)
with torch.no_grad():err=np.concatenate([(ae(x)-x).square().mean((1,2,3)).cpu().numpy() for x in X.split(256)])
np.savez_compressed('/content/embeddings.npz',vicreg=v,masked_ae=w,reconstruction=err)
torch.save(ae.cpu().state_dict(),'/content/ae.pt')
print('output',v.shape,w.shape,'total_sec',time.time()-start,flush=True)
