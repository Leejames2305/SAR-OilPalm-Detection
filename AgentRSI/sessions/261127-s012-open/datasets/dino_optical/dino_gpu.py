"""Frozen pretrained DINOv2; no ground truth or supervised training."""
import os,time,json
os.environ['OMP_NUM_THREADS']='4'
import numpy as np,torch
import torch.nn.functional as F
torch.set_num_threads(4)
start=time.time()
print('torch',torch.__version__,'cuda',torch.cuda.is_available(),flush=True)
assert torch.cuda.is_available()
torch.hub.set_dir('/content/torch_cache')
model=torch.hub.load('facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8','dinov2_vits14',pretrained=True,trust_repo=True).cuda().eval()
data=np.load('/content/s010_optical.npz')['chips']
mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
out={}
for name,side,bands in [('rgb80',8,[0,1,2]),('rgb160',16,[0,1,2]),('rgb320',32,[0,1,2]),('nrg160',16,[3,0,1])]:
    features=[]
    for i in range(0,len(data),96):
        a=torch.from_numpy(data[i:i+96,bands]).cuda();lo=16-side//2
        a=a[:,:,lo:lo+side,lo:lo+side]
        # Fixed reflectance mapping, no per-image histogram normalization.
        a=(a/(.3 if name.startswith('rgb') else .5)).clamp(0,1)
        a=F.interpolate(a,size=(224,224),mode='bilinear',align_corners=False)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            z=model((a-mean)/std)
        features.append(z.float().cpu().numpy())
    out[name]=np.concatenate(features).astype(np.float32)
    print(name,out[name].shape,'seconds',time.time()-start,flush=True)
np.savez_compressed('/content/s010_dino.npz',**out)
with open('/content/s010_dino_meta.json','w') as f:json.dump({'encoder':'dinov2_vits14','torch':torch.__version__,'seconds':time.time()-start,'views':list(out),'training':'none; frozen pretrained','input_shape':list(data.shape)},f)
print('DONE',flush=True)
