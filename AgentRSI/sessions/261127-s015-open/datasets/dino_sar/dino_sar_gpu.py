"""Frozen DINOv2 on fixed-display multi-date SAR patches. Label-free."""
import os,time,json
os.environ['OMP_NUM_THREADS']='4'
import numpy as np,torch
import torch.nn.functional as F
torch.set_num_threads(4);start=time.time();assert torch.cuda.is_available()
model=torch.hub.load('facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8','dinov2_vits14',pretrained=True,trust_repo=True).cuda().eval()
data=np.load('/content/s010_sar.npz')['chips'];mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None];out={}
for name,side,date in [('jun51',8,0),('oct51',8,1),('jun103',16,0),('oct103',16,1),('change103',16,2)]:
    feat=[]
    for i in range(0,len(data),96):
        if date<2:a=torch.from_numpy(data[i:i+96,date].copy()).cuda().float()/255
        else:
            a=torch.from_numpy(data[i:i+96].copy()).cuda().float()/255;a=(.5+(a[:,1]-a[:,0])).clamp(0,1)
        lo=16-side//2;a=a[:,:,lo:lo+side,lo:lo+side];a=F.interpolate(a,size=(224,224),mode='bilinear',align_corners=False)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):v=model((a-mean)/std)
        feat.append(v.float().cpu().numpy())
    out[name]=np.concatenate(feat);print(name,out[name].shape,'seconds',time.time()-start,flush=True)
np.savez_compressed('/content/s010_sar_dino.npz',**out)
print('DONE',flush=True)
