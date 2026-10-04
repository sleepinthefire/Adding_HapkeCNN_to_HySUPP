#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:58:49 2022


@author: Behnood
"""

import logging
import time
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from tqdm import tqdm

from src.model.blind.common import *
from .base import BlindUnmixingModel
from src.model.blind.UnmixArch import UnmixArch

from src.model.blind.UtilityMine import *
import scipy.linalg

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class HapkeCNN(nn.Module, BlindUnmixingModel):
    def __init__(self, 
                 niters=3000,
                 lr=0.001,
                 exp_weight=0.99,
                 lambd=0.1,
                 alpha=0.0001,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.niters = niters
        self.lr = lr
        self.exp_weight = exp_weight
        self.lambd = lambd
        self.alpha = alpha
        self.dtype = torch.cuda.FloatTensor

        self.p1 = 204
        self.rmax = 3

    def init_architecture(
        self,
        seed,
    ):
        # Set random seed
        torch.manual_seed(seed)

        self.conv1 = nn.Sequential(
        UnmixArch(
                self.p1, self.rmax,
                num_channels_down = [ 256],
                num_channels_up =   [ 256],
                num_channels_skip =    [ 4],  
                filter_size_up = 3,filter_size_down = 3,  filter_skip_size=1,
                upsample_mode='bilinear', # downsample_mode='avg',
                need1x1_up=False,
                need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(self.dtype)
                )
        self.dconv4 = nn.Sequential(
                        nn.Conv2d(self.rmax, self.p1, 1,1, padding=[1//2, 1//2],bias=False),
                        )

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.dconv4(x)
        return x, x1
    
    def loss(self, target, End2, alpha,lamb, out_,out_spec,rmax):
        m=1
        m0=1

        # Albedo Loss
        W=torch.pow((m0+m)**2*torch.pow(End2,2)+torch.mul(1+4*m*m0*End2, 1-End2),0.5)-(m0+m)*End2       
        W1=torch.div(W,1+4*m*m0*End2)
        End3=1-torch.pow(W1,2)
        HR=torch.mm(End3.view(self.p1,rmax),out_.view(rmax,self.nr1*self.nc1))
        Temp1=1+2*m*torch.pow((1-HR),0.5)
        Temp2=1+2*m0*torch.pow((1-HR),0.5)
        out_HR=torch.div(HR, torch.mul(Temp1,Temp2))
        loss = 0.5*torch.norm((out_HR.view(1,self.p1,self.nr1,self.nc1) - target), 'fro')**2

        #---- Net Loss-------
        loss1 = 0.5*torch.norm((out_spec.view(1,self.p1,self.nr1,self.nc1) - target), 'fro')**2

        #------Minimum Volume Penalty: TV-----
        O = torch.from_numpy(np.zeros((self.p1, rmax))).type(self.dtype)
        B = np_to_torch(np.identity(rmax) - np.ones((rmax))/rmax).type(self.dtype)
        loss2 = torch.norm(torch.mm(End3,B.view((rmax,rmax)))-O, 'fro')**2

        return loss+alpha*loss1+lamb*loss2
    
    def compute_endmembers_and_abundances(self, Y, p, H, W, seed=0, *args, **kwargs):
        tic = time.time()
        logger.debug("Solving started...")

        img_noisy_np = Y.reshape(-1, H, W)
        self.p1, self.nr1, self.nc1 = img_noisy_np.shape
        self.rmax = p

        img_resh=np.reshape(img_noisy_np,(self.p1,self.nr1*self.nc1))
        V, SS, U = scipy.linalg.svd(img_resh, full_matrices=False)
        PC=np.diag(SS)@U
        img_resh_DN=V[:,:self.rmax]@V[:,:self.rmax].transpose(1,0)@img_resh
        img_resh_np_clip=np.clip(img_resh_DN, 0, 1)
        II,III = Endmember_extract(img_resh_np_clip,self.rmax)
        E_np1=img_resh_np_clip[:,II]
        #%% Set up Simulated 
        INPUT = 'noise' # 'meshgrid'
        OPT_OVER = 'net' 
        
        # 
        LR1 = self.lr
        exp_weight = self.exp_weight

        self.init_architecture(seed=seed)

        img_noisy_torch = torch.from_numpy(img_resh_DN).view(1,self.p1,self.nr1,self.nc1).type(self.dtype)
        net_input1 = get_noise(self.p1, INPUT,
            (img_noisy_np.shape[1], img_noisy_np.shape[2])).type(self.dtype).detach()
        E_torch = torch.from_numpy(E_np1).type(self.dtype)
        #%%
        out_avg = True
        
        self.dconv4[0].weight = torch.nn.Parameter(E_torch.view(self.p1,self.rmax,1,1))       
        p11 = get_params(OPT_OVER, self, net_input1)
        optimizer = torch.optim.Adam(p11, lr=LR1, betas=(0.9, 0.999), eps=1e-8,
                  weight_decay= 0, amsgrad=False)

        progress = tqdm(range(self.niters))
        for j in progress:
            optimizer.zero_grad()

            out_LR,out_spec = self(net_input1)
            # Smoothing
            if out_avg is None:
                out_avg = out_LR.detach()
            else:
                out_avg = out_avg * exp_weight + out_LR.detach() * (1 - exp_weight)

            total_loss = self.loss(img_noisy_torch, self.dconv4[0].weight.view(self.p1,self.rmax),
                                self.alpha, self.lambd,out_LR,out_spec, self.rmax)
            
            if torch.isnan(total_loss):
                print(f"Got nan Loss")
                raise RuntimeError("Loss became Nan")

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            optimizer.step()
            self.dconv4[0].weight.data[self.dconv4[0].weight <= 0] = 0
            self.dconv4[0].weight.data[self.dconv4[0].weight >= 1] = 1
            if j>0:
                Ehat=self.dconv4[0].weight.detach().cpu().squeeze().numpy()
                  
        Ahat = out_avg.detach().cpu().squeeze().numpy().reshape(-1, H*W)
       
        self.time = time.time() - tic
        logger.info(self.print_time())

        return Ehat, Ahat

