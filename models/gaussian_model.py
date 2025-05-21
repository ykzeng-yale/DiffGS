#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl 

import sys
import os 
from pathlib import Path
import numpy as np 
import math

from einops import rearrange, reduce

from models.archs.gs_decoder import *
from models.encoder_factory import get_encoder
import inspect
from utils import evaluate


class GsModel(pl.LightningModule):

    def __init__(self, specs, point2gs=False):
        super().__init__()
        
        self.specs = specs
        model_specs = self.specs["GSModelSpecs"]
        self.hidden_dim = model_specs["hidden_dim"]
        self.latent_dim = model_specs["latent_dim"]
        self.skip_connection = model_specs.get("skip_connection", True)
        self.tanh_act = model_specs.get("tanh_act", False)
        self.pn_hidden = model_specs.get("pn_hidden_dim", self.latent_dim)

        encoder_type = self.specs.get("encoder_type", "conv_pointnet")
        encoder_kwargs = self.specs.get("encoder_kwargs", {})

        in_dim = 3 if point2gs else 59
        self.pointnet = get_encoder(
            encoder_type,
            c_dim=self.latent_dim,
            dim=in_dim,
            hidden_dim=self.pn_hidden,
            plane_resolution=64,
            out_channels=self.latent_dim,
            **encoder_kwargs,
        )
        self._pointnet_argcount = len(inspect.signature(self.pointnet.forward).parameters)
        
        self.model = GSDecoder(latent_size=self.latent_dim, hidden_dim=self.hidden_dim, skip_connection=self.skip_connection, tanh_act=self.tanh_act)

        self.occ_model = OccDecoder(latent_size=self.latent_dim, hidden_dim=self.hidden_dim, skip_connection=self.skip_connection, tanh_act=self.tanh_act)

        self.color_model = ColorDecoder(latent_size=self.latent_dim, hidden_dim=self.hidden_dim, skip_connection=self.skip_connection, tanh_act=self.tanh_act)
        
        self.occ_model.train()
        self.color_model.train()

            
    def forward(self, pc, gs):
        if self._pointnet_argcount >= 2:
            out = self.pointnet(pc, gs)
        else:
            out = self.pointnet(pc)
        shape_features = out[0] if isinstance(out, tuple) else out

        return self.model(gs, shape_features).squeeze()

    def forward_with_plane_features(self, plane_features, gs):
        gs = gs[:,:,:3]
        if hasattr(self.pointnet, 'forward_with_plane_features'):
            point_features = self.pointnet.forward_with_plane_features(plane_features, gs)
        else:
            raise NotImplementedError('Encoder does not support plane features')
        pred_color = self.color_model( torch.cat((gs, point_features),dim=-1))
        pred_gs = self.model( torch.cat((gs, point_features),dim=-1))
        return pred_color, pred_gs # [B, num_points]
    

    def forward_with_plane_features_occ(self, plane_features, gs):
        if hasattr(self.pointnet, 'forward_with_plane_features'):
            point_features = self.pointnet.forward_with_plane_features(plane_features, gs)
        else:
            raise NotImplementedError('Encoder does not support plane features')
        pred_occ = self.occ_model( torch.cat((gs, point_features),dim=-1) )
        return pred_occ # [B, num_points]
    