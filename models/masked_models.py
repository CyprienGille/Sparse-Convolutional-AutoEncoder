# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:27:08 2022

Models for masked layers. It is specific for the CAE case

@author: YFGI6212
"""
import sys

sys.path.append("../")
sys.path.append("../../")

import torch
import torch.nn as nn

from models.cae_32x32x32_zero_pad_comp import CAE
import projection.projection as proj


class Masked_CAE(CAE):
    def __init__(self):

        super(Masked_CAE, self).__init__()

        self._get_masked_layers()

    def _get_masked_layers(self):

        self._masks = {}

        self._layers = {}

        for index, param in enumerate(list(self.parameters())):
            self._masks[index] = torch.ones_like(param).to(param)

    def _init_masks(self):

        for index, param in enumerate(list(self.parameters())):
            self._masks[index] = torch.ones_like(param).to(param)

    def to(self, device):

        super(Masked_CAE, self).to(device)

        for index, param in enumerate(list(self.parameters())):
            self._masks[index].to(device)
            self._masks[index].to(device)

    def project(self, projection, projection_params):
        """ 
        Projects only the weights, but generate as well 
        the mask for the bias 
        """

        self._init_masks()

        proj_params = projection_params

        if "omit" not in proj_params.keys():
            omit_indices = []
        else:
            omit_indices = projection_params["omit"]
            pop_item = proj_params.pop("omit", None)

        for index, param in enumerate(list(self.parameters())):
            if index not in omit_indices:
                Proj = projection(param, **proj_params)
                self._masks[index][Proj == 0] = 0

    def apply_masks_on_layers(self):

        for index, param in enumerate(list(self.parameters())):
            param.data[self._masks[index] == 0] = 0

    def project_and_mask(self, projection, projection_params):

        self.project(projection, projection_params)
        self.apply_masks_on_layers()

    def mask_grad(self):

        for index, param in enumerate(list(self.parameters())):
            param.grad[self._masks[index] == 0] = 0
