# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:56:00 2022

Retraining Models with the possibility of masking (lottery ticket)

@author: YFGI6212
"""

import sys

sys.path.append("../")
sys.path.append("../../")


import torch

from models.cae_32x32x32_zero_pad_comp import CAE
from models.masked_models import Masked_CAE


def load_projected_model(
    model_file=None, initial_model_file=None, projection=None, projection_params=None
):
    """
    model: typically the mest model of an experiment
    
    # Load the model
    1 - load model 
    # Project the model and generates the corresponding masks
    2 - model2 = model.project(projection, projection_params)
    # Load the state of a second model (typically an initial model in model2
    # Model2 is then masked and can be retrained
    3 - model2.state_dict() = initial_model.state_dict()
    # Apply the mask on the re-initialised weights of the model
    4 - model2.apply_mask_on_layers()
    4 - return the masked model2
    """

    new_model = Masked_CAE()

    save_dict = torch.load(model_file, map_location="cpu")

    new_model.load_state_dict(save_dict["model_state_dict"])

    new_model.project(projection, projection_params)  # create masks

    masks = new_model._masks

    save_dict = torch.load(initial_model_file, map_location="cpu")

    new_model.load_state_dict(save_dict["model_state_dict"])

    new_model._masks = masks
    new_model.apply_masks_on_layers()

    return new_model

