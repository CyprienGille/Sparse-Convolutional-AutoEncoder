# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:14:35 2021

Basic class for a recortder specific to this autoencoder case 

@author: YFGI6212
"""
import os

import sys

sys.path.append("../")
sys.path.append("../../")

import pandas as pd
import torch
from models.masked_models import Masked_CAE


class Simple_Recorder:
    """ A simple class allowing:
                - collecting training history (training and validation losses, temporary models)
                - collecting configuration
                - saving the best model
                
        model            : a pytorch model
        optimizer        : a pytorch optimizer 
        
        
    """

    def __init__(self, model, optimizer, config):

        self._config = config
        self._model = model
        self._optimizer = optimizer

        self._best_model_file = os.path.join(
            self._config["expe_dir"], "checkpoint", f"best_model.pth"
        )
        self._init_model_file = os.path.join(
            self._config["expe_dir"], "checkpoint", f"model_0.pth"
        )
        self._summary_file = os.path.join(
            self._config["expe_dir"],
            "summary",
            f"model_{config['nickname']}_summary.csv",
        )
        self._summary = None

    def update(
        self,
        epoch,
        training_loss,
        training_entropy,
        validation_loss,
        validation_entropy,
    ):

        # check is this is the best model
        is_best = False
        best_char = ""
        indicator = {}
        results = {}

        if self._summary is None:
            is_best = "*"
        else:
            is_best = self._summary["validation_loss"].min() >= validation_loss

        if is_best:
            best_char = "*"

        if self._summary is None:
            Columns = [
                "epoch",
                "training_loss",
                "training_entropy",
                "validation_loss",
                "validation_entropy",
                "best_loss",
            ]
            Values = [
                epoch,
                training_loss,
                training_entropy,
                validation_loss,
                validation_entropy,
                "*",
            ]

            self._summary = pd.DataFrame([Values], columns=Columns)

        else:

            Values = [
                epoch,
                training_loss,
                training_entropy,
                validation_loss,
                validation_entropy,
                best_char,
            ]

            self._summary.loc[len(self._summary)] = Values

        Is_Best = {}

        to_save = {
            "config": self._config,
            "summary": self._summary,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
        }

        if isinstance(self._model, Masked_CAE):
            to_save["masks"] = self._model._masks

        save_file = os.path.join(
            self._config["expe_dir"], "checkpoint", f"model_{epoch}.pth"
        )

        if epoch == 0:
            torch.save(to_save, self._init_model_file)
        else:
            torch.save(to_save, save_file)

        self._summary.to_csv(self._summary_file)

        if is_best:
            torch.save(to_save, self._best_model_file)

        Is_Best["loss"] = is_best

        return Is_Best

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model

    @property
    def summary(self):
        return self._summary
