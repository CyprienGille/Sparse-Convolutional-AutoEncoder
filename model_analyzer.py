# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:27:08 2022

!!!!!
Model analyzer specific for the cae model as defined in cae_32x32x32_zero_pad_bin_comp.py.
It can not be used for other models.

Not intended to be used outside of generate_results.py

Current per layers indicators
            - number of input channels
            - input shape
            - number of output_channels
            - output shape
            - number of learnable parameters / sparsity (on the final line)
            - number of weight vanishing coefficients
            - number of weight vanishing kernel columns
            - number of weight vanishing kernel rows
            - bias number of vanishing coefficients
            - weight entropy
            - bias entropy
            - MACC
            - reduced MACC

@author: YFGI6212
"""


import pandas as pd
import torch
from fvcore.nn import FlopCountAnalysis

# model specific
from models.cae_32x32x32_zero_pad_bin_comp import CAE


class CAEModelAnalyzer:
    def __init__(self):

        self._model = CAE()
        self._indicator_names = [
            "type",
            "in_channels",
            "in_shape",
            "out_channels",
            "out_shape",
            "params/sparsity",
            "v_weight",
            "v_r_weight",
            "v_c_weight",
            "v_bias",
            "weight_ent",
            "bias_ent",
            "MACC",
            "reduced MACC",
            "KB32",
        ]

    def load_model(self, model_path):

        self._model.load_state_dict(
            torch.load(model_path, map_location="cpu")["model_state_dict"]
        )
        self._model.eval()

        self._input_h = 128  # horizontal size of input image
        self._input_w = 128  # vertical size of input image
        self._input_channels = self._model.e_conv_1[1].in_channels
        self._fake_input = torch.rand(
            1, self._input_channels, self._input_h, self._input_w, requires_grad=False
        )
        self._get_all_layers()

    def _get_all_layers(self):

        inputs = self._fake_input.clone()

        encoder = [
            self._model.e_conv_1,
            self._model.e_conv_2,
            self._model.e_block_1,
            self._model.e_block_2,
            self._model.e_block_3,
            self._model.e_conv_3,
        ]

        self.encoder = torch.nn.Sequential(*encoder)
        self.encoder_tuples = [k[i] for k in encoder for i in range(k.__len__())]

        for i in range(len(self.encoder_tuples)):
            input_shape = inputs.shape
            inputs = self.encoder_tuples[i](inputs)
            self.encoder_tuples[i] = (
                self.encoder_tuples[i],
                list(input_shape),
                list(inputs.shape),
            )

        decoder = [
            self._model.d_up_conv_1,
            self._model.d_block_1,
            self._model.d_block_2,
            self._model.d_block_3,
            self._model.d_up_conv_2,
            self._model.d_up_conv_3,
        ]

        self.decoder = torch.nn.Sequential(*decoder)
        self.decoder_tuples = [k[i] for k in decoder for i in range(k.__len__())]

        for i in range(len(self.decoder_tuples)):
            input_shape = inputs.shape
            inputs = self.decoder_tuples[i](inputs)
            self.decoder_tuples[i] = (
                self.decoder_tuples[i],
                list(input_shape),
                list(inputs.shape),
            )

    def analyze_encoder(self, add_sum=True):

        results = []
        for i in range(len(self.encoder_tuples)):
            layer_type = eval(f"Analyze{self.encoder_tuples[i][0]._get_name()}")
            results.append(layer_type(self.encoder_tuples[i]))

        if add_sum:
            theSum = ["General"]  # 0 type
            theSum.append("-")  # 1 in_channels
            theSum.append("-")  # 2 in_shape
            theSum.append("-")  # 3 out_channels
            theSum.append("-")  # 4 out_shape
            # theSum.append(sum([results[i][5] for i in range(len(results))]))  # 5 params
            theSum.append(ModuleSparsity(self.encoder_tuples))  # 5 sparsity
            theSum.append(
                sum([results[i][6] for i in range(len(results))])
            )  # 6 v_weight
            theSum.append(0.0)  # 7 v_r_weight
            theSum.append(0.0)  # 8 v_c_weight
            theSum.append(0.0)  # 9 v_bias
            theSum.append(ModuleEntropy(self.encoder_tuples))  # 10 weight_ent
            theSum.append(0.0)  # 11 bias_ent
            theSum.append(sum([results[i][12] for i in range(len(results))]))  # 12 MACC
            theSum.append(
                sum([results[i][13] for i in range(len(results))])
            )  # 13 reduced MACC
            theSum.append(sum([results[i][14] for i in range(len(results))]))  # 14 KB32
            # theSum.append(ModuleFlops(self.encoder, self._fake_input))  # 14 FLOPs

            results.append(theSum)

        return pd.DataFrame(results, columns=self._indicator_names)

    def analyze_decoder(self, add_sum=True):

        results = []
        for i in range(len(self.decoder_tuples)):
            layer_type = eval(f"Analyze{self.decoder_tuples[i][0]._get_name()}")
            results.append(layer_type(self.decoder_tuples[i]))

        if add_sum:
            theSum = ["General"]  # 0 type
            theSum.append("-")  # 1 in_channels
            theSum.append("-")  # 2 in_shape
            theSum.append("-")  # 3 out_channels
            theSum.append("-")  # 4 out_shape
            # theSum.append(sum([results[i][5] for i in range(len(results))]))  # 5 params
            theSum.append(ModuleSparsity(self.decoder_tuples))
            theSum.append(
                sum([results[i][6] for i in range(len(results))])
            )  # 6 v_weight
            theSum.append(0.0)  # 7 v_r_weight
            theSum.append(0.0)  # 8 v_c_weight
            theSum.append(0.0)  # 9 v_bias
            theSum.append(ModuleEntropy(self.decoder_tuples))  # 10 weight_ent
            theSum.append(0.0)  # 11 bias_ent
            theSum.append(sum([results[i][12] for i in range(len(results))]))  # 12 MACC
            theSum.append(
                sum([results[i][13] for i in range(len(results))])
            )  # 13 reduced MACC
            theSum.append(sum([results[i][14] for i in range(len(results))]))  # 14 KB32
            # theSum.append(ModuleFlops(self.decoder, self.encoder(self._fake_input)))

            results.append(theSum)

        return pd.DataFrame(results, columns=self._indicator_names)

    def show_model(self):

        return self._model

    def showencoder_tuples(self, add_sum=True):

        encoder = []
        names = ["layer", "input shape", "output shape"]
        for k in self.encoder_tuples:
            encoder.append(list(k))

        return pd.DataFrame(encoder, columns=names)

    def showdecoder_tuples(self, add_sum=True):

        decoder = []
        names = ["layer", "input shape", "output shape"]
        for k in self.decoder_tuples:
            decoder.append(list(k))

        return pd.DataFrame(decoder, columns=names)


# Analysis for specific layers
# The layer_tuple must be of the form (layer_object, input_shape, output_shape)
# with input_shape (C_in,h,w) and similarily for output_shape

# Useful methods


def TensorEntropy(tensor, tol=2):
    """ Determine the entropy of a tensor.
        - tol: number of decimals  
    """
    value, count = torch.unique(tensor.clone().detach(), return_counts=True)

    p = count / count.sum().item()

    return round(-(p * torch.log2(p)).sum().item(), tol)


# Entropy of a module tuple (as used in CAEModelAnalyzer)
def ModuleEntropy(module):
    """Returns the entropy of a pytorch module"""
    all_weights = GetAllParams(module)
    return TensorEntropy(all_weights)


def ModuleSparsity(module):
    """Returns the sparsity (proportion of zero parameters) of a pytorch module"""
    all_weights = GetAllParams(module)
    N = all_weights.shape[0]  # weights array is flat
    sparsity = (N - all_weights.count_nonzero()) / N
    return sparsity.item()


def GetAllParams(module):
    """Return a flattened array of all learnable parameters of a model"""
    all_weights = []
    for layer_i in range(len(module)):
        for param in module[layer_i][0].parameters():
            all_weights.append(param.view(-1))

    return torch.cat(all_weights)


def ModuleFlops(module, input):
    flops = FlopCountAnalysis(module, input)
    return flops.total()


# Conv2d
def AnalyzeConv2d(layer_tuple):
    """
    Parameters
    ----------
    layer : torch.nn.modules.conv.Conv2d

    """
    assert (
        layer_tuple[0]._get_name() == "Conv2d"
    ), f"Layer must be a Conv2d but received {layer_tuple[0]}"

    indicators = [layer_tuple[0]._get_name()]
    indicators.append(layer_tuple[0].in_channels)
    indicators.append((layer_tuple[1][1], layer_tuple[1][2], layer_tuple[1][3]))
    indicators.append(layer_tuple[0].out_channels)
    indicators.append((layer_tuple[2][1], layer_tuple[2][3], layer_tuple[2][3]))
    indicators.append(Conv2dLearnableParameters(layer_tuple))
    indicators.append(Conv2dWeightVanishingCoefficients(layer_tuple[0]))
    indicators.append(Conv2dWeightVanishingRows(layer_tuple[0]))
    indicators.append(Conv2dWeightVanishingColumns(layer_tuple[0]))
    indicators.append(Conv2dBiasVanishingCoefficients(layer_tuple[0]))
    indicators.append(Conv2dWeightEntropy(layer_tuple[0]))
    indicators.append(Conv2dBiasEntropy(layer_tuple[0]))
    indicators.append(Conv2dMacc(layer_tuple))
    indicators.append(Conv2dReducedMacc(layer_tuple))
    indicators.append(Conv2dKB32(layer_tuple))
    # indicators.append(0)
    return indicators


def Conv2dMacc(layer_tuple):
    """

    Parameters
    ----------
    layer_tuple : (layer object, input_shape, output_shape)
    with
        - input_shape = (batch_size,C_in,x_in,y_in)
        - output_shape = (batch_size,C_out,x_out,y_out)

    Returns: the number of MACC given with K_x * K_y * C_in * C_out * x_out * y_out 
             
            
    """

    return (
        layer_tuple[0].kernel_size[0]
        * layer_tuple[0].kernel_size[1]
        * layer_tuple[0].out_channels
        * layer_tuple[0].in_channels
        * layer_tuple[2][2]
        * layer_tuple[2][3]
    )


def Conv2dReducedMacc(layer_tuple):
    """

    Parameters
    ----------
    layer_tuple : (layer object, input_shape, output_shape)
    with
        - input_shape = (batch_size,C_in,x_in,y_in)
        - output_shape = (batch_size,C_out,x_out,y_out)

    Returns: the reduced number of MACC given with K_x * K_y * C_in * C_out * x_out * y_out 
             
            
    """

    return (
        layer_tuple[0].kernel_size[0]
        * layer_tuple[0].kernel_size[1]
        * (layer_tuple[0].out_channels - Conv2dWeightVanishingRows(layer_tuple[0]))
        * (layer_tuple[0].in_channels - Conv2dWeightVanishingColumns(layer_tuple[0]))
        * layer_tuple[2][2]
        * layer_tuple[2][3]
    )


def Conv2dLearnableParameters(layer_tuple):
    """

    Parameters
    ----------
    layer_tuple : (layer object, input_shape, output_shape)
    with
        - input_shape = (batch_size,C_in,x_in,y_in)
        - output_shape = (batch_size,C_out,x_out,y_out)

    Returns: the number of learnable parameters: #weight + #bias
            
    """
    return layer_tuple[0].weight.nelement() + layer_tuple[0].bias.nelement()


def Conv2dReducedLearnableParameters(layer_tuple):
    """

    Parameters
    ----------
    layer_tuple : (layer object, input_shape, output_shape)
    with
        - input_shape = (batch_size,C_in,x_in,y_in)
        - output_shape = (batch_size,C_out,x_out,y_out)

    Returns: estimates the number of learnable parameters of the 
    projected layer
            
    """
    return (
        (layer_tuple[0].in_channels - Conv2dWeightVanishingColumns(layer_tuple[0]))
        * (layer_tuple[0].out_channels - Conv2dWeightVanishingRows(layer_tuple[0]))
        * layer_tuple[0].kernel_size[0]
        * layer_tuple[0].kernel_size[1]
        + layer_tuple[0].out_channels  # weight
        - Conv2dWeightVanishingRows(layer_tuple[0])
    )  # bias


def Conv2dWeightVanishingCoefficients(layer):
    """

    Parameters
    ----------
    layer : Conv2d object

    Returns: the number of vanishing coefficients in the weight
            
    """
    return torch.where(layer.weight == 0, 1, 0).sum().item()


def Conv2dWeightVanishingRows(layer):
    """

    Parameters
    ----------
    layer : Conv2d object

    Returns: the number of vanishing kernel rows in the weight
            
    """

    return sum(
        [int(torch.norm(layer.weight[i], p=1) == 0) for i in range(layer.out_channels)]
    )


def Conv2dWeightVanishingColumns(layer):
    """

    Parameters
    ----------
    layer : Conv2d object

    Returns: the number of vanishing kernel columns in the weight
            
    """

    return sum(
        [
            int(torch.norm(layer.weight[:, i], p=1) == 0)
            for i in range(layer.in_channels)
        ]
    )


def Conv2dBiasVanishingCoefficients(layer):
    """

    Parameters
    ----------
    layer : Conv2d object

    Returns: the number of vanishing coefficients in the bias
            
    """
    return torch.where(layer.bias == 0, 1, 0).sum().item()


def Conv2dWeightEntropy(layer):
    """

    Parameters
    ----------
    layer : Conv2d object

    Returns: entropy of the weight
            
    """

    return TensorEntropy(layer.weight)


def Conv2dBiasEntropy(layer):
    """

    Parameters
    ----------
    layer : Conv2d object

    Returns: entropy of the bias
            
    """

    return TensorEntropy(layer.bias)


def Conv2dKB32(layer_tuple):
    """

    Parameters
    ----------
    layer : Conv2d object

    Returns: evaluation the storage requierments in KB
            
    """

    return (
        32.0
        * Conv2dLearnableParameters(layer_tuple)
        * Conv2dWeightEntropy(layer_tuple[0])
        / 8000.0
    )


# ZeroPad2d


def AnalyzeZeroPad2d(layer_tuple):
    """
    Parameters
    ----------
    layer : torch.nn.modules.conv.Conv2d

    """
    assert (
        layer_tuple[0]._get_name() == "ZeroPad2d"
    ), f"Layer must be a ZeroPad2d by received {layer_tuple[0]}"

    indicators = [layer_tuple[0]._get_name()]
    # number of input channels
    indicators.append(layer_tuple[1][0])
    # input shape
    indicators.append((layer_tuple[1][1], layer_tuple[1][2], layer_tuple[1][3]))
    # number of output channels
    indicators.append(layer_tuple[2][0])
    # output shape
    indicators.append((layer_tuple[2][1], layer_tuple[2][3], layer_tuple[2][3]))
    # number of learnable parameters
    indicators.append(0)
    # number of weight vanishing coefficients
    indicators.append(0)
    # number of weight vanishing kernel columns
    indicators.append(0)
    # number of weight vanishing kernel rows
    indicators.append(0)
    # bias number of vanishing coefficients
    indicators.append(0)
    # weight entropy
    indicators.append(0)
    # bias entropy
    indicators.append(0)
    # MACC
    indicators.append(0)
    # Reduced MACC
    indicators.append(0)
    # KB32
    indicators.append(0)

    return indicators


# LeakyReLU


def AnalyzeLeakyReLU(layer_tuple):
    """
    Parameters
    ----------
    layer : torch.nn.modules.conv.Conv2d

    """
    assert (
        layer_tuple[0]._get_name() == "LeakyReLU"
    ), f"Layer must be a LeakyReLU by received {layer_tuple[0]}"

    indicators = [layer_tuple[0]._get_name()]
    # number of input channels
    indicators.append(layer_tuple[1][0])
    # input shape
    indicators.append((layer_tuple[1][1], layer_tuple[1][2], layer_tuple[1][3]))
    # number of output channels
    indicators.append(layer_tuple[2][0])
    # output shape
    indicators.append((layer_tuple[2][1], layer_tuple[2][3], layer_tuple[2][3]))
    # number of learnable parameters
    indicators.append(0)
    # number of weight vanishing coefficients
    indicators.append(0)
    # number of weight vanishing kernel columns
    indicators.append(0)
    # number of weight vanishing kernel rows
    indicators.append(0)
    # bias number of vanishing coefficients
    indicators.append(0)
    # weight entropy
    indicators.append(0)
    # bias entropy
    indicators.append(0)
    # MACC
    indicators.append(0)
    # Reduced MACC
    indicators.append(0)
    # KB32
    indicators.append(0)

    return indicators


# Tanh


def AnalyzeTanh(layer_tuple):
    """
    Parameters
    ----------
    layer : torch.nn.modules.conv.Conv2d

    """
    assert (
        layer_tuple[0]._get_name() == "Tanh"
    ), f"Layer must be a Tanh by received {layer_tuple[0]}"

    indicators = [layer_tuple[0]._get_name()]
    # number of input channels
    indicators.append(layer_tuple[1][0])
    # input shape
    indicators.append((layer_tuple[1][1], layer_tuple[1][2], layer_tuple[1][3]))
    # number of output channels
    indicators.append(layer_tuple[2][0])
    # output shape
    indicators.append((layer_tuple[2][1], layer_tuple[2][3], layer_tuple[2][3]))
    # number of learnable parameters
    indicators.append(0)
    # number of weight vanishing coefficients
    indicators.append(0)
    # number of weight vanishing kernel columns
    indicators.append(0)
    # number of weight vanishing kernel rows
    indicators.append(0)
    # bias number of vanishing coefficients
    indicators.append(0)
    # weight entropy
    indicators.append(0)
    # bias entropy
    indicators.append(0)
    # MACC
    indicators.append(0)
    # Reduced MACC
    indicators.append(0)
    # KB32
    indicators.append(0)

    return indicators


# ReflectionPad2d
def AnalyzeReflectionPad2d(layer_tuple):
    """
    Parameters
    ----------
    layer : torch.nn.modules.conv.Conv2d

    """
    assert (
        layer_tuple[0]._get_name() == "ReflectionPad2d"
    ), f"Layer must be a ReflectionPad2d by received {layer_tuple[0]}"

    indicators = [layer_tuple[0]._get_name()]
    # number of input channels
    indicators.append(layer_tuple[1][0])
    # input shape
    indicators.append((layer_tuple[1][1], layer_tuple[1][2], layer_tuple[1][3]))
    # number of output channels
    indicators.append(layer_tuple[2][0])
    # output shape
    indicators.append((layer_tuple[2][1], layer_tuple[2][3], layer_tuple[2][3]))
    # number of learnable parameters
    indicators.append(0)
    # number of weight vanishing coefficients
    indicators.append(0)
    # number of weight vanishing kernel columns
    indicators.append(0)
    # number of weight vanishing kernel rows
    indicators.append(0)
    # bias number of vanishing coefficients
    indicators.append(0)
    # weight entropy
    indicators.append(0)
    # bias entropy
    indicators.append(0)
    # MACC
    indicators.append(0)
    # Reduced MACC
    indicators.append(0)
    # KB32
    indicators.append(0)

    return indicators


# Analysis of TransposeConv2d layers
def AnalyzeConvTranspose2d(layer_tuple):

    assert (
        layer_tuple[0]._get_name() == "ConvTranspose2d"
    ), f"Layer must be a ConvTranspose2d by received {layer_tuple[0]}"

    indicators = [layer_tuple[0]._get_name()]
    # number of input channels
    indicators.append(layer_tuple[1][0])
    # input shape
    indicators.append((layer_tuple[1][1], layer_tuple[1][2], layer_tuple[1][3]))
    # number of output channels
    indicators.append(layer_tuple[2][0])
    # output shape
    indicators.append((layer_tuple[2][1], layer_tuple[2][3], layer_tuple[2][3]))
    # number of learnable parameters
    indicators.append(ConvTranspose2dLearnableParameters(layer_tuple[0]))
    # number of weight vanishing coefficients
    indicators.append(ConvTranspose2dWeightVanishingCoefficients(layer_tuple[0]))
    # number of weight vanishing kernel columns
    indicators.append(ConvTranspose2dWeightVanishingColumns(layer_tuple[0]))
    # number of weight vanishing kernel rows
    indicators.append(ConvTranspose2dWeightVanishingRows(layer_tuple[0]))
    # bias number of vanishing coefficients
    indicators.append(ConvTranspose2dBiasVanishingCoefficients(layer_tuple[0]))
    # weight entropy
    indicators.append(ConvTranspose2dWeightEntropy(layer_tuple[0]))
    # bias entropy
    indicators.append(ConvTranspose2dBiasEntropy(layer_tuple[0]))
    # MACC
    indicators.append(ConvTranspose2dMacc(layer_tuple))
    # Reduced MACC
    indicators.append(ConvTranspose2dReducedMacc(layer_tuple))
    # KB32
    indicators.append(ConvTranspose2dKB32(layer_tuple))

    return indicators


def ConvTranspose2dLearnableParameters(layer):
    """

    Parameters
    ----------
    layer_tuple : (layer object, input_shape, output_shape)
    with
        - input_shape = (batch_size,C_in,x_in,y_in)
        - output_shape = (batch_size,C_out,x_out,y_out)

    Returns: the number of learnable parameters: #weight + #bias
            
    """

    return layer.weight.nelement() + layer.bias.nelement()


def ConvTranspose2dWeightVanishingCoefficients(layer):
    """

    Parameters
    ----------
    layer : ConvTranspose2d object

    Returns: the number of vanishing coefficients in the weight
            
    """
    return torch.where(layer.weight == 0, 1, 0).sum().item()


def ConvTranspose2dBiasVanishingCoefficients(layer):
    """

    Parameters
    ----------
    layer : ConvTranspose2d object

    Returns: the number of vanishing coefficients in the bias
            
    """
    return torch.where(layer.bias == 0, 1, 0).sum().item()


def ConvTranspose2dWeightVanishingRows(layer):
    """

    Parameters
    ----------
    layer : ConvTranspose2d object

    Returns: the number of vanishing kernel rows in the weight
            
    """

    return sum(
        [int(torch.norm(layer.weight[i], p=1) == 0) for i in range(layer.in_channels)]
    )


def ConvTranspose2dWeightVanishingColumns(layer):
    """

    Parameters
    ----------
    layer : ConvTranspose2d object

    Returns: the number of vanishing kernel columns in the weight
            
    """

    return sum(
        [
            int(torch.norm(layer.weight[:, i], p=1) == 0)
            for i in range(layer.out_channels)
        ]
    )


def ConvTranspose2dBiasVanishingCoefficients(layer):
    """

    Parameters
    ----------
    layer : ConvTranspose2d object

    Returns: the number of vanishing coefficients in the bias
            
    """
    return torch.where(layer.bias == 0, 1, 0).sum().item()


def ConvTranspose2dWeightEntropy(layer):
    """

    Parameters
    ----------
    layer : TransposeConv2d object

    Returns: entropy of the weight
            
    """

    return TensorEntropy(layer.weight)


def ConvTranspose2dBiasEntropy(layer):
    """

    Parameters
    ----------
    layer : TransposeConv2d object

    Returns: entropy of the bias
            
    """

    return TensorEntropy(layer.bias)


def ConvTranspose2dMacc(layer_tuple):
    """

    Parameters
    ----------
    layer_tuple : (layer object, input_shape, output_shape)
    with
        - input_shape = (batch_size,C_in,x_in,y_in)
        - output_shape = (batch_size,C_out,x_out,y_out)

    Returns: the number of MACC given with K_x * K_y * C_in * C_out * x_in * y_in 
             
            
    """

    return (
        layer_tuple[0].kernel_size[0]
        * layer_tuple[0].kernel_size[1]
        * layer_tuple[0].out_channels
        * layer_tuple[0].in_channels
        * layer_tuple[1][2]
        * layer_tuple[1][3]
    )


def ConvTranspose2dReducedMacc(layer_tuple):
    """

    Parameters
    ----------
    layer_tuple : (layer object, input_shape, output_shape)
    with
        - input_shape = (batch_size,C_in,x_in,y_in)
        - output_shape = (batch_size,C_out,x_out,y_out)

    Returns: the reduced number of MACC given with K_x * K_y * C_in * C_out * x_out * y_out 
             
            
    """

    return (
        layer_tuple[0].kernel_size[0]
        * layer_tuple[0].kernel_size[1]
        * (
            layer_tuple[0].out_channels
            - ConvTranspose2dWeightVanishingRows(layer_tuple[0])
        )
        * (
            layer_tuple[0].in_channels
            - ConvTranspose2dWeightVanishingColumns(layer_tuple[0])
        )
        * layer_tuple[1][2]
        * layer_tuple[1][3]
    )


def ConvTranspose2dKB32(layer_tuple):

    """

    Parameters
    ----------
    layer : Conv2d object

    Returns: evaluation the storage requirements in KB
            
    """

    return (
        32.0
        * ConvTranspose2dLearnableParameters(layer_tuple[0])
        * ConvTranspose2dWeightEntropy(layer_tuple[0])
        / 8000.0
    )


### to be suppressed

# MA = CAEModelAnalyzer()
# MA.load_model("../experiments/trainComp_40_200/checkpoint/best_model_Mask.pth")

