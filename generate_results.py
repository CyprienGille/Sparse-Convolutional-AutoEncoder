# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:56:00 2022


Example for the usage of CAEModelAnalyzer
Will generate a csv with the characteristics of a given model

@author: YFGI6212
"""
#%%
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import math

pd.set_option("display.max_columns", 100)
pd.set_option("expand_frame_repr", True)

import matplotlib.pyplot as plt
import seaborn as sns
from functions.logger import Logger

from model_analyzer import CAEModelAnalyzer

logger = Logger(__name__, colorize=True)

Config = {}
Config["root_dir"] = "experiments"
Config["model_dir"] = "checkpoint"
Config["result_dir"] = "results"
Config["figures_dir"] = "figures"
Config["beta"] = "0.0"
Config["eta"] = "0.5"
Config["initial_model"] = [
    "trainCompSWA_100_0.0_Flickr_initial",
    "best",
    "initial",
]
Config["proj_type"] = "L1inf"
Config["projected_models"] = [
    [
        f"trainComp_100_{Config['beta']}_fullproj_Flickr_{Config['proj_type']}_{Config['eta']}",
        Config["proj_type"],
        f"_{Config['eta']}",
    ],
]

Config["analyzed_layers"] = ["Conv2d", "ConvTranspose2d"]


def generateResults(
    conf=Config,
    save_analysis: bool = True,
    save_figures: bool = False,
    use_latex: bool = False,
):

    logger.info(
        f"\n==========================================================================="
    )
    logger.info(f'Generates results and graphics for {conf["initial_model"]}')

    models = [k for k in conf["projected_models"]]
    models = [conf["initial_model"]] + models

    logger.info(f" -Comparisons of {models}")

    # generate results csv
    experiment_path = pathlib.Path(f'D:/I3S/{conf["root_dir"]}')

    logger.warning(f"Generates results ==========================")

    model_path = experiment_path / conf["model_dir"]

    if save_analysis:
        for k in models:
            logger.info(f"     - Generates results for {k[0]}")
            # model_path = pathlib.Path(
            #     f'{experiment_path}/{k[0]}/{conf["model_dir"]}/best_model.pth'
            # )
            model_path = f'{experiment_path}/{k[0]}/{conf["model_dir"]}/best_model.pth'
            logger.warning(f"        - loads {model_path}")
            MA = CAEModelAnalyzer()
            MA.load_model(model_path)
            Best_encoder = MA.analyze_encoder()
            Best_decoder = MA.analyze_decoder()

            save_dir = pathlib.Path(f'{experiment_path}/{k[0]}/{conf["result_dir"]}')

            if not save_dir.exists():
                save_dir.mkdir()

            logger.warning(f" - CSV results saved in {save_dir}")

            Best_encoder.to_csv(save_dir / f"encoder_{k[1]}{k[2]}.csv", sep=";")
            Best_decoder.to_csv(save_dir / f"decoder_{k[1]}{k[2]}.csv", sep=";")

    # generates figures
    if save_figures:

        experiment = conf["initial_model"][0]
        save_dir = pathlib.Path(f'{experiment_path}/{experiment}/{conf["figures_dir"]}')
        print(f"\b - Generates figures saved in {save_dir}")
        if not save_dir.exists():
            save_dir.mkdir()

        conf["models"] = models
        makeEntropyGraphics(experiment, conf=conf, use_latex=use_latex)


def makeEntropyGraphics(experiment, conf=Config, use_latex=False):

    root_dir = pathlib.Path(f'../{conf["root_dir"]}')
    result_dir = pathlib.Path(f'../{conf["root_dir"]}') / conf["result_dir"]

    figures_dir = (
        pathlib.Path(f'../{conf["root_dir"]}') / experiment / conf["figures_dir"]
    )

    # MACCs
    allResults = []
    for k in conf["models"]:
        results_dir = pathlib.Path(f'../{conf["root_dir"]}/{k[0]}') / conf["result_dir"]
        results_encoder = pd.read_csv(
            root_dir / k[0] / "results" / f"encoder_{k[1]}{k[2]}.csv", sep=";"
        )
        results_encoder[["KB32"]] = round(results_encoder[["KB32"]]) / 1000.0
        results_encoder = results_encoder[results_encoder["type"] != "General"]
        results_encoder["part"] = ["encoder" for i in range(results_encoder.shape[0])]
        results_decoder = pd.read_csv(
            root_dir / k[0] / "results" / f"decoder_{k[1]}{k[2]}.csv", sep=";"
        )
        results_decoder[["KB32"]] = round(results_decoder[["KB32"]]) / 1000.0
        results_decoder = results_decoder[results_decoder["type"] != "General"]
        results_decoder["part"] = ["decoder" for i in range(results_decoder.shape[0])]
        results = pd.concat(
            [results_encoder, results_decoder], axis=0, ignore_index=True, sort=False
        )
        results["model"] = [k[1] for i in range(results.shape[0])]

        filtered_results = results[
            ["type", "reduced MACC", "model", "part", "weight_ent", "KB32"]
        ]

        filtered_results = filtered_results[results.type.isin(conf["analyzed_layers"])]
        allResults.append(filtered_results)

    allResults = pd.concat(allResults, axis=0, ignore_index=True, sort=False)
    allResults["reduced MACC"] = allResults["reduced MACC"] / 10 ** 8
    ### Encoder graphics

    dfs = []

    for k in conf["models"]:
        enc_graph = renameType(
            allResults[(allResults.part == "encoder") & (allResults.model == k[1])],
            use_latex=use_latex,
        )
        dec_graph = renameType(
            allResults[(allResults.part == "decoder") & (allResults.model == k[1])],
            use_latex=use_latex,
        )
        dfs.append(enc_graph)
        dfs.append(dec_graph)

    all_graphs = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    plt.clf()

    if use_latex:
        sns.set(rc={"text.usetex": True})
        Labels = ["initial", "$\ell_{1}$", "$\ell_{11}$"]
    else:
        Labels = ["initial", conf["proj_type"], "placeholder"]

    sns.set_style("whitegrid")

    # encoder MACC
    encoder_macc = sns.barplot(
        x="type",
        y="reduced MACC",
        hue="model",
        ci=None,
        data=all_graphs[all_graphs.part == "encoder"][
            ["type", "reduced MACC", "model", "part"]
        ],
    )

    if use_latex:
        encoder_macc.set(xlabel="layer", ylabel="MACC (x $10^{8}$)")
    else:
        encoder_macc.set(xlabel="layer", ylabel="MACC (x 10e+8)")

    encoder_macc.set(ylim=(0, 7))
    plt.ticklabel_format(style="plain", axis="y")

    plt.legend(labels=Labels, loc="upper right", framealpha=0.8)
    plt.title(f'Encoder MACC eta={conf["eta"]}')
    plt.savefig(f'{figures_dir}/encoders_macc_{conf["eta"]}.png')
    plt.clf()

    # decoder MACC
    decoder_macc = sns.barplot(
        x="type",
        y="reduced MACC",
        hue="model",
        ci=None,
        data=all_graphs[all_graphs.part == "decoder"],
    )

    if use_latex:
        decoder_macc.set(xlabel="layer", ylabel="MACC (x $10^{-8}$)")
    else:
        decoder_macc.set(xlabel="layer", ylabel="MACC (x 10e+8)")

    decoder_macc.set(ylim=(0, 7))
    plt.ticklabel_format(style="plain", axis="y")

    plt.legend(loc="upper right", labels=Labels, framealpha=0.8)

    plt.title(f'Decoder MACC  eta={conf["eta"]}')
    plt.savefig(f'{figures_dir}/decoders_macc_{conf["eta"]}.png')
    plt.clf()

    # encoder entropy
    encoder_entropy = sns.barplot(
        x="type",
        y="weight_ent",
        hue="model",
        ci=None,
        data=all_graphs[all_graphs.part == "encoder"][
            ["type", "weight_ent", "model", "part"]
        ],
    )
    encoder_entropy.set(xlabel="layer", ylabel="entropy")
    plt.ticklabel_format(style="plain", axis="y")
    encoder_entropy.set(ylim=(0, 20))
    plt.legend(loc="upper right", labels=Labels, framealpha=0.8)
    plt.title(f'Encoder entropy eta={conf["eta"]}')
    plt.savefig(f'{figures_dir}/encoders_entropy_{conf["eta"]}.png')
    plt.clf()

    # decoder entropy
    decoder_entropy = sns.barplot(
        x="type",
        y="weight_ent",
        hue="model",
        ci=None,
        data=all_graphs[all_graphs.part == "decoder"],
    )
    decoder_entropy.set(xlabel="layer", ylabel="entropy")
    decoder_entropy.set(ylim=(0, 20))
    plt.ticklabel_format(style="plain", axis="y")
    plt.legend(loc="upper right", labels=Labels, framealpha=0.8)
    plt.title(f'Decoder entropy  eta={conf["eta"]}')
    plt.savefig(f'{figures_dir}/decoders_entropy_{conf["eta"]}.png')
    plt.clf()

    # encoder memory
    encoder_entropy = sns.barplot(
        x="type",
        y="KB32",
        hue="model",
        ci=None,
        data=all_graphs[all_graphs.part == "encoder"][
            ["type", "KB32", "model", "part"]
        ],
    )
    encoder_entropy.set(xlabel="layer", ylabel="size (Mb)")
    plt.ticklabel_format(style="plain", axis="y")
    encoder_entropy.set(ylim=(0, 14))
    plt.legend(loc="upper right", labels=Labels, framealpha=0.8)
    plt.title(f'Encoder memory size  eta={conf["eta"]}')
    plt.savefig(f'{figures_dir}/encoders_memory_{conf["eta"]}.png')
    plt.clf()

    # decoder memory
    decoder_entropy = sns.barplot(
        x="type",
        y="KB32",
        hue="model",
        ci=None,
        data=all_graphs[all_graphs.part == "decoder"],
    )
    decoder_entropy.set(xlabel="layer", ylabel="size (Mb)")
    decoder_entropy.set(ylim=(0, 14))
    plt.ticklabel_format(style="plain", axis="y")
    plt.legend(loc="upper right", labels=Labels, framealpha=0.8)
    plt.title(f'Decoder memory size  eta={conf["eta"]}')
    plt.savefig(f'{figures_dir}/decoders_memory_{conf["eta"]}.png')
    plt.clf()

    return allResults


def renameType(df, names={"Conv2d": "C", "ConvTranspose2d": "TC"}, use_latex=True):

    df2 = df
    Types = list(df["type"])
    New_Types = []

    for i in range(len(Types)):
        if use_latex:
            New_Types.append(f"{names[Types[i]]}" + "$_{" + f"{i}" + "}$")
        else:
            New_Types.append(f"{names[Types[i]]}_{i}")
    df2["type"] = New_Types

    return df2


def makeAccuracyGraphics(conf=Config, eta_list=None, use_latex=False):

    root_dir = pathlib.Path(f'../{conf["root_dir"]}')

    figures_dir = pathlib.Path(
        f'../{conf["root_dir"]}/{conf["initial_model"][0]}/{conf["figures_dir"]}'
    )

    etas = eta_list
    etas.sort()

    initial_model = f'{root_dir}/{conf["initial_model"][0]}/summary/model_{conf["beta"]}_initial_summary.csv'
    initial_data = pd.read_csv(initial_model, sep=",")

    data = initial_data.loc[
        initial_data["validation_loss"] == initial_data["validation_loss"].min()
    ]
    data["model"] = "initial"
    data["eta"] = 400

    initial_loss = initial_data["validation_loss"].min()

    for eta in etas:

        l1_model = f'{root_dir}/{conf["projected_models"]}_{conf["beta"]}_L1_{eta}/summary/model_{conf["beta"]}_L1_{eta}_summary.csv'
        l1_data = pd.read_csv(l1_model, sep=",")

        datal1 = l1_data.loc[
            l1_data["validation_loss"] == l1_data["validation_loss"].min()
        ]
        datal1["model"] = "l1"
        datal1["eta"] = eta
        data = pd.concat((data, datal1), axis=0)

        l11_model = f'{root_dir}/{conf["projected_models"]}_{conf["beta"]}_L11_{eta}/summary/model_{conf["beta"]}_L11_{eta}_summary.csv'
        l11_data = pd.read_csv(l11_model, sep=",")
        datal11 = l11_data.loc[
            l11_data["validation_loss"] == l11_data["validation_loss"].min()
        ]
        datal11["model"] = "l11"
        datal11["eta"] = eta
        data = pd.concat((data, datal11), axis=0)

    initial_loss = data.iloc[0]["validation_loss"]
    data["PSNR"] = 10 * (math.log((255 ** 2) / initial_loss, 10)) - 10 * np.log10(
        (255 ** 2) / data["validation_loss"]
    )

    datal1 = data[data["model"] == "l1"]
    datal11 = data[data["model"] == "l11"]

    datal1.to_csv("dataL1.csv")
    datal11.to_csv("dataL11.csv")

    fig, ax = plt.subplots()

    if use_latex:
        ax.set_xlabel(r"projection radius $\eta$")
        ax.set_ylabel(r"$\Delta$PSNR")

        ax.plot(datal1["eta"], datal1["PSNR"], ".-", label=r"$\ell_1$")
        ax.plot(datal11["eta"], datal11["PSNR"], ".-", label=r"$\ell_{11}$")
        ax.set_title(
            r"$\Delta$PSNR relatively to the non-projected model"
        )  # Add a title to the axes.
    else:
        ax.set_xlabel("projection radius eta")
        ax.set_ylabel("Delta PSNR")

        ax.plot(datal1["eta"], datal1["PSNR"], ".-", label="l1")
        ax.plot(datal11["eta"], datal11["PSNR"], ".-", label="l11")
        ax.set_title(
            "Delta PSNR relatively to the non-projected model"
        )  # Add a title to the axes.
        ax.set_xlim(150, 300)
        plt.xticks([150, 175, 200, 225, 250, 275, 300])

    ax.legend()

    plt.savefig(figures_dir / "Losses.png")

    return ()


generateResults(use_latex=False)
pathlib.PosixPath = temp

# %%
