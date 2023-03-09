"""
Script to train a CAE from scratch

One should pay attention to:
 - (l.144-155) the experiment's nickname, which will inform the results folder name
 - (l.483-523) the experiment's parameters, namely the number of epochs and the data directory
 - (l.544-545) the projections and projections parameters to use for the second descent
"""

#%%
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import sys

sys.path.append("../")
sys.path.append("../../")

import numpy as np
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from functions.data_loader import ImageFolder720p, FlickrFolder

from functions.logger import Logger

from models.cae_32x32x32_zero_pad_comp import CAE
from models.masked_models import Masked_CAE


import util.reproducibility as reproducible

from recorders.recorder import Simple_Recorder
import projection.projection as proj
import functions.retrain_model as retrain_model

logger = Logger(__name__, colorize=True)

DEBUG = False

#%%
# Define functions
# Entropy for a collection of values
def compute_entropy(tensor_data):
    min_val = tensor_data.min()
    max_val = tensor_data.max()
    nb_bins = torch.round(max_val - min_val).item() + 1
    # nb_bins = torch.round(torch.Tensor(max_val - min_val + 1)).item()
    hist = torch.histc(
        tensor_data, bins=int(nb_bins), min=min_val.item(), max=max_val.item()
    )
    hist_prob = hist / hist.sum()
    hist_prob[hist_prob == 0] = 1
    entropy = -(hist_prob * torch.log2(hist_prob)).sum()

    return entropy


def compute_binary_entropy(tensor_data):
    """ On the latent space, tensor are binary. We can therefore 
        easily determine de entropy
    """

    q = torch.round(tensor_data)
    p = q.sum() / q.nelement()

    return -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))


def train_model(cfg, retrain=False):

    assert cfg["device"] == "cpu" or (
        cfg["device"] == "cuda" and torch.cuda.is_available()
    )

    # ensure reproducibility
    reproducible.seed_all(cfg["random_seed"], cuda=cfg["device"] == "cuda")

    logger.info(f'Initial model for experiment {cfg["exp_name"]}')

    # directories management
    logger.info(f"  -Check configuration directories")

    root_dir = Path(__file__).resolve().parents[1]
    exp_dir = root_dir / "experiments" / cfg["exp_name"]

    # add the exp_dir to config for recorder purpose

    cfg["expe_dir"] = exp_dir

    # create model
    logger.info(f'  -Create model {cfg["model_class"]}')

    if retrain:
        assert (
            "model_file" in cfg["retrain"]
        ), 'configuration["retrain"]["model_file"] must be defined'
        assert (
            "initial_model_file" in cfg["retrain"]
        ), 'configuration["retrain"]["initial_model_file"] must be defined'
        assert (
            "projection" in cfg["retrain"]
        ), 'configuration["retrain"]["projection"] must be defined'
        assert (
            "projection_params" in cfg["retrain"]
        ), 'configuration["retrain"]["projection_params"] must be defined'

        logger.warning(
            f'       -Retrain model {cfg["retrain"]["model_file"]} with projection {cfg["retrain"]["projection"]} and parameters {cfg["retrain"]["projection_params"]}'
        )

        if cfg["retrain"]["projection"] in ["l1", "L1"]:
            projector = proj.proj_l1ball
            nickname = "L1"
        elif cfg["retrain"]["projection"] in ["l11", "L11"]:
            projector = proj.proj_l11ball
            nickname = "L11"
        elif cfg["retrain"]["projection"] in ["l21", "L21"]:
            projector = proj.proj_l21ball
            nickname = "L21"
        elif cfg["retrain"]["projection"] in ["l1inf", "L1inf"]:
            projector = proj.proj_l1infball
            nickname = "L1inf"
        elif cfg["retrain"]["projection"] in ["sparse", "threshold"]:
            projector = proj.sparse_global
            nickname = "threshold"

        else:
            raise ValueError(f'Projection {cfg["projection"]} not defined')

        model = retrain_model.load_projected_model(
            model_file=cfg["retrain"]["model_file"],
            initial_model_file=cfg["retrain"]["initial_model_file"],
            projection=projector,
            projection_params=cfg["retrain"]["projection_params"],
        )

        model.apply_masks_on_layers()

        if nickname == "threshold":
            cfg[
                "nickname"
            ] = f'fullproj_Flickr_{nickname}_{cfg["retrain"]["projection_params"]["fraction"]}'
        else:
            cfg[
                "nickname"
            ] = f'fullproj_Flickr_{nickname}_{cfg["retrain"]["projection_params"]["eta"]}'

        cfg[
            "exp_name"
        ] = f'trainComp_{cfg["num_epochs"]}_{cfg["beta"]}_{cfg["nickname"]}'
        exp_dir = root_dir / "experiments" / cfg["exp_name"]
        cfg["expe_dir"] = exp_dir
        logger.warning(f"       -Retrained models saved in {exp_dir}")

    else:
        logger.info(f'       - Initial training for model {cfg["model_class"]}')
        logger.warning(f"       -Models saved in {exp_dir}")
        model = cfg["model_class"]()

    model.to(cfg["device"])

    # save directory
    for d in ["out", "checkpoint", "logs", "summary"]:
        os.makedirs(exp_dir / d, exist_ok=True)

    # create optimizer
    logger.info(
        f'  -Create optimizer: class:{cfg["optimizer_class"]}, params: {cfg["optimizer_params"]}'
    )
    logger.info(f'  -Create scheduler:{cfg["scheduler_params"]}')

    optimizer = cfg["optimizer_class"](model.parameters(), **cfg["optimizer_params"])
    if cfg["scheduler_params"]["type"] == "theis":
        update_func = (
            lambda epoch: (
                cfg["scheduler_params"]["tau"]
                / (cfg["scheduler_params"]["tau"] + epoch)
            )
            ** cfg["scheduler_params"]["kappa"]
        )
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=update_func, verbose=True
        )

        def scheduler_step(*args):
            scheduler.step()

    elif cfg["scheduler_params"]["type"] == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg["scheduler_params"]["factor"],
            patience=cfg["scheduler_params"]["patience"],
            verbose=True,
        )

        def scheduler_step(val_loss):
            scheduler.step(val_loss)

    else:
        # scheduler = optim.lr_scheduler.LambdaLR(
        #     optimizer, lr_lambda=lambda epoch: 1, verbose=False
        # )  # identity scheduler
        def scheduler_step(*args):
            return None

    # create reconstruction loss
    logger.info(
        f'  -Create reconstruction loss: class: {cfg["reconstruction_loss_class"]}, params: {cfg["reconstruction_loss_params"]}'
    )
    logger.info(
        f'    -Reconstruction loss parameters {cfg["reconstruction_loss_params"]}'
    )
    loss_criterion = cfg["reconstruction_loss_class"](
        **cfg["reconstruction_loss_params"]
    )

    # create dataloaders
    logger.info(f"  -Create dataloaders")

    if cfg["device"] == "cuda":
        pin_memory = True
    else:
        pin_memory = False

    # Training_dataloader = DataLoader(
    #     dataset=ImageFolder720p(
    #         root=cfg["dataset_path"], files_list=cfg["train_files"]
    #     ),
    #     batch_size=cfg["batch_size"],
    #     shuffle=cfg["shuffle"],
    #     num_workers=cfg["num_workers"],
    #     worker_init_fn=reproducible.seed_worker,
    #     pin_memory=pin_memory,
    # )
    Training_dataloader = DataLoader(
        dataset=FlickrFolder(root=cfg["dataset_path"], files_list=cfg["train_files"],),
        batch_size=cfg["batch_size"],
        shuffle=cfg["shuffle"],
        num_workers=cfg["num_workers"],
        worker_init_fn=reproducible.seed_worker,
        pin_memory=pin_memory,
    )
    logger.info(
        f'    -Training: instances {len(cfg["train_files"])}, batch_size: {cfg["batch_size"]}, workers: {cfg["num_workers"]}, pin: {pin_memory}'
    )

    # Validation_dataloader = DataLoader(
    #     dataset=ImageFolder720p(root=cfg["dataset_path"], files_list=cfg["test_files"]),
    #     batch_size=cfg["validation_batch_size"],
    #     shuffle=cfg["shuffle"],
    #     num_workers=cfg["num_workers"],
    #     worker_init_fn=reproducible.seed_worker,
    #     pin_memory=pin_memory,
    # )
    Validation_dataloader = DataLoader(
        dataset=FlickrFolder(root=cfg["dataset_path"], files_list=cfg["test_files"],),
        batch_size=cfg["validation_batch_size"],
        shuffle=cfg["shuffle"],
        num_workers=cfg["num_workers"],
        worker_init_fn=reproducible.seed_worker,
        pin_memory=pin_memory,
    )

    logger.info(
        f'    -Validation: instances {len(cfg["test_files"])}, batch_size: {cfg["validation_batch_size"]}, workers: {cfg["num_workers"]}, pin: {pin_memory}'
    )

    # create recorder
    logger.info(f"  -Create recorder")
    recorder = Simple_Recorder(model, optimizer, cfg)

    ## Performance of initial model
    logger.warning(f" ")
    logger.info(f"Evaluate initial model performance")

    validation = validate_model(
        model, loss_criterion, 0, cfg["num_epochs"], Validation_dataloader
    )

    recorder.update(
        0,
        validation["loss"],
        validation["entropy"],
        validation["loss"],
        validation["entropy"],
    )

    logger.info(
        "[%3d/%3d] validation loss: %.8f and validation entropy: %.8f"
        % (0, cfg["num_epochs"], validation["loss"], validation["entropy"])
    )

    logger.info(f"==== Start Training ====")

    for epoch in range(cfg["num_epochs"]):

        training_perf = train_epoch(
            model,
            loss_criterion,
            optimizer,
            epoch + 1,
            cfg["num_epochs"],
            Training_dataloader,
        )
        validation_perf = validate_model(
            model, loss_criterion, epoch + 1, cfg["num_epochs"], Validation_dataloader
        )
        scheduler_step(validation_perf["loss"])

        recorder.update(
            epoch + 1,
            training_perf["loss"],
            training_perf["entropy"],
            validation_perf["loss"],
            validation_perf["entropy"],
        )
        logger.info(
            "[%3d/%3d] TRAINING: loss: %.8f, entropy: %.8f  VALIDATION loss: %.8f,  entropy: %.8f"
            % (
                epoch + 1,
                cfg["num_epochs"],
                training_perf["loss"],
                training_perf["entropy"],
                validation_perf["loss"],
                validation_perf["entropy"],
            )
        )


def validate_model(model, loss_criterion, epoch, total_epoch, dataloader):
    """ Validation of a model on a test set """

    epoch_avg = 0.0

    model.eval()

    avg_loss = 0.0
    avg_entropy = 0.0

    for batch_idx, data in enumerate(dataloader):

        img, patches, (nb_pat_x, nb_pat_y) = data
        tot_nb_pat = nb_pat_x[0].item() * nb_pat_y[0].item()
        patches = patches.to(cfg["device"], non_blocking=True)

        avg_loss_per_image = 0.0
        avg_entropy_per_image = 0.0
        for i in range(nb_pat_y[0].item()):

            for j in range(nb_pat_x[0].item()):

                x = patches[:, :, i, j, :, :]
                x_quantized, y = model(x)

                entropy = compute_entropy(x_quantized)
                loss = loss_criterion(x, y) + cfg["beta"] * entropy.detach()

                avg_entropy_per_image += (1 / tot_nb_pat) * entropy.item()
                avg_loss_per_image += (1 / tot_nb_pat) * loss.item()

        avg_loss += avg_loss_per_image
        epoch_avg += avg_loss_per_image
        avg_entropy += avg_entropy_per_image

        if DEBUG:
            logger.debug(
                "[%3d/%3d][%5d/%5d] Validation: cumulated loss: %.8f and  cumulated entropy: %.8f"
                % (
                    epoch,
                    total_epoch,
                    batch_idx + 1,
                    dataloader.__len__(),
                    avg_loss,
                    avg_entropy,
                )
            )

    avg_loss = avg_loss / dataloader.dataset.__len__()
    avg_entropy = avg_entropy / dataloader.dataset.__len__()

    if DEBUG:
        logger.info(
            "[%3d/%3d] Validation: average loss: %.8f and average entropy: %.8f"
            % (epoch, total_epoch, avg_loss, avg_entropy)
        )
    return {"epoch": epoch, "loss": avg_loss, "entropy": avg_entropy}


def train_epoch(model, loss_criterion, optimizer, epoch, total_epoch, dataloader):
    """ Validation of a model on a test set """

    avg_loss = 0.0
    avg_entropy = 0.0

    model.train()

    if isinstance(model, Masked_CAE):
        MASKED = True
    else:
        MASKED = False

    for batch_idx, data in enumerate(dataloader):

        _, patches, (nb_pat_x, nb_pat_y) = data
        tot_nb_pat = nb_pat_x[0].item() * nb_pat_y[0].item()
        patches = patches.to(cfg["device"], non_blocking=True)

        avg_loss_per_image = 0.0
        avg_entropy_per_image = 0.0

        for i in range(nb_pat_y[0].item()):

            for j in range(nb_pat_x[0].item()):
                optimizer.zero_grad()
                x = patches[:, :, i, j, :, :]
                x_encoded, y = model(x)

                entropy = compute_entropy(x_encoded)
                # entropy = compute_binary_entropy(x_encoded)
                loss = loss_criterion(x, y) + cfg["beta"] * entropy.detach()

                avg_entropy_per_image += entropy.item()
                avg_loss_per_image += loss.item()
                loss.backward()

                # Apply mask when necessary
                if MASKED:
                    model.mask_grad()

                optimizer.step()
        avg_loss += avg_loss_per_image / tot_nb_pat
        avg_entropy += avg_entropy_per_image / tot_nb_pat

        if DEBUG:
            logger.debug(
                "[%3d/%3d][%5d/%5d] Training: cumulated loss: %.8f and  cumulated entropy: %.8f"
                % (
                    epoch,
                    total_epoch,
                    batch_idx + 1,
                    dataloader.__len__(),
                    avg_loss,
                    avg_entropy,
                )
            )

        avg_loss = avg_loss / dataloader.batch_size
        avg_entropy = avg_entropy / dataloader.batch_size

    if DEBUG:
        logger.info(
            "[%3d/%3d] Training: average loss: %.8f and average entropy: %.8f"
            % (epoch, total_epoch, avg_loss, avg_entropy)
        )
    return {"epoch": epoch, "loss": avg_loss, "entropy": avg_entropy}


#%%
if __name__ == "__main__":

    logger.info("   Starting  ")

    # Training_files = [f"frame_{k}.bmp" for k in range(200)]
    # Test_files = [f"frame_{k}.bmp" for k in range(2001, 2286)]

    cfg = {}
    cfg["run"] = True

    ############################### CHOICE: TRAINING AND/OR RETRAINING #################
    cfg["train_initial"] = True
    cfg["retrain_projected"] = False
    ####################################################################################

    if cfg["run"]:

        cfg["num_epochs"] = 100

        cfg["resume"] = False
        cfg["checkpoint"] = None
        cfg["start_epoch"] = 1

        cfg["batch_every"] = 1
        cfg["save_every"] = 20
        cfg["epoch_every"] = 1
        cfg["shuffle"] = True
        cfg["random_seed"] = 3526
        cfg["device"] = "cuda"  # ["cpu", "cuda"]
        # model
        cfg["model_class"] = CAE

        # Data
        cfg["dataset_path"] = "flickr-compression-dataset/images"
        files_list = os.listdir(cfg["dataset_path"])
        n_files = len(files_list)
        idx = np.arange(0, n_files)
        np.random.shuffle(idx)
        train_idx = idx[: int(0.8 * n_files)]
        test_idx = idx[int(0.8 * n_files) :]
        cfg["train_files"] = list(map(files_list.__getitem__, train_idx))
        cfg["test_files"] = list(map(files_list.__getitem__, test_idx))

        cfg["num_workers"] = 0
        cfg["batch_size"] = 16
        cfg["validation_batch_size"] = 16

        # Adam optimizer parameters
        cfg["optimizer_class"] = optim.Adam
        cfg["optimizer_params"] = {"lr": 0.0001, "weight_decay": 1e-5}
        cfg["scheduler_params"] = {"type": "None"}
        # cfg["scheduler_params"] = {"type": "theis", "kappa": 0.5, "tau": 2000}
        # cfg["scheduler_params"] = {"type": "plateau", "factor": 0.5, "patience": 3}

        # loss criterion
        cfg["reconstruction_loss_class"] = torch.nn.HuberLoss
        cfg["reconstruction_loss_params"] = {"reduction": "sum", "delta": 1.0}
        cfg["beta"] = 0.0  # Loss = MSLoss + beta * entropy

        # save files

        cfg["nickname"] = f"{cfg['beta']}_Flickr_initial"
        cfg["exp_name"] = f'trainComp_{cfg["num_epochs"]}_{cfg["nickname"]}'

        ###########  INITIAL TRAINING SECTION ############################################################
        if cfg["train_initial"]:
            logger.warning(
                f'=== Train expe: {cfg["exp_name"]} =========================='
            )
            train_model(cfg, retrain=False)
            logger.info(
                f'=== expe: {cfg["exp_name"]} initial model trained =========================='
            )

        ############ RETRAIN SECTION ######################################################################
        if cfg["retrain_projected"]:

            # in case of retraining, set the the correct lists for the etas and the projections to be used
            for PARAM in [400]:
                for PROJ in ["L1"]:

                    cfg["num_epochs"] = 1  # set
                    cfg["retrain"] = {}
                    cfg["retrain"]["model_file"] = (
                        Path("../experiments")
                        / cfg["exp_name"]
                        / "checkpoint"
                        / "best_model.pth"
                    )
                    cfg["retrain"]["initial_model_file"] = (
                        Path("../experiments")
                        / cfg["exp_name"]
                        / "checkpoint"
                        / "model_0.pth"
                    )
                    cfg["retrain"]["projection"] = PROJ

                    if PROJ.upper() in ["L11", "L21"]:
                        cfg["retrain"]["projection_params"] = {
                            "eta": PARAM,
                            "direction": "row",
                        }
                    elif PROJ.upper() in ["L1", "L1INF"]:
                        cfg["retrain"]["projection_params"] = {
                            "eta": PARAM
                        }  # direction is not pertinent for L1 projection
                    elif PROJ in ["threshold", "sparse_global"]:
                        cfg["retrain"]["projection_params"] = {"fraction": PARAM}
                    else:
                        raise ValueError(f"Projection {PROJ} unknown")

                    ######################################################################################
                    # Do not sparsify the last layer
                    cfg["retrain"]["projection_params"]["omit"] = [
                        i for i in range(18, 42)
                    ]

                    # Does not allow sparsifiying the first (resp. last) layer when direction = columns
                    # (resp. direction = row)
                    if "direction" in cfg["retrain"]["projection_params"].keys():
                        if cfg["retrain"]["projection_params"]["direction"] == "col":
                            cfg["retrain"]["projection_params"]["omit"] += [
                                0,
                                1,
                            ]  # does not sparsify input layer

                    logger.info(f"  ")

                    logger.warning(
                        f'=== Retrain expe: {cfg["exp_name"]} proj:{PROJ} eta:{PARAM}  =========================='
                    )
                    config = copy.deepcopy(cfg)
                    train_model(config, retrain=True)
                    logger.info(
                        f'=== expe: {cfg["exp_name"]}  model has been retrained =========================='
                    )

#%%
