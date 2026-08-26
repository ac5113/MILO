"""
Utilities for loading input and models (reading pre-processed data)
"""

import os
import importlib
import glob
from omegaconf import OmegaConf

import torch

from body_model import BodyModel


ROOT_DIR = os.path.abspath(f"{__file__}/../../../../")


def load_config_from_log(log_dir):
    hydra_dir = f"{log_dir}/.hydra"
    cfg_path = f"{hydra_dir}/config.yaml"
    assert os.path.isdir(hydra_dir), f"{hydra_dir} does not exist"
    assert os.path.isfile(cfg_path), f"{cfg_path} does not exist"
    return OmegaConf.load(cfg_path)


def resolve_cfg_paths(cfg):
    paths = cfg.paths
    for name, rel_path in paths.items():
        if rel_path.startswith("/"):  # absolute path
            continue
        paths[name] = os.path.join(ROOT_DIR, rel_path)
    print("RESOLVED PATHS", paths)
    return cfg


def load_smpl_body_model(
    path,
    batch_size,
    num_betas=16,
    model_type="smplh",
    use_vtx_selector=True,
    device=None,
):
    """
    Load SMPL model
    """
    if device is None:
        device = torch.device("cpu")
    fit_gender = path.split("/")[-2]
    return (
        BodyModel(
            bm_path=path,
            num_betas=num_betas,
            batch_size=batch_size,
            use_vtx_selector=use_vtx_selector,
            model_type=model_type,
        ).to(device),
        fit_gender,
    )


def expid2model(expr_dir):
    """ "
    Reading VPoser models (https://github.com/nghorbani/human_body_prior).
    """
    from configer import Configer

    if not os.path.exists(expr_dir):
        raise ValueError("Could not find the experiment directory: %s" % expr_dir)

    best_model_fname = sorted(
        glob.glob(os.path.join(expr_dir, "snapshots", "*.pt")), key=os.path.getmtime
    )[-1]
    try_num = os.path.basename(best_model_fname).split("_")[0]

    print(("Found Trained Model: %s" % best_model_fname))

    default_ps_fname = glob.glob(os.path.join(expr_dir, "*.ini"))[0]
    if not os.path.exists(default_ps_fname):
        raise ValueError(
            "Could not find the appropriate vposer_settings: %s" % default_ps_fname
        )
    ps = Configer(
        default_ps_fname=default_ps_fname,
        work_dir=expr_dir,
        best_model_fname=best_model_fname,
    )

    return ps, best_model_fname


def load_vposer(expr_dir, vp_model="snapshot"):
    """
    :param expr_dir:
    :param vp_model: either 'snapshot' to use the experiment folder's code or a VPoser imported module, e.g.
    from human_body_prior.train.vposer_smpl import VPoser, then pass VPoser to this function
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    """
    ps, trained_model_fname = expid2model(expr_dir)
    if vp_model == "snapshot":

        vposer_path = sorted(
            glob.glob(os.path.join(expr_dir, "vposer_*.py")), key=os.path.getmtime
        )[-1]

        spec = importlib.util.spec_from_file_location("VPoser", vposer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        vposer_pt = getattr(module, "VPoser")(
            num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape
        )
    else:
        vposer_pt = vp_model(
            num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape
        )

    vposer_pt.load_state_dict(torch.load(trained_model_fname, map_location="cpu"))
    vposer_pt.eval()

    return vposer_pt, ps
