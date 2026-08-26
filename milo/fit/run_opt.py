import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset_from_cfg, expand_source_paths

from optim.base_scene import BaseSceneModel
from optim.optimizers import (
    RootOptimizer,
    SmoothOptimizer,
)
from optim.output import (
    save_track_info,
    save_camera_json,
    save_input_poses,
    save_initial_predictions,
)
from vis.viewer import init_viewer

from util.loaders import (
    load_vposer,
    load_smpl_body_model,
    resolve_cfg_paths,
)
from util.logger import Logger
from util.tensor import get_device, move_to

from run_vis import run_vis

import hydra
from omegaconf import DictConfig, OmegaConf


N_STAGES = 2


def run_opt(cfg, dataset, out_dir, device):
    args = cfg.data
    B = len(dataset)
    T = dataset.seq_len
    loader = DataLoader(dataset, batch_size=B, shuffle=False)

    obs_data = move_to(next(iter(loader)), device)
    cam_data = move_to(dataset.get_camera_data(), device)
    print("OBS DATA", obs_data.keys())
    print("CAM DATA", cam_data.keys())

    # save cameras
    cam_R, cam_t = dataset.cam_data.cam2world()
    intrins = dataset.cam_data.intrins
    save_camera_json(f"cameras.json", cam_R, cam_t, intrins)

    # check whether the cameras are static
    # if static, cannot optimize scale
    cfg.model.opt_scale &= not dataset.cam_data.is_static
    Logger.log(f"OPT SCALE {cfg.model.opt_scale}")

    # loss weights for all stages
    all_loss_weights = cfg.optim.loss_weights
    assert all(len(wts) == N_STAGES for wts in all_loss_weights.values())
    stage_loss_weights = [
        {k: wts[i] for k, wts in all_loss_weights.items()} for i in range(N_STAGES)
    ]

    # load models
    cfg = resolve_cfg_paths(cfg)
    paths = cfg.paths
    Logger.log(f"Loading pose prior from {paths.vposer}")
    pose_prior, _ = load_vposer(paths.vposer)
    pose_prior = pose_prior.to(device)

    Logger.log(f"Loading body model from {paths.smpl}")
    body_model, fit_gender = load_smpl_body_model(paths.smpl, B * T, device=device)

    margs = cfg.model
    base_model = BaseSceneModel(
        B, T, body_model, pose_prior, fit_gender=fit_gender, **margs
    )
    base_model.initialize(obs_data, cam_data)
    base_model.to(device)

    # save initial results for later visualization
    save_input_poses(dataset, os.path.join(out_dir, "phalp"), args.seq)
    save_initial_predictions(base_model, os.path.join(out_dir, "init"), args.seq)

    opts = cfg.optim.options
    vis_scale = 0.25
    vis = None
    if opts.vis_every > 0:
        vis = init_viewer(
            dataset.img_size,
            cam_data["intrins"][0],
            vis_scale=vis_scale,
            bg_paths=dataset.sel_img_paths,
            fps=cfg.fps,
        )
    print("OPTIMIZER OPTIONS:", opts)

    writer = SummaryWriter(out_dir)

    # Stage 1: optimizing smpl and object trans
    optim = RootOptimizer(base_model, stage_loss_weights, **opts)
    initial_time = time.time()
    optim.run(obs_data, cfg.optim.root.num_iters, out_dir, vis, writer)
    root_time = time.time() - initial_time

    # Snapshot root_fit output as init targets for subsequent stages
    with torch.no_grad():
        root_fit_params = base_model.params.get_dict()
        obs_data["init_fitted_latent_pose"] = root_fit_params["latent_pose"].detach().clone()
        obs_data["init_fitted_hand_pose"] = root_fit_params["hand_pose"].detach().clone()

    args = cfg.optim.smooth
    optim = SmoothOptimizer(
        base_model, stage_loss_weights, opt_scale=args.opt_scale, **opts
    )
    initial_time = time.time()
    optim.run(obs_data, args.num_iters, out_dir, vis, writer)
    smooth_time = time.time() - initial_time

    print("TIMINGS:")
    print(f"  Root opt: {root_time:.1f} sec")
    print(f"  Smooth opt: {smooth_time:.1f} sec")


@hydra.main(version_base=None, config_path="confs", config_name="config.yaml")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)

    out_dir = os.getcwd()
    print("out_dir", out_dir)
    Logger.init(f"{out_dir}/opt_log.txt")

    # make sure we get all necessary inputs
    cfg.data.sources = expand_source_paths(cfg.data.sources)
    print("SOURCES", cfg.data.sources)

    dataset = get_dataset_from_cfg(cfg)
    save_track_info(dataset, out_dir)

    if cfg.run_opt:
        device = get_device(0)
        run_opt(cfg, dataset, out_dir, device)

    if cfg.run_vis:
        run_vis(
            cfg, dataset, out_dir, 0, **cfg.get("vis", dict())
        )


if __name__ == "__main__":
    main()
