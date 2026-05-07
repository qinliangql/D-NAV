import argparse
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from ppo_dyna import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType




FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train_dyna", version_base=None)
def main(cfg):
    # Simulation App
    print("cfg*****************************************************88: ", cfg)
    if cfg.show:
        sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})
    else:
        sim_app = SimulationApp({
            "headless": cfg.headless,
            "width": 4,
            "height": 3,
            "anti_aliasing": 0,
            "multi_samples": 0,
            "display_options": 0,
            "renderer": "RayTracedLighting",
            "max_fps": 30,
        })
    
    # Use Wandb to monitor training
    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            # entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )
 
    # Navigation Training Environment
    from env_train_dyna_lidar import NavigationEnv
    # from env_test_dyna import NavigationEnv
    env = NavigationEnv(cfg)

    # Transformed Environment
    transforms = []
    # transforms.append(ravel_composite(env.observation_spec, ("agents", "intrinsics"), start_dim=-1))
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)    
    # PPO Policy
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)

    checkpoint = "/home/ustc/UAV/D-NAV/isaac-training/ckpt/checkpoint_40500.pt"
    policy.load_state_dict(torch.load(checkpoint), strict=True)

    # checkpoint_state_dict = torch.load(checkpoint)
    # model_state_dict = policy.state_dict()
    # filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    # policy.load_state_dict(filtered_state_dict, strict=False)

    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    env.enable_render(True)
    env.eval()
    eval_info = evaluate(
        env=transformed_env, 
        policy=policy,
        seed=cfg.seed, 
        cfg=cfg,
        exploration_type=ExplorationType.MEAN
    )
    env.enable_render(not cfg.headless)
    env.train()
    env.reset()
    print("eval_info: ", eval_info)
    print("\n[NavRL]: evaluation done.")

    run.log(eval_info)
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
    