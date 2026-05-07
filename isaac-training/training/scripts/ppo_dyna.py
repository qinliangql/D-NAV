import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world

GLOBAL_HIDDEN_STATE = None

class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.his_inv_depth_encoder = nn.Sequential(
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
        ).to(self.device)

        self.point_encoder = make_mlp(
            [3, 32, 8]
        ).to(self.device)

        self.diff_mlp = make_mlp(
            [32, 64, 8]
        ).to(self.device)
        
        # Feature extractor for LiDAR
        self.lidar_feature_extractor = nn.Sequential(
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        ).to(self.device)

        # Fully connected layers for final feature extraction
        self.final_mlp = make_mlp([256, 128]).to(self.device)


        # Actor etwork
        self.n_agents, self.action_dim = action_spec.shape
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")], 
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # Critic network
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"] 
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        # Loss related
        self.gae = GAE(0.99, 0.95) # generalized adavantage esitmation
        self.critic_loss_fn = nn.HuberLoss(delta=10) # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # Optimizer
        self.feature_extractor_optim = torch.optim.Adam(
            list(self.his_inv_depth_encoder.parameters()) +
            # list(self.now_depth_encoder.parameters()) +
            list(self.point_encoder.parameters()) +
            list(self.diff_mlp.parameters()) +
            list(self.lidar_feature_extractor.parameters()) +
            list(self.final_mlp.parameters()),  # 如果有 RNN 模块，也需要加入
            lr=cfg.feature_extractor.learning_rate
        )
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.actor.learning_rate)

        # Dummy Input for nn lazymodule
        dummy_input = observation_spec.zero()
        # print("dummy_input: ", dummy_input)


        self.__call__(dummy_input)

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)
    
    def encode_points(self, pts):  # (B, K, 3)
        B, K, _ = pts.shape
        feat = self.point_encoder(pts.reshape(B*K, 3))
        feat = feat.view(B, K, -1)
        return feat.max(dim=1).values   # (B, 16)
    
    def extract_features(self, tensordict):
        global GLOBAL_HIDDEN_STATE

        # Extract LiDAR features
        # now_depth = self.now_depth_encoder(tensordict[("agents", "observation", "lidar")])
        inv_depths_feature = self.his_inv_depth_encoder(tensordict[("agents", "observation", "his_depth")])

        # Extract dynamic point features
        pre_dyn_points = tensordict[("agents", "observation", "pre_dyna_points")]
        now_dyn_points = tensordict[("agents", "observation", "now_dyna_points")]
        f_pre = self.encode_points(pre_dyn_points)
        f_now = self.encode_points(now_dyn_points)

        delta_input = torch.cat([
            f_now,
            f_pre,
        ], dim=-1)

        f_delta = self.diff_mlp(delta_input)

        tensordict["_cnn_feature"] = self.lidar_feature_extractor(inv_depths_feature)
        
        # Combine features
        combined_features = torch.cat([
            tensordict["_cnn_feature"],
            f_now,
            f_delta,
            tensordict[("agents", "observation", "state")]
        ], dim=-1)

        # Final feature extraction
        final_features = self.final_mlp(combined_features)
        tensordict["_feature"] = final_features
        return tensordict

    def __call__(self, tensordict):
        tensordict = self.extract_features(tensordict)
        # self.feature_extractor(tensordict) # extract features
        self.actor(tensordict)
        self.critic(tensordict)

        # Cooridnate change: transform local to world
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict

    def train(self, tensordict):
        # tensordict: (num_env, num_frames, dim), batchsize = num_env * num_frames
        next_tensordict = tensordict["next"]

        with torch.no_grad():
            # next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict) # calculate features for next state value calculation
            # processed_tensordicts = []
            for env_idx in range(next_tensordict.shape[1]):  # 遍历每个环境
                tensordict_env = next_tensordict[:,env_idx]  # 提取单个环境的数据
                processed_tensordict = self.extract_features(tensordict_env)  # 调用 feature_extractor
                # processed_tensordict = self.feature_extractor(tensordict_env)  # 调用 feature_extractor
                # processed_tensordicts.append(processed_tensordict)
                next_tensordict[:, env_idx] = processed_tensordict

            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict["next", "agents", "reward"] # Reward obtained by state transition
        dones = tensordict["next", "terminated"] # Whether the next states are terminal states

        values = tensordict["state_value"] # This is calculated stored when we called forward to obtain actions
        values = self.value_norm.denormalize(values) # denomalize values based on running mean and var of return
        next_values = self.value_norm.denormalize(next_values)

        # calculate GAE: Generalized Advantage Estimation   优势函数和累计奖励
        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret) # update running mean and var for return
        ret = self.value_norm.normalize(ret)  # normalize return
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        # Training
        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        infos = torch.stack(infos).to_tensordict()
        
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}    

    
    def _update(self, tensordict): # tensordict shape (batch_size, )
        tensordict = self.extract_features(tensordict)
        # self.feature_extractor(tensordict)
        
        # Get action from the current policy
        action_dist = self.actor.get_dist(tensordict) # this does an actor forward to get "loc" and "scale" and use them to build multivariate normal distribution
        log_probs = action_dist.log_prob(tensordict[("agents", "action_normalized")]) # based on the gaussian, we can calculate the log prob of the action from the current policy

        # Entropy Loss 熵损失
        action_entropy = action_dist.entropy()
        entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)

        # Actor Loss
        advantage = tensordict["adv"] # the advantage is calculated based on GAE in hte previous step
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim 

        # Critic Loss 
        b_value = tensordict["state_value"]
        ret = tensordict["ret"] # Return G
        value = self.critic(tensordict)["state_value"] 
        value_clipped = b_value + (value - b_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio) # this guarantee that critic update is clamped
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)

        # Total Loss
        loss = entropy_loss + actor_loss + critic_loss

        # Optimize
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()

        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.) # to prevent gradient growing too large
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        return TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])