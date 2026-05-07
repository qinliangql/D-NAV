import torch
import einops
import numpy as np
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time

from omni.isaac.range_sensor import _range_sensor
import omni.kit.commands
from pxr import Gf, PhysicsSchemaTools, Sdf, UsdGeom, UsdLux, UsdPhysics, PhysxSchema
from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf, Tf, UsdPhysics
from omni.physx.scripts import utils as physx_utils

# import rospy
# from sensor_msgs.msg import PointCloud2, PointField
# import sensor_msgs.point_cloud2 as pc2
# import std_msgs.msg
from torch_cluster import fps  # 需要安装 torch-cluster 库
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
import cv2

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._functorch")

def euler_to_quaternion_my(euler):
    """将欧拉角 (roll, pitch, yaw) 转换为四元数 (w, x, y, z)。"""
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)

def quaternion_multiply(q1, q2):
    """计算两个四元数的乘积 q1 * q2。"""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

class NavigationEnv(IsaacEnv):

    # In one step:
    # 1. _pre_sim_step (apply action) -> step isaac sim
    # 2. _post_sim_step (update lidar)
    # 3. increment progress_buf
    # 4. _compute_state_and_obs (get observation and states, update stats)
    # 5. _compute_reward_and_done (update reward and calculate returns)

    def __init__(self, cfg):
        print("[Navigation Environment]: Initializing Env...")
        # LiDAR params:
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams +1 
        self.lidar_vres = (self.lidar_vfov[1]-self.lidar_vfov[0])/cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)
        self.lidar_static_range = 4.0

        self.h_bins=cfg.sensor.lidar_h_bins
        self.v_bins=cfg.sensor.lidar_v_bins
        self.lidar_fps_num = cfg.sensor.lidar_fps_num

        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams) 
        self.center_num = 8
        self.queue_length = 2  # 设置队列长度为3

        self.sample_num = 16


        self.dynamic_region_bin = 6

        super().__init__(cfg, cfg.headless)

         # 新增队列相关变量
        # 为每个环境创建一个队列，存储点云和位置信息
        self.pointcloud_queue = [[] for _ in range(self.num_envs)]  # 存储点云数据的队列
        self.position_queue = [[] for _ in range(self.num_envs)]   # 存储位置数据的队列
        self.is_reset = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.is_reset = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        # # # Initialize ROS node and publisher
        # rospy.init_node("lidar_publisher", anonymous=True)
        # # self.lidar_pub = rospy.Publisher("/lidar_pointcloud", PointCloud2, queue_size=10)
        # self.lidar_pub_all = rospy.Publisher("/lidar_all_pointcloud", PointCloud2, queue_size=10)
        # self.lidar_pub_filter = rospy.Publisher("/lidar_filter_pointcloud", PointCloud2, queue_size=10)
        # # self.lidar_pub_fps = rospy.Publisher("/lidar_fps_pointcloud", PointCloud2, queue_size=10)  # 新增的 Publisher
        # # self.lidar_pub_fps_pre2now = rospy.Publisher("/lidar_fps_pre2now_pointcloud", PointCloud2, queue_size=10)  # 新增的 Publisher
        # # self.lidar_pub_gt = rospy.Publisher("/lidar_gt_pointcloud", PointCloud2, queue_size=10)  # 新增的 Publisher

        # Drone Initialization
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        # # LiDAR Intialization
        # ray_caster_cfg = RayCasterCfg(
        #     prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
        #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
        #     attach_yaw_only=True,
        #     # attach_yaw_only=False,
        #     pattern_cfg=patterns.BpearlPatternCfg(
        #         horizontal_fov=360,
        #         horizontal_res=self.lidar_hres, # horizontal default is set to 10
        #         vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams) 
        #     ),
        #     debug_vis=False,    # 这个选项还是有用的，显示hit_points
        #     mesh_prim_paths=["/World/ground"],
        #     # mesh_prim_paths=["/World"],
        # )
        # self.lidar = RayCaster(ray_caster_cfg)
        # self.lidar._initialize_impl()

        # LiDAR Initialization
        self.lidar_path_list = []
        for env_idx in range(self.num_envs):
            drone_base_prim = f"/World/envs/env_{env_idx}/Hummingbird_0/base_link"
            lidar_prim_path = f"/World/envs/env_{env_idx}/Hummingbird_0/Lidar"
            _, lidar = omni.kit.commands.execute(
                "RangeSensorCreateLidar",
                path=lidar_prim_path,
                parent=drone_base_prim,
                min_range=0.2,
                max_range=self.lidar_range,
                draw_points=False,
                draw_lines=False,
                horizontal_fov=360.0,
                vertical_fov=self.lidar_vfov[1] - self.lidar_vfov[0],
                horizontal_resolution=self.lidar_hres,
                vertical_resolution=self.lidar_vres,
                rotation_rate=0.0,
                high_lod=True,
            )

            lidar.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, 0.1))
            self.lidar_path_list.append(str(lidar.GetPath()))
            
        self.lidar_interfaces = _range_sensor.acquire_lidar_sensor_interface()
        self.lidar_initialized = False

        # self.dbscan = DBSCAN(eps=0.5, min_samples=3)  # eps 是聚类半径，min_samples 是最小点数
        
        # start and target 
        with torch.device(self.device):
            # self.start_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            
            # Coordinate change: add target direction variable
            self.target_dir = torch.zeros(self.num_envs, 1, 3)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 1 , 3)
            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.     


    def _design_scene(self):
        # Initialize a drone in prim /World/envs/envs_0
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] # drone model class
        cfg = drone_model.cfg_cls(force_sensor=False)
        self.drone = drone_model(cfg=cfg)
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.0)])[0]
        self.drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # lighting
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        # Ground Plane
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        self.map_range = [22.0, 22.0, 4.5]

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(self.map_range[0]*2, self.map_range[1]*2), 
                border_width=5.0,
                num_rows=1, 
                num_cols=1, 
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,
                        obstacle_height_mode="range",
                        obstacle_width_range=(0.4, 1.1),
                        obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],
                        obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
                        platform_width=0.0,
                    ),
                },
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )
        terrain_importer = TerrainImporter(terrain_cfg)

        # print("self.cfg.env_dyn.num_obstacles: ", self.cfg.env_dyn.num_obstacles)

        if (self.cfg.env_dyn.num_obstacles == 0):
            return
        # Dynamic Obstacles
        # NOTE: we use cuboid to represent 3D dynamic obstacles which can float in the air 
        # and the long cylinder to represent 2D dynamic obstacles for which the drone can only pass in 2D 
        # The width of the dynamic obstacles is divided into N_w=4 bins
        # [[0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]]
        # The height of the dynamic obstacles is divided into N_h=2 bins
        # [[0, 0.5], [0.5, inf]] we want to distinguish 3D obstacles and 2d obstacles
        N_w = 8 # number of width intervals between [0, 1]
        N_h = 2 # number of height: current only support binary
        max_obs_width = 3.0
        self.max_obs_3d_height = 4.0
        self.max_obs_2d_height = 5.0
        self.dyn_obs_width_res = max_obs_width/float(N_w)
        dyn_obs_category_num = N_w * N_h
        self.dyn_obs_num_of_each_category = int(self.cfg.env_dyn.num_obstacles / dyn_obs_category_num)
        self.cfg.env_dyn.num_obstacles = self.dyn_obs_num_of_each_category * dyn_obs_category_num # in case of the roundup error


        # Dynamic obstacle info initialization
        self.dyn_obs_list = []
        self.dyn_obs_state = torch.zeros((self.cfg.env_dyn.num_obstacles, 13), dtype=torch.float, device=self.cfg.device) # 13 is based on the states from sim, we only care the first three which is position
        self.dyn_obs_state[:, 3] = 1. # Quaternion
        # self.dyn_obs_angular_vel = 0.5*torch.ones((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)  # 障碍物的角速度
        # 随机选择旋转轴（x、y 或 z）
        rotation_axes = torch.randint(0, 3, (self.cfg.env_dyn.num_obstacles,), device=self.cfg.device)  # 每个障碍物随机选择一个旋转轴
        # 初始化角速度为 0
        self.dyn_obs_angular_vel = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        # 随机生成 [0, 1) 范围内的值
        random_values = torch.rand((self.cfg.env_dyn.num_obstacles,), dtype=torch.float, device=self.cfg.device)
        # 将一半的值映射到 [-1, -0.5]，另一半映射到 [0.5, 1]
        random_angular_velocities = torch.where(
            random_values < 0.5,  # 条件：随机值小于 0.5
            -1.0 + random_values * 1.0,  # 映射到 [-1, -0.5]
            0.5 + (random_values - 0.5) * 1.0  # 映射到 [0.5, 1]
        )
        self.dyn_obs_angular_vel[torch.arange(self.cfg.env_dyn.num_obstacles), rotation_axes] = random_angular_velocities
        
        self.dyn_obs_goal = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_origin = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_vel = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_step_count = 0 # dynamic obstacle motion step count
        self.dyn_obs_size = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.device) # size of dynamic obstacles


        # helper function to check pos validity for even distribution condition
        def check_pos_validity(prev_pos_list, curr_pos, adjusted_obs_dist):
            for prev_pos in prev_pos_list:
                if (np.linalg.norm(curr_pos - prev_pos) <= adjusted_obs_dist):
                    return False
            return True            
        
        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles) # prefered distance between each dynamic obstacle
        curr_obs_dist = obs_dist
        prev_pos_list = [] # for distance check
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num/N_h) # 4
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # create all origins for 3D dynamic obstacles of this category (size)
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                # random sample an origin until satisfy the evenly distributed condition
                start_time = time.time()
                while (True):
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                    if (category_idx < cuboid_category_num):
                        oz = np.random.uniform(low=0.0, high=self.map_range[2]) 
                    else:
                        oz = self.max_obs_2d_height/2. # half of the height
                    curr_pos = np.array([ox, oy])
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)
                    curr_time = time.time()
                    if (curr_time - start_time > 0.1):
                        curr_obs_dist *= 0.8
                        start_time = time.time()
                    if (valid):
                        prev_pos_list.append(curr_pos)
                        break
                curr_obs_dist = obs_dist
                origin = [ox, oy, oz]
                self.dyn_obs_origin[origin_idx+category_idx*self.dyn_obs_num_of_each_category] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)     
                self.dyn_obs_state[origin_idx+category_idx*self.dyn_obs_num_of_each_category, :3] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)                        
                prim_utils.create_prim(f"/World/Origin{origin_idx+category_idx*self.dyn_obs_num_of_each_category}", "Xform", translation=origin)

            # Spawn various sizes of dynamic obstacles 
            # 相同尺寸的障碍物共用一个prim
            if (category_idx < cuboid_category_num):
                # spawn for 3D dynamic obstacles
                if (category_idx%2 == 0):
                    obs_width = width = float(category_idx+1) * max_obs_width/float(N_w)
                    length =  max_obs_width/float(N_w)
                else:
                    length = float(category_idx+1) * max_obs_width/float(N_w)
                    obs_width = width =  max_obs_width/float(N_w)
                if category_idx==1:
                    obs_height = 6.0
                else:
                    obs_height = self.max_obs_3d_height
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cuboid",
                    spawn=sim_utils.CuboidCfg(
                        size=[width, length, obs_height],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cuboid_cfg)
            else:
                radius = float(category_idx-cuboid_category_num+1) * (max_obs_width/3)/float(N_w) / 2.
                obs_width = radius * 2
                obs_height = self.max_obs_2d_height
                # spawn for 2D dynamic obstacles
                cylinder_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cylinder",
                    spawn=sim_utils.CylinderCfg(
                        radius = radius,
                        height = self.max_obs_2d_height, 
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cylinder_cfg)
            self.dyn_obs_list.append(dynamic_obstacle)
            self.dyn_obs_size[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category] \
                = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float, device=self.cfg.device)

    def move_dynamic_obstacle(self):
        # Step 1: Random sample new goals for required update dynamic obstacles
        # Check whether the current dynamic obstacles need new goals
        dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1)) if self.dyn_obs_step_count !=0 \
            else torch.zeros(self.dyn_obs_state.size(0), device=self.cfg.device)
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5 # change to a new goal if less than the threshold
        
        # sample new goals in local range
        num_new_goal = torch.sum(dyn_obs_new_goal_mask)
        sample_x_local = -self.cfg.env_dyn.local_range[0] + 2. * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_y_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_z_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)
    
        # apply local goal to the global range
        self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
        # clamp the range if out of the static env range
        self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0])
        self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1])
        self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], min=0., max=self.map_range[2])
        self.dyn_obs_goal[int(self.dyn_obs_goal.size(0)/2):, 2] = self.max_obs_2d_height/2. # for 2d obstacles


        # Step 2: Random sample velocity for roughly every 2 seconds
        if (self.dyn_obs_step_count % int(2.0/self.cfg.sim.dt) == 0):
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (self.cfg.env_dyn.vel_range[1] \
              - self.cfg.env_dyn.vel_range[0]) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * \
                (self.dyn_obs_goal - self.dyn_obs_state[:, :3])/torch.norm((self.dyn_obs_goal - self.dyn_obs_state[:, :3]), dim=1, keepdim=True)

        # Step 3: Calculate new position update for current timestep
        # self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt   
        # 原本是所有的物体都会移动到新的位置，现在改为部分原地不动只旋转
        rota_slice = int(self.dyn_obs_state.size(0)*0.20)
        self.dyn_obs_state[rota_slice:, :3] += (self.dyn_obs_vel * self.cfg.sim.dt)[rota_slice:, :] 

        angular_displacement = (self.dyn_obs_angular_vel * self.cfg.sim.dt)[:rota_slice, :]   # 角位移 = 角速度 * 时间步长
        current_quat = self.dyn_obs_state[:rota_slice, 3:7]  # 当前四元数 (w, x, y, z)
        delta_quat = euler_to_quaternion_my(angular_displacement)  # 将角位移转换为四元数
        new_quat = quaternion_multiply(current_quat, delta_quat)  # 更新四元数
        self.dyn_obs_state[:rota_slice, 3:7] = new_quat / torch.norm(new_quat, dim=1, keepdim=True)  # 归一化四元数



        # Step 4: Update Visualized Location in Simulation
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category]) 
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1


    def _set_specs(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 5  # 10

        # Observation Spec
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, self.h_bins, self.v_bins), device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "his_depth": UnboundedContinuousTensorSpec((self.queue_length, self.h_bins, self.v_bins), device=self.device),
                    "pre_dyna_points": UnboundedContinuousTensorSpec((self.sample_num, 3), device=self.device),
                    "now_dyna_points": UnboundedContinuousTensorSpec((self.sample_num, 3), device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # Action Spec
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # number of motor
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # Done Spec
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    
    def reset_target(self, env_ids: torch.Tensor):
        if (self.training):
            # decide which side
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)


            # generate random positions
            target_pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            target_pos[:, 0, 2] = heights# height
            target_pos = target_pos * selected_masks + selected_shifts
            
            # apply target pos
            self.target_pos[env_ids] = target_pos

            # target_pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            # target_pos[:, 0, 1] = -24.
            # target_pos[:, 0, 2] = 2. 
            # self.target_pos[env_ids] = target_pos

        else:
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = -24.
            self.target_pos[:, 0, 2] = 2. 

        # self.target_pos[:, 0, 0] = -24
        # self.target_pos[:, 0, 1] = -24.
        # self.target_pos[:, 0, 2] = 2.            


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)
        if (self.training):
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)

            # generate raorch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            pos[:, 0, 2] = heights# height
            pos = pos * selected_masks + selected_shifts

            # pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            # pos[:, 0, 1] = 24.
            # pos[:, 0, 2] = 2.
            
        else:
            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            pos[:, 0, 1] = 24.
            pos[:, 0, 2] = 2.
        
        # pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
        # pos[:, 0, 0] = 0.
        # pos[:, 0, 1] = 0.
        # pos[:, 0, 2] = 2.
        # print("pos:", pos)
        
        # Coordinate change: after reset, the drone's target direction should be changed
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # Coordinate change: after reset, the drone's facing direction should face the current goal
        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        diff = self.target_pos[env_ids] - pos
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        rpy[..., 2] = facing_yaw

        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.prev_drone_vel_w[env_ids] = 0.
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])

        self.stats[env_ids] = 0. 
        self.is_reset[env_ids] = 1
        
    
    def manage_his_pointcloud(self, now_loc_sample_hits): 
        # 将当前帧数据添加到队列中
        for env_idx in range(self.num_envs):
            # 检查环境是否被重置
            if self.is_reset[env_idx] == 1:
                # 重置时清空队列
                self.pointcloud_queue[env_idx] = []
                self.position_queue[env_idx] = []
                self.is_reset[env_idx] = 0
                for i in range(self.queue_length-1):
                    self.pointcloud_queue[env_idx].append(now_loc_sample_hits[env_idx].clone())
                    self.position_queue[env_idx].append(self.root_state[env_idx, :, :3].clone())
            
            # 将当前帧数据添加到队列末尾
            self.pointcloud_queue[env_idx].append(now_loc_sample_hits[env_idx].clone())
            self.position_queue[env_idx].append(self.root_state[env_idx, :, :3].clone())
            
            # 保持队列长度不超过设定值
            if len(self.pointcloud_queue[env_idx]) > self.queue_length:
                self.pointcloud_queue[env_idx].pop(0)
                self.position_queue[env_idx].pop(0)

            # 有多个时间段就全部点云转到当前的坐标系下
            if len(self.pointcloud_queue[env_idx]) > 1:
                # 获取队列的当前位置和上一个时刻位置
                last_position = self.position_queue[env_idx][-2]
                now_position = self.position_queue[env_idx][-1]
                # 计算相对于当前位置，之前的点云的相对位置
                shift_pos = (last_position - now_position)
                # 遍历列表中的每个点云张量，将位置偏移应用到每个张量上
                for i in range(len(self.pointcloud_queue[env_idx][:-1])):
                    # print("shift_pos:", shift_pos.shape)
                    # print("self.pointcloud_queue[env_idx][i]:", self.pointcloud_queue[env_idx][i].shape)
                    if self.pointcloud_queue[env_idx][i].numel() > 0:
                        self.pointcloud_queue[env_idx][i] = shift_pos + self.pointcloud_queue[env_idx][i]

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")] 
        self.drone.apply_action(actions) 

    def _post_sim_step(self, tensordict: TensorDictBase):
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.move_dynamic_obstacle()
        # self.lidar.update(self.dt)
    
    def get_lidar_pose(self,lidar_path,idx):
        # # 获取 USD Stage
        # stage = omni.usd.get_context().get_stage()
        # # 获取 LiDAR 的 Prim
        # lidar_prim = stage.GetPrimAtPath(lidar_path)
        # if not lidar_prim.IsValid():
        #     raise ValueError(f"Invalid LiDAR path: {lidar_path}")
        # # 获取 LiDAR 的 Transform
        # xform_cache = UsdGeom.XformCache()
        # transform = xform_cache.GetLocalToWorldTransform(lidar_prim)
        # # 提取位姿信息
        # lidar_offset_position = np.array(transform.ExtractTranslation())

        # 获取无人机的全局位姿
        # drone_position = self.root_state[idx, 0, :3].cpu().numpy()  # 无人机的全局位置
        drone_rotation = self.root_state[idx, 0, 3:7].cpu().numpy()  # 无人机的全局旋转（四元数）

        # 计算 LiDAR 的全局位
        # lidar_offset_position[2] = drone_position[2]  # 平移
        return None,drone_rotation

    def transform_pointcloud_to_world(self, pointcloud, position, rotation):
        # pointcloud: [B, N, 3] or [N, 3]
        # rotation: [B, 4] (w, x, y, z) or [4]
        # position: [B, 3] or [3] or None

        # 支持单个或批量
        if pointcloud.ndim == 2:
            pointcloud = pointcloud.unsqueeze(0)  # [1, N, 3]
            rotation = rotation.unsqueeze(0)      # [1, 4]
            # if position is not None:
            #     position = position.unsqueeze(0)  # [1, 3]

        # 四元数 [w, x, y, z] -> [x, y, z, w] for torch implementation
        q = rotation
        if q.shape[-1] == 4:
            # # [B, 4] -> [B, 4]
            qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            # 旋转矩阵公式
            xx = qx * qx
            yy = qy * qy
            zz = qz * qz
            xy = qx * qy
            xz = qx * qz
            yz = qy * qz
            wx = qw * qx
            wy = qw * qy
            wz = qw * qz

            rot = torch.stack([
                1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy),
                2 * (xy + wz),         1 - 2 * (xx + zz),     2 * (yz - wx),
                2 * (xz - wy),         2 * (yz + wx),         1 - 2 * (xx + yy)
            ], dim=-1).reshape(-1, 3, 3)  # [B, 3, 3]

        else:
            raise ValueError("rotation shape error")

        # 批量点云旋转
        points_rot = torch.bmm(pointcloud, rot.transpose(1, 2))  # [B, N, 3]
        # if position is not None:
        #     points_rot = points_rot + position.unsqueeze(1)  # [B, N, 3]
        
        return points_rot

    def rotate_pointcloud_to_target_dir(self, points_w):
        """
        将点云从世界坐标系旋转到目标方向的坐标系，仅转换 x 和 y 坐标。
        Args:
            points_w: [B, N, 3] 世界坐标系下的点云
        Returns:
            points_rot: [B, N, 3] 旋转后的点云
        """
        # 确保 target_dir_2d 的 z 分量为 0
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[..., 2] = 0
        # print("target_dir_2d",target_dir_2d)

        # 计算目标方向的夹角
        theta = torch.atan2(target_dir_2d[..., 1], target_dir_2d[..., 0])  # [B]
        # print("theta:", theta)

        # 构造二维旋转矩阵
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rot_matrix_2d = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1),
        ], dim=-2).reshape(-1, 2, 2)  # [B, 2, 2]

        # print("rot_matrix_2d:", rot_matrix_2d[0])

        # 提取点云的 x 和 y 坐标
        points_xy = points_w[..., :2]  # [B, N, 2]
        # print("points_xy:", points_xy[0,0])

        # 应用二维旋转
        points_xy_rot = torch.bmm(points_xy, rot_matrix_2d)  # [B, N, 2]
        # print("points_xy_rot:", points_xy_rot[0,0])

        # 保留 z 坐标
        points_rot = torch.cat([points_xy_rot, points_w[..., 2:3]], dim=-1)  # [B, N, 3]

        return points_rot

    def process_lidar_spherical_quantization(self):
        lidar_range = self.lidar_static_range
        # 固定角度范围
        min_v = np.deg2rad(-10)
        max_v = np.deg2rad(40)
        h_bins = self.h_bins
        v_bins = self.v_bins

        # 分环境处理
        quantized_distances_all = torch.full(
            (self.num_envs, h_bins, v_bins), float('inf'), device=self.lidar_relative_hits.device
        )
        filtered_positions_all = []
        filtered_positions_block = []   # 每个点所属的动态 block（0~5）

        for env_idx in range(self.num_envs):
            # 筛选当前环境的点云数据
            mask = self.lidar_ray_dis[env_idx] < lidar_range  # (num_points,)
            indices = torch.nonzero(mask).squeeze(-1)  # (total_filtered,)
            
            if indices.numel() == 0:
                filtered_positions_all.append(torch.tensor([], device=self.lidar_relative_hits.device))
                filtered_positions_block.append(torch.tensor([], device=self.lidar_relative_hits.device))    # 无效点站位
                continue  # 如果当前环境没有有效点，跳过

            filtered_positions = self.lidar_relative_hits[env_idx, indices]
            filtered_distances = self.lidar_ray_dis[env_idx, indices]

            # 计算水平和垂直角度
            horizontal_angles = torch.atan2(filtered_positions[:, 1], filtered_positions[:, 0])  # [-pi, pi]
            vertical_angles = torch.asin(filtered_positions[:, 2] / filtered_distances)          # [-pi/2, pi/2]

            # 只保留在[min_v, max_v]范围内的点
            valid_mask = (vertical_angles >= min_v) & (vertical_angles <= max_v)
            horizontal_angles = horizontal_angles[valid_mask]
            vertical_angles = vertical_angles[valid_mask]
            filtered_distances = filtered_distances[valid_mask]
            filtered_positions_all.append(filtered_positions[valid_mask])

            # if env_idx == 0:
            #     # 获取 LiDAR 的击中点
            #     lidar_hits = filtered_positions.cpu().numpy()  # Shape: (N, 3)
            #     # lidar_hits = self.lidar_relative_hits[0].cpu().numpy()  # Shape: (N, 3)
            #     # print("sampled_hits_all:",sampled_hits_all)
            #     # 创建 PointCloud2 消息
            #     header = std_msgs.msg.Header()
            #     header.stamp = rospy.Time.now()
            #     header.frame_id = "map"  # 设置坐标系
            #     # 将点云数据转换为 PointCloud2 格式
            #     pointcloud_msg = pc2.create_cloud_xyz32(header, lidar_hits.tolist())
            #     # 发布点云消息
            #     self.lidar_pub_filter.publish(pointcloud_msg)

            #     # 获取 LiDAR 的击中点
            #     lidar_hits = filtered_positions[valid_mask].cpu().numpy()  # Shape: (N, 3)
            #     # lidar_hits = self.lidar_relative_hits[0].cpu().numpy()  # Shape: (N, 3)
            #     # print("sampled_hits_all:",sampled_hits_all)
            #     # 创建 PointCloud2 消息
            #     header = std_msgs.msg.Header()
            #     header.stamp = rospy.Time.now()
            #     header.frame_id = "map"  # 设置坐标系
            #     # 将点云数据转换为 PointCloud2 格式
            #     pointcloud_msg = pc2.create_cloud_xyz32(header, lidar_hits.tolist())
            #     # 发布点云消息
            #     self.lidar_pub_gt.publish(pointcloud_msg)

            # 量化索引（在固定区间内归一化）
            h_indices = ((horizontal_angles + torch.pi) / (2 * torch.pi) * h_bins).long().clamp(0, h_bins - 1)
            v_indices = ((vertical_angles - min_v) / (max_v - min_v) * v_bins).long().clamp(0, v_bins - 1)
            block_indices = h_indices // self.dynamic_region_bin
            filtered_positions_block.append(block_indices)

            # 更新当前环境的量化距离
            linear_indices = h_indices * v_bins + v_indices
            flat_quantized = torch.full((h_bins * v_bins,), float('inf'), device=filtered_positions.device)
            flat_quantized.scatter_reduce_(0, linear_indices, filtered_distances, reduce='amin', include_self=False)

            quantized_distances_all[env_idx] = flat_quantized.view(h_bins, v_bins)
        # print("quantized_distances_all:",quantized_distances_all[0])

        # 将无穷值替换为最大范围
        quantized_distances_all[quantized_distances_all == float('inf')] = lidar_range
        quantized_distances_all = quantized_distances_all.reshape(self.num_envs, 1, h_bins, v_bins)

        return quantized_distances_all, filtered_positions_all, filtered_positions_block

    def process_lidar_spherical_quantization_history(self, now_depth):
        lidar_range = self.lidar_static_range
        # 固定角度范围
        min_v = np.deg2rad(-10)
        max_v = np.deg2rad(40)
        h_bins = self.h_bins
        v_bins = self.v_bins

        # 分环境处理
        quantized_distances_all = torch.full(
            (self.num_envs, self.queue_length, h_bins, v_bins), float('inf'), device=self.lidar_relative_hits.device
        )

        filtered_positions_block_pre = []
        filtered_positions_pre2now = []

        for env_idx in range(self.num_envs):
            for i in range(self.queue_length-1):
                filtered_positions = self.pointcloud_queue[env_idx][i]
                filtered_distances = filtered_positions.norm(dim=-1)


                if filtered_positions.numel() == 0:
                    if i == 0:
                        filtered_positions_block_pre.append(torch.tensor([], device=self.lidar_relative_hits.device))
                        filtered_positions_pre2now.append(torch.tensor([], device=self.lidar_relative_hits.device))    # 无效点站位
                    continue  # 如果当前环境没有有效点，跳过

                # 计算水平和垂直角度
                horizontal_angles = torch.atan2(filtered_positions[:, 1], filtered_positions[:, 0])  # [-pi, pi]
                vertical_angles = torch.asin(filtered_positions[:, 2] / filtered_distances)          # [-pi/2, pi/2]

                # 只保留在[min_v, max_v]范围内的点
                valid_mask = (vertical_angles >= min_v) & (vertical_angles <= max_v)
                horizontal_angles = horizontal_angles[valid_mask]
                vertical_angles = vertical_angles[valid_mask]
                filtered_distances = filtered_distances[valid_mask]

                # 量化索引（在固定区间内归一化）
                h_indices = ((horizontal_angles + torch.pi) / (2 * torch.pi) * h_bins).long().clamp(0, h_bins - 1)
                v_indices = ((vertical_angles - min_v) / (max_v - min_v) * v_bins).long().clamp(0, v_bins - 1)

                if i == 0: # 只算时间距离最久的是点云的block
                    block_indices = h_indices // self.dynamic_region_bin
                    filtered_positions_block_pre.append(block_indices)
                    filtered_positions_pre2now.append(filtered_positions[valid_mask])

                # 更新当前环境的量化距离
                linear_indices = h_indices * v_bins + v_indices
                flat_quantized = torch.full((h_bins * v_bins,), float('inf'), device=filtered_positions.device)
                flat_quantized.scatter_reduce_(0, linear_indices, filtered_distances, reduce='amin', include_self=False)

                quantized_distances_all[env_idx, i] = flat_quantized.view(h_bins, v_bins)
                
            quantized_distances_all[env_idx, -1] = now_depth[env_idx,0] 

        # 将无穷值替换为最大范围
        quantized_distances_all[quantized_distances_all == float('inf')] = lidar_range
        inv_his_depths = (lidar_range - quantized_distances_all)/lidar_range

        return inv_his_depths, filtered_positions_block_pre, filtered_positions_pre2now

    def get_dynamic_points(self, max_dyn_block, points_all, block_all, sample_num=16):
        """
        max_dyn_block: (num_envs,)
        points_all:    list[num_envs] of (Ni, 3)
        block_all:     list[num_envs] of (Ni,)
        return:
            dyn_points_sample: (num_envs, sample_num, 3)
        """

        device = self.lidar_relative_hits.device

        dyn_points_sample = torch.zeros(
            (self.num_envs, sample_num, 3), device=device
        )

        for env_idx in range(self.num_envs):
            target_block = max_dyn_block[env_idx]

            pts = points_all[env_idx]      # (Ni,3)
            blocks = block_all[env_idx]    # (Ni,)

            # 无效情况直接跳过（保持 0）
            if blocks.numel() == 0 or target_block < 0:
                continue

            mask = (blocks == target_block)
            dyn_points = pts[mask]         # (K,3)

            num_dyn = dyn_points.shape[0]
            if num_dyn == 0:
                continue

            # ===== 点数控制 =====
            if num_dyn >= sample_num:
                center = dyn_points.mean(dim=0, keepdim=True)
                dists = (dyn_points - center).norm(dim=1)
                idx = torch.topk(dists, sample_num, largest=False).indices
                dyn_points_sample[env_idx] = dyn_points[idx]
            else:
                # 不足 sample_num，用 0 padding
                dyn_points_sample[env_idx, :num_dyn] = dyn_points

        return dyn_points_sample

    # get current states/observation
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state(env_frame=False) # (world_pos, orientation (quat), world_vel_and_angular, heading, up, 4motorsthrust)
        self.info["drone_state"][:] = self.root_state[..., :13] # info is for controller
                
        # >>>>>>>>>>>>The relevant code starts from here<<<<<<<<<<<<
        # -----------Network Input I: LiDAR range data--------------
        # self.lidar.data.ray_hits_w：表示 LiDAR 射线在世界坐标系下的击中点（即障碍物的坐标）。
        # self.lidar.data.pos_w：表示 LiDAR 的发射点（即无人机的位置）。
        # self.lidar_range：LiDAR 的最大探测范围。
        # self.lidar_relative_hits = self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1)  # [num_drones, num_rays,3]

        if not self.lidar_initialized:
            # 尝试读取一次雷达数据，判断是否已初始化完成
            test_data = self.lidar_interfaces.get_point_cloud_data(self.lidar_path_list[0])
            if 0 != test_data.shape[0]:
                self.lidar_initialized = True
                print("LiDAR initialization completed.")
            else:
                self.lidar_relative_hits = torch.zeros(self.num_envs, self.lidar_resolution[0]*self.lidar_resolution[1], 3, device=self.cfg.device)  # [num_drones, num_rays,3]
                # time.sleep(5)  # 等待 5 秒钟，直到 LiDAR 初始化完成

        if self.lidar_initialized:
            # 假设 get_point_cloud_data 支持批量返回所有env的点云
            # pointclouds: [envs, 360, 59, 3]
            pointclouds = np.stack([
                self.lidar_interfaces.get_point_cloud_data(lidar_path)
                for lidar_path in self.lidar_path_list
            ])  # shape: [envs, 360, 59, 3]
            points = torch.from_numpy(pointclouds).reshape(self.num_envs, -1, 3).to('cuda')  # [envs, 360*59, 3]

            # 获取所有env的旋转（假设 get_lidar_pose 支持批量，或者你可以直接用 self.root_state）
            rotations = torch.from_numpy(
                np.stack([self.get_lidar_pose(lidar_path, i)[1] for i, lidar_path in enumerate(self.lidar_path_list)])
            ).to('cuda')  # [envs, 4]

            # 批量旋转点云
            # 你可以用 torch 或 scipy 的批量旋转函数
            # 这里假设 transform_pointcloud_to_world 支持批量
            points[points == float('inf')] = self.lidar_range
            points_w = self.transform_pointcloud_to_world(points, None, rotations)  # [envs, 360*59, 3]

            self.lidar_relative_hits = self.rotate_pointcloud_to_target_dir(points_w)  # 已经在cuda上


            # lidar_hits = self.lidar_relative_hits[0].cpu().numpy()  # Shape: (N, 3)
            # # lidar_hits = self.lidar_relative_hits[0].cpu().numpy()  # Shape: (N, 3)
            # # print("sampled_hits_all:",sampled_hits_all)
            # # 创建 PointCloud2 消息
            # header = std_msgs.msg.Header()
            # header.stamp = rospy.Time.now()
            # header.frame_id = "map"  # 设置坐标系
            # # 将点云数据转换为 PointCloud2 格式
            # pointcloud_msg = pc2.create_cloud_xyz32(header, lidar_hits.tolist())
            # # 发布点云消息
            # self.lidar_pub_all.publish(pointcloud_msg)


        self.lidar_ray_dis = self.lidar_relative_hits.norm(dim=-1).clamp_max(self.lidar_range)   # [num_drones, num_rays]
        # t1 = time.time()
        now_depth, now_filter_points, now_filter_block = self.process_lidar_spherical_quantization()  # [num_drones, 1, h_bins, v_bins]
        self.lidar_scan = self.lidar_static_range - now_depth
        # t2 = time.time()

        self.manage_his_pointcloud(now_filter_points) # 得到当前位置的历史点云
        inv_his_depths, pre2now_filter_block, pre2now_filtered_points = self.process_lidar_spherical_quantization_history(now_depth)

        # ===== 计算变化最剧烈的动态 block =====
        # inv_his_depths: (num_envs, 3, 36, 6)
        diff = torch.abs(inv_his_depths[:, 0] - inv_his_depths[:, -1])  # (E,36,6)

        # 36×6 → 6×6×6
        diff_blocks = diff.view(
            self.num_envs,
            self.h_bins // self.dynamic_region_bin,  # 6
            self.dynamic_region_bin,                 # 6
            self.v_bins                              # 6
        )

        # block 内求平均
        diff_block_score = diff_blocks.mean(dim=(2,3))  # (E,6)

        # 每个 env 一个 block index
        max_dyn_block = diff_block_score.argmax(dim=1)  # (E,)

        now_dyn_points = self.get_dynamic_points(   # [E,sample_num,3]
            max_dyn_block,
            now_filter_points,
            now_filter_block,
            sample_num=self.sample_num
        )



        pre_dyn_points = self.get_dynamic_points(   # [E,sample_num,3]
            max_dyn_block,
            pre2now_filtered_points,
            pre2now_filter_block,
            sample_num=self.sample_num
        )
      

        # t3 = time.time()
        # print("lidar process time: ", t2-t1, t3-t2) 

        # # 可视化
        # inv_his_depths_np0 = inv_his_depths[0,0].cpu().numpy()
        # inv_his_depths_np1 = inv_his_depths[0,1].cpu().numpy()
        # inv_his_depths_np2 = inv_his_depths[0,2].cpu().numpy()

        # inv_his_depths_np0_show = cv2.resize(inv_his_depths_np0*255, (180, 300), interpolation=cv2.INTER_NEAREST)
        # inv_his_depths_np1_show = cv2.resize(inv_his_depths_np1*255, (180, 300), interpolation=cv2.INTER_NEAREST)
        # inv_his_depths_np2_show = cv2.resize(inv_his_depths_np2*255, (180, 300), interpolation=cv2.INTER_NEAREST)  # 对应当前的观测
        
        # cv2.imwrite("inv_his_depths_np0.png", cv2.applyColorMap(inv_his_depths_np0_show.astype(np.uint8), cv2.COLORMAP_JET))
        # cv2.imwrite("inv_his_depths_np1.png", cv2.applyColorMap(inv_his_depths_np1_show.astype(np.uint8), cv2.COLORMAP_JET))
        # cv2.imwrite("inv_his_depths_np2.png", cv2.applyColorMap(inv_his_depths_np2_show.astype(np.uint8), cv2.COLORMAP_JET))

        # diff1 = inv_his_depths_np1 - inv_his_depths_np0
        # diff2 = inv_his_depths_np2 - inv_his_depths_np1
        # diff = inv_his_depths_np2 - inv_his_depths_np0
        # diff1_show = cv2.resize(abs(diff1)*255, (180, 300), interpolation=cv2.INTER_NEAREST)
        # diff2_show = cv2.resize(abs(diff2)*255, (180, 300), interpolation=cv2.INTER_NEAREST)
        # diff_show = cv2.resize(abs(diff)*255, (180, 300), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite("diff1.png", cv2.applyColorMap(diff1_show.astype(np.uint8), cv2.COLORMAP_JET))
        # cv2.imwrite("diff2.png", cv2.applyColorMap(diff2_show.astype(np.uint8), cv2.COLORMAP_JET))
        # cv2.imwrite("diff.png", cv2.applyColorMap(diff_show.astype(np.uint8), cv2.COLORMAP_JET))

        # cv2.waitKey(1)

        # ---------Network Input II: Drone's internal states---------
        # a. distance info in horizontal and vertical plane
        # print("self.root_state[..., :3] :", self.root_state[..., :3])
        rpos = self.target_pos - self.root_state[..., :3]        
        distance = rpos.norm(dim=-1, keepdim=True) # start to goal distance
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)
        
        
        # b. unit direction vector to goal
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[..., 2] = 0

        rpos_clipped = rpos / distance.clamp(1e-6) # unit vector: start to goal direction
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d) # express in the goal coodinate
        
        # c. velocity in the goal frame
        vel_w = self.root_state[..., 7:10] # world vel
        vel_g = vec_to_new_frame(vel_w, target_dir_2d)   # coordinate change for velocity

        # final drone's internal states
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).squeeze(1)

            
        # -----------------Network Input Final--------------
        obs = {
            "state": drone_state,
            "lidar": self.lidar_scan/self.lidar_static_range,
            "direction": target_dir_2d,
            "his_depth": inv_his_depths,
            "pre_dyna_points": pre_dyn_points,
            "now_dyna_points": now_dyn_points,
        }


        # -----------------Reward Calculation-----------------
        # a. safety reward for static obstacles
        reward_safety_static = torch.log((self.lidar_static_range-self.lidar_scan).clamp(min=1e-6, max=self.lidar_static_range)).mean(dim=(2, 3))

        # 最近障碍物的距离
        closest_obstacle_distance = self.lidar_static_range - einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max")  # [num_envs, 1]

        # 定义距离阈值
        min_distance = 0.3  # 最小距离（碰撞距离）
        max_distance = 0.5  # 最大距离（开始惩罚的距离）

        # 计算惩罚值
        penalty_close_to_obstacle = torch.zeros_like(closest_obstacle_distance)
        mask = (closest_obstacle_distance < max_distance) & (closest_obstacle_distance >= min_distance)
        penalty_close_to_obstacle[mask] = 2 * (max_distance - closest_obstacle_distance[mask]) / (max_distance - min_distance)

        # 对距离小于最小距离的情况，直接设置为最大惩罚
        penalty_close_to_obstacle[closest_obstacle_distance < min_distance] = 2.0

        # print("penalty_close_to_obstacle:", penalty_close_to_obstacle)
        

        # b. safety reward for dynamic obstacles
        # if (self.cfg.env_dyn.num_obstacles != 0):
        #     reward_safety_dynamic = torch.log((closest_dyn_obs_distance_reward).clamp(min=1e-6, max=self.lidar_static_range)).mean(dim=-1, keepdim=True)

        # c. velocity reward for goal direction
        vel_direction = rpos / distance.clamp_min(1e-6)
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1)#.clip(max=2.0)
        
        # d. smoothness reward for action smoothness
        penalty_smooth = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1)
        
        # e. height penalty reward for flying unnessarily high or low
        penalty_height = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        penalty_height[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)] = ( (self.drone.pos[..., 2] - self.height_range[..., 1] - 0.2)**2 )[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)]
        penalty_height[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)] = ( (self.height_range[..., 0] - 0.2 - self.drone.pos[..., 2])**2 )[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)]


        # f. Collision condition with its penalty
        collision_dis = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max")
        # print("collision_dis:", collision_dis)
        static_collision = collision_dis >  (self.lidar_static_range - 0.3) # 0.3 collision radius
        # collision = static_collision | dynamic_collision
        collision = static_collision
        # print("collision:", collision)
        
        # print("reward_vel:",reward_vel)
        # print("reward_safety_static:",reward_safety_static)
        # print("reward_safety_dynamic:",reward_safety_dynamic)
        # print("penalty_smooth:",penalty_smooth*0.1)
        # print("penalty_height:",penalty_height*8.0)

        # Final reward calculation
        # if (self.cfg.env_dyn.num_obstacles != 0):
        #     self.reward = reward_vel + 1. + reward_safety_static * 1.0 + reward_safety_dynamic * 1.0 - penalty_smooth * 0.1 - penalty_height * 8.0 - penalty_close_to_obstacle
        # else:
        # self.reward = reward_vel*3 + 1. + reward_safety_static * 1.0 - penalty_smooth * 0.1 - penalty_height * 8.0 - penalty_close_to_obstacle
        self.reward = reward_vel + 1. + reward_safety_static - penalty_smooth * 0.1 - penalty_height * 8.0

        # Terminal reward
        # self.reward[collision] -= 10. # collision

        # Terminate Conditions
        # print("distance to goal:", distance.squeeze(-1))
        reach_goal = (distance.squeeze(-1) < 0.5)
        # self.reward[reach_goal] += 5 # reach goal

        below_bound = self.drone.pos[..., 2] < 0.2
        # print("below_bound:", below_bound)
        above_bound = self.drone.pos[..., 2] > 4.
        left_bound = self.drone.pos[..., 0] > 25.
        right_bound = self.drone.pos[..., 0] < -25.
        back_bound = self.drone.pos[..., 1] > 25.
        front_bound = self.drone.pos[..., 1] < -25.
        # print("above_bound:", above_bound)
        out_bound = below_bound | above_bound | left_bound | right_bound | back_bound | front_bound
        self.terminated = collision | out_bound
        # print("self.terminated:", self.terminated)
        # self.reward[out_bound] -= 10 # reach goal
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # progress buf is to track the step number

        # update previous velocity for smoothness calculation in the next ieteration
        self.prev_drone_vel_w = self.drone.vel_w[..., :3]

        # # -----------------Training Stats-----------------
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["reach_goal"] = reach_goal.float()
        # print("self.stats[reach_goal]:", self.stats["reach_goal"])
        self.stats["collision"] = collision.float()
        self.stats["truncated"] = self.truncated.float()

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )