from __future__ import annotations  
  
import torch  
from typing import TYPE_CHECKING  
from collections.abc import Sequence  
  
from isaaclab.managers import SceneEntityCfg, CommandTerm, CommandTermCfg  
from isaaclab.sensors import ContactSensor  
from isaaclab.utils import configclass  
  
if TYPE_CHECKING:  
    from isaaclab.envs import ManagerBasedRLEnv  
  
  
class BallDirectionCommand(CommandTerm):  
    """Command term that generates velocity commands towards the ball."""  
      
    cfg: 'BallDirectionCommandCfg'  
      
    def __init__(self, cfg: 'BallDirectionCommandCfg', env: ManagerBasedRLEnv):  
        """Initialize the command term."""  
        super().__init__(cfg, env)  
        # Initialize command buffer  
        self.command_buffer = torch.zeros(self.num_envs, 3, device=self.device)  
      
    @property  
    def command(self) -> torch.Tensor:  
        """The current velocity commands. Shape is (num_envs, 3)."""  
        return self.command_buffer  
      
    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:  
        """Generate velocity commands towards the ball."""  
        # This method is not used by CommandManager, but kept for compatibility  
        return self.command  
      
    def _update_metrics(self):  
        """Update metrics based on current state."""  
        # You can add metrics here if needed  
        pass  
      
    def _resample_command(self, env_ids: Sequence[int]):  
        """Resample command for specified environments.  
          
        For this command, we don't actually resample - we always compute  
        the command towards the ball in _update_command.  
        """  
        pass  
      
    def _update_command(self):  
        """Update the command based on current state."""  
        # Get robot and ball positions  
        robot = self._env.scene[self.cfg.asset_name]  
        ball = self._env.scene["ball"]  
          
        robot_pos = robot.data.root_pos_w  
        robot_quat = robot.data.root_quat_w  
        ball_pos = ball.data.root_pos_w  
          
        # Calculate direction to ball in world frame  
        direction_world = ball_pos[:, :2] - robot_pos[:, :2]  
        distance = torch.norm(direction_world, dim=1, keepdim=True)  
          
        # Normalize direction  
        direction_world = direction_world / (distance + 1e-6)  
          
        # Convert direction to robot frame  
        qw, qx, qy, qz = robot_quat[:, 3], robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2]  
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))  
          
        # Rotate direction to robot frame  
        cos_yaw = torch.cos(-yaw)  
        sin_yaw = torch.sin(-yaw)  
        direction_robot_x = direction_world[:, 0] * cos_yaw - direction_world[:, 1] * sin_yaw  
        direction_robot_y = direction_world[:, 0] * sin_yaw + direction_world[:, 1] * cos_yaw  
          
        # Calculate desired heading angle  
        desired_heading = torch.atan2(direction_robot_y, direction_robot_x)  
          
        # Scale linear speed based on distance  
        speed_scale = torch.clamp(  
            (distance.squeeze() - self.cfg.stop_distance) /   
            (self.cfg.min_distance_for_max_speed - self.cfg.stop_distance),  
            0.0, 1.0  
        )  
          
        # Calculate velocity commands  
        lin_vel_x = direction_robot_x * speed_scale * self.cfg.max_linear_speed  
        lin_vel_y = direction_robot_y * speed_scale * self.cfg.max_linear_speed  
          
        # Calculate angular velocity command  
        ang_vel_z = torch.clamp(  
            desired_heading * self.cfg.max_angular_speed / torch.pi,  
            -self.cfg.max_angular_speed, self.cfg.max_angular_speed  
        )  
          
        # Update command buffer  
        self.command_buffer[:, 0] = lin_vel_x  
        self.command_buffer[:, 1] = lin_vel_y  
        self.command_buffer[:, 2] = ang_vel_z  
  
  
@configclass  
class BallDirectionCommandCfg(CommandTermCfg):  
    """Configuration for ball direction command generator.  
      
    Generates velocity commands that guide the robot towards the ball.  
    """  
      
    class_type: type = BallDirectionCommand  
      
    asset_name: str = "robot"  
    resampling_time_range: tuple[float, float] = (10.0, 10.0)  
    debug_vis: bool = True  
      
    # Speed parameters  
    max_linear_speed: float = 1.0  # Maximum linear speed towards ball  
    max_angular_speed: float = 0.5  # Maximum angular speed for turning  
      
    # Distance-based speed scaling  
    min_distance_for_max_speed: float = 2.0  # Distance at which to use max speed  
    stop_distance: float = 0.5  # Distance at which to stop  

    # 添加这些属性以兼容导出系统  
    ranges: dict = None  
    limit_ranges: dict = None 

    def __post_init__(self):  
        """Post-initialization to set required attributes."""  
        self.__name__ = "BallDirectionCommandCfg"  
        # 设置默认范围以兼容导出系统  
        self.ranges = {  
            "lin_vel_x": (-self.max_linear_speed, self.max_linear_speed),  
            "lin_vel_y": (-self.max_linear_speed, self.max_linear_speed),  
            "ang_vel_z": (-self.max_angular_speed, self.max_angular_speed)  
        }  
        self.limit_ranges = self.ranges.copy()
  
  
def relative_position(env: ManagerBasedRLEnv, source: str, target: str) -> torch.Tensor:  
    """Compute the relative position between two assets."""  
    source_asset = env.scene[source.split('/')[0]]  
    target_asset = env.scene[target.split('/')[0]]  
      
    # Get source position  
    if len(source.split('/')) > 1:  
        # If source is a body part  
        body_name = source.split('/')[1]  
        source_pos = source_asset.data.body_pos_w[:, source_asset.find_bodies(body_name)[0], :]  
    else:  
        # If source is the root  
        source_pos = source_asset.data.root_pos_w  
      
    # Get target position  
    if len(target.split('/')) > 1:  
        # If target is a body part  
        body_name = target.split('/')[1]  
        target_pos = target_asset.data.body_pos_w[:, target_asset.find_bodies(body_name)[0], :]  
    else:  
        # If target is the root  
        target_pos = target_asset.data.root_pos_w  
      
    return target_pos - source_pos  
  
  
def distance_to_target(env: ManagerBasedRLEnv, source: str, target: str) -> torch.Tensor:  
    """Compute the distance between two assets."""  
    rel_pos = relative_position(env, source, target)  
    return torch.norm(rel_pos, dim=-1)  
  
  
def contact_between_assets(env: ManagerBasedRLEnv, source: str, target: str) -> torch.Tensor:  
    """Check if there is contact between two assets."""  
    # Get contact sensors for both assets  
    source_contact = None  
    target_contact = None  
      
    for sensor_name, sensor in env.scene.sensors.items():  
        if isinstance(sensor, ContactSensor):  
            if source in sensor.cfg.prim_path:  
                source_contact = sensor  
            if target in sensor.cfg.prim_path:  
                target_contact = sensor  
      
    if source_contact is None or target_contact is None:  
        return torch.zeros(env.num_envs, device=env.device)  
      
    # Check if there are any contacts  
    source_contacts = torch.any(source_contact.data.current_contact_force_magnitude > 0, dim=1)  
    target_contacts = torch.any(target_contact.data.current_contact_force_magnitude > 0, dim=1)  
      
    # Return 1.0 if both have contacts (assuming they're in contact with each other)  
    return (source_contacts & target_contacts).float()  
  
  
def body_velocity_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:  
    """Compute the L2 norm of the body velocity."""  
    asset = env.scene[asset_cfg.name]  
    if hasattr(asset, "data") and hasattr(asset.data, "root_lin_vel_w"):  
        return torch.norm(asset.data.root_lin_vel_w, dim=-1)  
    return torch.zeros(env.num_envs, device=env.device)  
  
  
def get_phase(env: ManagerBasedRLEnv) -> torch.Tensor:  
    """判断当前处于哪个阶段"""  
    distance = distance_to_target(env, "robot", "ball")  
    # 返回1.0表示击球模式，0.0表示接近模式  
    return (distance <= 1.0).float()  
  
  
def straight_line_imitation(env: ManagerBasedRLEnv) -> torch.Tensor:  
    """计算直线轨迹模仿奖励"""  
    # 获取右手位置  
    robot = env.scene["robot"] 
    right_hand_id = robot.find_bodies("right_rubber_hand")[0]  
    hand_pos = robot.data.body_pos_w[:, right_hand_id, :]  
      
    # 获取球的位置（目标点就是球本身）  
    ball_pos = env.scene["ball"].data.root_pos_w  
      
    # 计算手到球的距离  
    distance_to_ball = torch.norm(ball_pos - hand_pos, dim=1)  
      
    return distance_to_ball  
  
  
def two_phase_reward(env: ManagerBasedRLEnv) -> torch.Tensor:  
    """两阶段奖励函数"""  
    # 获取当前阶段  
    phase = get_phase(env)  
      
    # 接近模式奖励  
    # 现在使用速度命令跟踪奖励来引导机器人接近球  
    # 这里只保留一个小的距离奖励作为辅助  
    distance_reward = -0.1 * distance_to_target(env, "robot", "ball")  
      
    # 击球模式奖励  
    # 接触奖励  
    contact_reward = 10.0 * contact_between_assets(env, "robot/right_hand/paddle", "ball")  
    # 球速度奖励  
    ball_asset_cfg = SceneEntityCfg("ball")  
    velocity_reward = 5.0 * body_velocity_l2(env, ball_asset_cfg)  
    # 直线轨迹模仿奖励  
    imitation_reward = -0.5 * straight_line_imitation(env)  
      
    # 组合奖励  
    # 接近模式：速度命令跟踪奖励（在RewardsCfg中定义）+ 小的距离奖励  
    # 击球模式：接触奖励 + 速度奖励 + 模仿奖励  
    reward = (  
        (1 - phase) * distance_reward +  
        phase * (contact_reward + velocity_reward + imitation_reward)  
    )  
      
    return reward  
  
  
def attach_paddle_to_hand(env: ManagerBasedRLEnv, env_ids: torch.Tensor, hand_link_name: str = "right_wrist_yaw_link") -> None:   
    """Attach paddle to robot hand by setting paddle position relative to hand.  
  
    Simplified version following DeepWiki approach.  
    """  
    # Get robot and paddle  
    robot = env.scene["robot"]  
    paddle = env.scene["paddle"]  
  
    # Get hand pose in world frame  
    hand_body_idx = robot.find_bodies(hand_link_name)[0]  
    hand_pose = robot.data.body_state_w[:, hand_body_idx, :7].squeeze(1)
  
    # Define paddle offset (0.1m in front of hand)  
    paddle_offset = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=env.device)  
  
    # Set paddle pose  
    paddle.write_root_pose_to_sim(hand_pose + paddle_offset)