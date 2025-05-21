from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1Cfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.05] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_joint' : -0.2,   

           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_joint' : -0.2,   

           'torso_joint' : 0., 

           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,

           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
        }
       
    class commands ( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 2.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        # stiffness = {'hip_yaw': 200,
        #              'hip_roll': 200,
        #              'hip_pitch': 200,
        #              'knee': 300,
        #              'ankle': 40,
        #              'torso': 300,
        #              'shoulder': 100,
        #              "elbow":100,
        #              }  # [N*m/rad]
        # damping = {  'hip_yaw': 5,
        #              'hip_roll': 5,
        #              'hip_pitch': 5,
        #              'knee': 6,
        #              'ankle': 2,
        #              'torso': 6,
        #              'shoulder': 2,
        #              "elbow":2,
        #              }  # [N*m/rad]  # [N*m*s/rad]
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 150,
                     'knee': 200,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 150,
                     "elbow":100,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'torso': 6,
                     'shoulder': 2,
                     "elbow":2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale_low = 0.25
        action_scale_up = 0.25
        curriculum = 0.25e-2
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4 # 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf'
        name = "h1"
        foot_name = "ankle"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis", "shoulder", "hip", "knee"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False


    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            termination = -1.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            lin_vel_z = -0.5
            ang_vel_xy = -0.025
            orientation = -1.25
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -0.1
            foot_clearance = -0.25
            action_rate = -0.01
            smoothness = -0.01
            collision = -0.0
            torques = -2.5e-6
            dof_vel = -1e-4
            dof_pos_limits = -2.0
            dof_vel_limits = -0.1
            torque_limits = -0.1
            stumble = -3.
            joint_tracking = -0.25
            arm_deviation = -0.1
            waist_deviation = -0.5
            hip_deviation = -0.5
            feet_lateral_dist = 2.5
            feet_slip = -0.25
            no_fly = 0.25
            # feet_contact_forces = -2.5e-4
            feet_parallel = -2.5
            feet_ground_parallel = -2.
            contact_momentum = -2.5e-4

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.05
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.8
        min_foot_dist = 0.5

        
class H1CfgPPO( LeggedRobotCfgPPO ):
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 0.2

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'h1'
        max_iterations = 4000
        
        
    class policy ( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        # actor_hidden_dims = [512*4, 256*4, 128*4]
        critic_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [512*4, 256*4, 128*4]
  
