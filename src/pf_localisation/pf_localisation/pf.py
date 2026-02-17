from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point
from . pf_base import PFLocaliserBase
import math
from . util import rotateQuaternion, getHeading
import random

class PFLocaliser(PFLocaliserBase):
       
    def __init__(self, logger, clock):
        # ----- Call the superclass constructor
        super().__init__(logger, clock)
        
        # ----- Set motion model parameters
        self.NUMBER_PARTICLES = 1000
        self.ODOM_ROTATION_NOISE = 0.05
        self.ODOM_TRANSLATION_NOISE = 0.05
        self.ODOM_DRIFT_NOISE = 0.05
        
        # Kidnapping recovery parameter
        self.RANDOM_PARTICLE_RATIO = 0.25
 
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20
        
        # pre-resampling data (for weighted mean)
        self._particles_before_resample = []
        self._weights_before_resample = []
        
       
    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """

        particle_cloud = PoseArray()
        
        init_x = initialpose.pose.pose.position.x
        init_y = initialpose.pose.pose.position.y
        init_heading = getHeading(initialpose.pose.pose.orientation)

        sigma_x = 0.5
        sigma_y = 0.5
        sigma_theta = 0.3

        for i in range(self.NUMBER_PARTICLES):
            particle = Pose()

            noise_x = random.gauss(0, 1)
            noise_y = random.gauss(0, 1)
            noise_theta = random.gauss(0, 1)
 
            particle.position.x = init_x + sigma_x * noise_x
            particle.position.y = init_y + sigma_y * noise_y
            particle.position.z = 0.0

            noisy_heading = init_heading + sigma_theta * noise_theta
            particle.orientation = rotateQuaternion(Quaternion(w=1.0), noisy_heading)

            particle_cloud.poses.append(particle)

        return particle_cloud
    

    def _create_random_particle(self):
        
        particle = Pose()

        map_width = self.occupancy_map.info.width
        map_height = self.occupancy_map.info.height
        map_res = self.occupancy_map.info.resolution
        origin_x = self.occupancy_map.info.origin.position.x
        origin_y = self.occupancy_map.info.origin.position.y

        particle.position.x = origin_x + random.uniform(0, map_width * map_res)
        particle.position.y = origin_y + random.uniform(0, map_height * map_res)
        particle.position.z = 0.0

        heading = random.uniform(-math.pi, math.pi)
        particle.orientation = rotateQuaternion(Quaternion(w=1.0), heading)

        return particle
    

    def _copy_pose(self, pose):

        new_particle = Pose()

        new_particle.position.x = pose.position.x
        new_particle.position.y = pose.position.y
        new_particle.position.z = pose.position.z
        
        new_particle.orientation.x = pose.orientation.x
        new_particle.orientation.y = pose.orientation.y
        new_particle.orientation.z = pose.orientation.z
        new_particle.orientation.w = pose.orientation.w

        return new_particle


    def update_particle_cloud(self, scan):
        
        S = self.particlecloud.poses
        M = len(S)
        
        RESAMPLE_NOISE_X = 0.02      
        RESAMPLE_NOISE_Y = 0.02 
        RESAMPLE_NOISE_THETA = 0.02

        num_random_particles = int(self.RANDOM_PARTICLE_RATIO * M)
        num_resampled_particles = M - num_random_particles
        
        weights = []
        for pose in S:
            w_i = self.sensor_model.get_weight(scan, pose)
            weights.append(w_i)
        
        total_weight = sum(weights)

        if total_weight == 0:
            weights = [1.0] * M
            total_weight = M
        
        normalized_weights = [w / total_weight for w in weights]

        self._weights_before_resample = normalized_weights
        self._particles_before_resample = [self._copy_pose(pose) for pose in S]
        
        c = [normalized_weights[0]]
        for i in range(1, M):
            c_i = c[i-1] + normalized_weights[i]
            c.append(c_i)
        
        S_prime = []
        u = random.uniform(0, 1.0 / num_resampled_particles)
        i = 0
        
        for j in range(num_resampled_particles):

            while u > c[i]:
                i = i + 1
            
            selected = S[i]
            new_particle = Pose()
            
            new_particle.position.x = selected.position.x + random.gauss(0, RESAMPLE_NOISE_X)
            new_particle.position.y = selected.position.y + random.gauss(0, RESAMPLE_NOISE_Y)
            new_particle.position.z = 0.0
            
            old_heading = getHeading(selected.orientation)
            new_heading = old_heading + random.gauss(0, RESAMPLE_NOISE_THETA)
            new_particle.orientation = rotateQuaternion(Quaternion(w=1.0), new_heading)
            
            S_prime.append(new_particle)
            
            u = u + 1.0 / num_resampled_particles

    
        for j in range(num_random_particles):
            random_particle = self._create_random_particle()
            S_prime.append(random_particle)
        
        self.particlecloud.poses = S_prime


    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
        """
       
        S = self.particlecloud.poses

        weights = self._weights_before_resample
        max_weight_idx = weights.index(max(weights))
        best_particle = S[max_weight_idx]

        curr_x = best_particle.position.x
        curr_y = best_particle.position.y
        curr_theta = getHeading(best_particle.orientation)

        radius = 0.5 
        iterations = 5
        
        for _ in range(iterations):
            weighted_x_sum = 0.0
            weighted_y_sum = 0.0
            weighted_sin_sum = 0.0
            weighted_cos_sum = 0.0
            total_influence = 0.0

            for i, p in enumerate(S):
                
                dx = p.position.x - curr_x
                dy = p.position.y - curr_y
                dist_sq = dx**2 + dy**2

                
                if dist_sq < radius**2:
                    
                    influence = weights[i] * math.exp(-dist_sq / (2 * (radius/2)**2))
                    
                    weighted_x_sum += p.position.x * influence
                    weighted_y_sum += p.position.y * influence
                    
                    p_theta = getHeading(p.orientation)
                    weighted_sin_sum += math.sin(p_theta) * influence
                    weighted_cos_sum += math.cos(p_theta) * influence
                    
                    total_influence += influence

            if total_influence > 0:
                curr_x = weighted_x_sum / total_influence
                curr_y = weighted_y_sum / total_influence
                curr_theta = math.atan2(weighted_sin_sum, weighted_cos_sum)
            else:
                break 

        estimated_pose = Pose()
        estimated_pose.position.x = curr_x
        estimated_pose.position.y = curr_y
        estimated_pose.position.z = 0.0
        estimated_pose.orientation = rotateQuaternion(Quaternion(w=1.0), curr_theta)

        return estimated_pose
        