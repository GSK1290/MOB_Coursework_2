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
         # ----- Augmented MCL parameters (exponential weight filters)
        self.ALPHA_SLOW = 0.01  # Slow average decay rate (lower = longer memory)
        self.ALPHA_FAST = 0.1    # Fast average decay rate (higher = shorter memory)
        self.w_slow = 0.0        # Slow-decaying average of weights
        self.w_fast = 0.0        # Fast-decaying average of weights
        
        # Kidnapping recovery parameter
        self.RANDOM_PARTICLE_RATIO = 0.25
 
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20

        # ----- Adaptive MCL parameters
        self.MIN_PARTICLES = 300
        self.MAX_PARTICLES = 2000
        self.KLD_EPSILON = 0.05
        self.KLD_Z = 3.0
        self.BIN_SIZE_XY = 0.5
        self.BIN_SIZE_THETA = 0.2
        
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
        """
        Create a random particle within free space on the map.
        Only generates particles in areas that are not obstacles.
        """
        
        particle = Pose()

        map_width = self.occupancy_map.info.width
        map_height = self.occupancy_map.info.height
        map_res = self.occupancy_map.info.resolution
        origin_x = self.occupancy_map.info.origin.position.x
        origin_y = self.occupancy_map.info.origin.position.y

        # Try to find a valid free space position
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            # Generate random position within map bounds
            x = origin_x + random.uniform(0, map_width * map_res)
            y = origin_y + random.uniform(0, map_height * map_res)
            
            # Convert world coordinates to grid coordinates
            grid_x = int((x - origin_x) / map_res)
            grid_y = int((y - origin_y) / map_res)
            
            # Check if within map bounds
            if 0 <= grid_x < map_width and 0 <= grid_y < map_height:
                # Get occupancy value at this grid cell
                # Map data is row-major order: index = y * width + x
                map_index = grid_y * map_width + grid_x
                occupancy_value = self.occupancy_map.data[map_index]
                
                # Check if cell is free space
                # Typically: -1 = unknown, 0 = free, 100 = occupied
                if occupancy_value >= 0 and occupancy_value < 50:  # Free space threshold
                    particle.position.x = x
                    particle.position.y = y
                    particle.position.z = 0.0
                    
                    heading = random.uniform(-math.pi, math.pi)
                    particle.orientation = rotateQuaternion(Quaternion(w=1.0), heading)
                    
                    return particle
            
            attempts += 1
        
        # Fallback: if no free space found after max_attempts, return random position
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

    def _discretize_pose(self, x, y, theta):
        """
        Discretize a pose into a bin for KLD-sampling.
        
        :Args:
            | x: x position
            | y: y position  
            | theta: orientation angle
        :Return:
            | (tuple) bin identifier (x_bin, y_bin, theta_bin)
        """
        x_bin = int(math.floor(x / self.BIN_SIZE_XY))
        y_bin = int(math.floor(y / self.BIN_SIZE_XY))
        theta_bin = int(math.floor(theta / self.BIN_SIZE_THETA))
        
        return (x_bin, y_bin, theta_bin)
    
    def _compute_kld_bound(self, k):
        """
        Compute the KLD-sampling bound for the number of particles.
        Based on Fox (2001): "Adapting the Sample Size in Particle Filters Through KLD-Sampling"
        
        :Args:
            | k: number of non-empty bins
        :Return:
            | (int) minimum number of particles required
        """
        if k <= 1:
            return self.MAX_PARTICLES
        
        k_1 = k - 1
        
        term1 = 1.0 - 2.0 / (9.0 * k_1)
        term2 = math.sqrt(2.0 / (9.0 * k_1)) * self.KLD_Z
        
        M_x = (k_1 / (2.0 * self.KLD_EPSILON)) * math.pow(term1 + term2, 3)
        
        M_x = int(math.ceil(M_x))
        M_x = max(self.MIN_PARTICLES, min(self.MAX_PARTICLES, M_x))
        
        return M_x

    # def update_particle_cloud(self, scan):
        
    #     S = self.particlecloud.poses
    #     M = len(S)
        
    #     RESAMPLE_NOISE_X = 0.02      
    #     RESAMPLE_NOISE_Y = 0.02 
    #     RESAMPLE_NOISE_THETA = 0.02

    #     num_random_particles = int(self.RANDOM_PARTICLE_RATIO * M)
    #     num_resampled_particles = M - num_random_particles
        
    #     weights = []
    #     for pose in S:
    #         w_i = self.sensor_model.get_weight(scan, pose)
    #         weights.append(w_i)
        
    #     total_weight = sum(weights)

    #     if total_weight == 0:
    #         weights = [1.0] * M
    #         total_weight = M
        
    #     normalized_weights = [w / total_weight for w in weights]

    #     self._weights_before_resample = normalized_weights
    #     self._particles_before_resample = [self._copy_pose(pose) for pose in S]
        
    #     c = [normalized_weights[0]]
    #     for i in range(1, M):
    #         c_i = c[i-1] + normalized_weights[i]
    #         c.append(c_i)
        
    #     S_prime = []
    #     u = random.uniform(0, 1.0 / num_resampled_particles)
    #     i = 0
        
    #     for j in range(num_resampled_particles):

    #         while u > c[i]:
    #             i = i + 1
            
    #         selected = S[i]
    #         new_particle = Pose()
            
    #         new_particle.position.x = selected.position.x + random.gauss(0, RESAMPLE_NOISE_X)
    #         new_particle.position.y = selected.position.y + random.gauss(0, RESAMPLE_NOISE_Y)
    #         new_particle.position.z = 0.0
            
    #         old_heading = getHeading(selected.orientation)
    #         new_heading = old_heading + random.gauss(0, RESAMPLE_NOISE_THETA)
    #         new_particle.orientation = rotateQuaternion(Quaternion(w=1.0), new_heading)
            
    #         S_prime.append(new_particle)
            
    #         u = u + 1.0 / num_resampled_particles

    
    #     for j in range(num_random_particles):
    #         random_particle = self._create_random_particle()
    #         S_prime.append(random_particle)
        
    #     self.particlecloud.poses = S_prime

    def update_particle_cloud(self, scan):
        """
        Augmented Adaptive MCL: Combines KLD-sampling with exponential weight filters
        for dynamic particle count adjustment AND kidnapping detection/recovery.
        
        Based on:
        - Fox (2001): "Adapting the Sample Size in Particle Filters Through KLD-Sampling"
        - Fox (2003): "Adapting the Sample Size in Particle Filters Through KLD-Sampling" 
                      (extended version with augmented MCL)
        """
        
        S = self.particlecloud.poses
        M = len(S)
        
        RESAMPLE_NOISE_X = 0.02      
        RESAMPLE_NOISE_Y = 0.02 
        RESAMPLE_NOISE_THETA = 0.02
        
        # Calculate weights for all particles
        weights = []
        for pose in S:
            w_i = self.sensor_model.get_weight(scan, pose)
            weights.append(w_i)
        
        total_weight = sum(weights)

        if total_weight == 0:
            weights = [1.0] * M
            total_weight = M
        
        # Augmented MCL: Update exponential weight filters
        average_weight = total_weight / M
        
        # Initialize filters on first update
        if self.w_slow == 0.0:
            self.w_slow = average_weight
            self.w_fast = average_weight
        else:
            # Exponential moving averages
            self.w_slow += self.ALPHA_SLOW * (average_weight - self.w_slow)
            self.w_fast += self.ALPHA_FAST * (average_weight - self.w_fast)
        
        # Compute random particle injection probability
        # When w_fast << w_slow, particles are performing poorly (possible kidnapping)
        # Inject more random particles to recover
        if self.w_slow > 0.0:
            random_injection_prob = max(0.0, 1.0 - (self.w_fast / self.w_slow))
        else:
            random_injection_prob = 0.0
    
        
        normalized_weights = [w / total_weight for w in weights]

        # Store pre-resample data for estimate_pose
        self._weights_before_resample = normalized_weights[:]
        self._particles_before_resample = [self._copy_pose(pose) for pose in S]
        
        # Build cumulative distribution for low-variance resampling
        c = [normalized_weights[0]]
        for i in range(1, M):
            c_i = c[i-1] + normalized_weights[i]
            c.append(c_i)
        
        # Adaptive MCL: KLD-sampling with Augmented MCL random injection
        S_prime = []
        occupied_bins = set()
        
        # Start with minimum particles
        M_x = self.MIN_PARTICLES
        
        # Initialize low-variance resampling
        u = random.uniform(0, 1.0 / M)
        i = 0
        j = 0
        
        # Resample until we have enough particles
        while j < M_x and len(S_prime) < self.MAX_PARTICLES:
            
            # Augmented MCL: Randomly inject particles with probability based on filter ratio
            if random.random() < random_injection_prob:
                # Add random particle (kidnapping recovery)
                random_particle = self._create_random_particle()
                S_prime.append(random_particle)
                
                # Still track bins for KLD-sampling
                random_heading = getHeading(random_particle.orientation)
                particle_bin = self._discretize_pose(
                    random_particle.position.x,
                    random_particle.position.y,
                    random_heading
                )
                
                if particle_bin not in occupied_bins:
                    occupied_bins.add(particle_bin)
                    k = len(occupied_bins)
                    M_x = self._compute_kld_bound(k)
            
            else:
                # Standard low-variance resampling step
                while i < len(c) and u > c[i]:
                    i = i + 1
                
                # Safety check
                if i >= len(S):
                    i = len(S) - 1
                
                selected = S[i]
                new_particle = Pose()
                
                # Add resampling noise
                new_particle.position.x = selected.position.x + random.gauss(0, RESAMPLE_NOISE_X)
                new_particle.position.y = selected.position.y + random.gauss(0, RESAMPLE_NOISE_Y)
                new_particle.position.z = 0.0
                
                old_heading = getHeading(selected.orientation)
                new_heading = old_heading + random.gauss(0, RESAMPLE_NOISE_THETA)
                new_particle.orientation = rotateQuaternion(Quaternion(w=1.0), new_heading)
                
                S_prime.append(new_particle)
                
                # KLD-sampling: track which bins are occupied
                particle_bin = self._discretize_pose(
                    new_particle.position.x,
                    new_particle.position.y,
                    new_heading
                )
                
                # If new bin, recalculate required particle count
                if particle_bin not in occupied_bins:
                    occupied_bins.add(particle_bin)
                    k = len(occupied_bins)
                    M_x = self._compute_kld_bound(k)
            
            # Increment for low-variance resampling
            u = u + 1.0 / M
            j = j + 1
    
        self.particlecloud.poses = S_prime



    # def estimate_pose(self):
    #     """
    #     This should calculate and return an updated robot pose estimate based
    #     on the particle cloud (self.particlecloud).
        
    #     Create new estimated pose, given particle cloud
    #     E.g. just average the location and orientation values of each of
    #     the particles and return this.
        
    #     Better approximations could be made by doing some simple clustering,
    #     e.g. taking the average location of half the particles after 
    #     throwing away any which are outliers

    #     :Return:
    #         | (geometry_msgs.msg.Pose) robot's estimated pose.
    #     """
       
    #     S = self.particlecloud.poses

    #     weights = self._weights_before_resample
    #     max_weight_idx = weights.index(max(weights))
    #     best_particle = S[max_weight_idx]

    #     curr_x = best_particle.position.x
    #     curr_y = best_particle.position.y
    #     curr_theta = getHeading(best_particle.orientation)

    #     radius = 0.7 
    #     iterations = 5
        
    #     for _ in range(iterations):
    #         weighted_x_sum = 0.0
    #         weighted_y_sum = 0.0
    #         weighted_sin_sum = 0.0
    #         weighted_cos_sum = 0.0
    #         total_influence = 0.0

    #         for i, p in enumerate(S):
                
    #             dx = p.position.x - curr_x
    #             dy = p.position.y - curr_y
    #             dist_sq = dx**2 + dy**2

                
    #             if dist_sq < radius**2:
                    
    #                 influence = weights[i] * math.exp(-dist_sq / (2 * (radius/2)**2))
                    
    #                 weighted_x_sum += p.position.x * influence
    #                 weighted_y_sum += p.position.y * influence
                    
    #                 p_theta = getHeading(p.orientation)
    #                 weighted_sin_sum += math.sin(p_theta) * influence
    #                 weighted_cos_sum += math.cos(p_theta) * influence
                    
    #                 total_influence += influence

    #         if total_influence > 0:
    #             curr_x = weighted_x_sum / total_influence
    #             curr_y = weighted_y_sum / total_influence
    #             curr_theta = math.atan2(weighted_sin_sum, weighted_cos_sum)
    #         else:
    #             break 

    #     estimated_pose = Pose()
    #     estimated_pose.position.x = curr_x
    #     estimated_pose.position.y = curr_y
    #     estimated_pose.position.z = 0.0
    #     estimated_pose.orientation = rotateQuaternion(Quaternion(w=1.0), curr_theta)

    #     return estimated_pose

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
       
        # Use pre-resample particles and weights for consistent estimation
        if len(self._particles_before_resample) == 0:
            S = self.particlecloud.poses
            weights = [1.0 / len(S)] * len(S)
        else:
            S = self._particles_before_resample
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

    