% DifferentialRobot.m
classdef DifferentialRobot < handle
    properties (Constant = true)
        ROBOT_MASS = 15.0;       
        ROBOT_RADIUS = 0.3;      
        ROBOT_INERTIA = 0.5 * 15.0 * (0.3^2); 
        TIME_STEP = 0.05;        
        MAX_LINEAR_VEL  = 1.5;    
        MAX_ANGULAR_VEL = 1.0;    
        MAX_LINEAR_ACC  = 2.0;    
        MAX_ANGULAR_ACC = 3.0;    
        LIDAR_NUM_RAYS  = 9;       
        LIDAR_MAX_RANGE = 5.0;     
    end
    properties
        id          
        x           
        y           
        theta       
        v           
        omega       
    end
    methods
        function obj = DifferentialRobot(robot_id, start_x, start_y, start_theta)
            if nargin > 0
                obj.id = robot_id;
                obj.x = start_x;
                obj.y = start_y;
                obj.theta = start_theta;
                obj.v = 0.0;
                obj.omega = 0.0;
            end
        end
        
        function scan_distances = readLiDAR(obj, all_robots, map_walls)
            scan_distances = ones(1, obj.LIDAR_NUM_RAYS) * obj.LIDAR_MAX_RANGE;
            ray_angles = linspace(-pi/2, pi/2, obj.LIDAR_NUM_RAYS);             
            for k = 1:obj.LIDAR_NUM_RAYS
                global_ray_angle = obj.theta + ray_angles(k);
                for j = 1:length(all_robots)
                    other_bot = all_robots(j);
                    if other_bot.id == obj.id
                        continue; 
                    end
                    dx = other_bot.x - obj.x;
                    dy = other_bot.y - obj.y;
                    distance_to_center = sqrt(dx^2 + dy^2);
                    angle_to_other = atan2(dy, dx);
                    angular_error = wrapToPi(global_ray_angle - angle_to_other);         
                    if abs(angular_error) < atan2(other_bot.ROBOT_RADIUS, distance_to_center)
                        detected_dist = distance_to_center - other_bot.ROBOT_RADIUS;
                        if detected_dist > 0 && detected_dist < scan_distances(k)
                            scan_distances(k) = detected_dist;
                        end
                    end
                end
                x_ray_max = obj.x + obj.LIDAR_MAX_RANGE * cos(global_ray_angle);
                y_ray_max = obj.y + obj.LIDAR_MAX_RANGE * sin(global_ray_angle);  
                if x_ray_max < map_walls(1), scan_distances(k) = min(scan_distances(k), abs(obj.x - map_walls(1))); end
                if x_ray_max > map_walls(2), scan_distances(k) = min(scan_distances(k), abs(map_walls(2) - obj.x)); end
                if y_ray_max < map_walls(3), scan_distances(k) = min(scan_distances(k), abs(obj.y - map_walls(3))); end
                if y_ray_max > map_walls(4), scan_distances(k) = min(scan_distances(k), abs(map_walls(4) - obj.y)); end
            end
        end
        function [safe_v, safe_omega] = clampInputs(obj, target_v, target_omega)
            req_linear_acc  = (target_v - obj.v) / obj.TIME_STEP;
            req_angular_acc = (target_omega - obj.omega) / obj.TIME_STEP;
            clamped_linear_acc  = max(-obj.MAX_LINEAR_ACC,  min(obj.MAX_LINEAR_ACC,  req_linear_acc));
            clamped_angular_acc = max(-obj.MAX_ANGULAR_ACC, min(obj.MAX_ANGULAR_ACC, req_angular_acc));
            safe_v     = obj.v     + clamped_linear_acc  * obj.TIME_STEP;
            safe_omega = obj.omega + clamped_angular_acc * obj.TIME_STEP;
            safe_v     = max(-obj.MAX_LINEAR_VEL,  min(obj.MAX_LINEAR_VEL,  safe_v));
            safe_omega = max(-obj.MAX_ANGULAR_VEL, min(obj.MAX_ANGULAR_VEL, safe_omega));
        end
        
        function updateKinematics(obj, commanded_v, commanded_omega)
            [safe_v, safe_omega] = obj.clampInputs(commanded_v, commanded_omega);
            x_dot = safe_v * cos(obj.theta);
            y_dot = safe_v * sin(obj.theta);
            theta_dot = safe_omega;
            obj.x = obj.x + x_dot * obj.TIME_STEP;
            obj.y = obj.y + y_dot * obj.TIME_STEP;
            obj.theta = obj.theta + theta_dot * obj.TIME_STEP;
            obj.theta = wrapToPi(obj.theta);
            obj.v = safe_v;
            obj.omega = safe_omega;
        end
    end
end
