
classdef MARLEnvWrapper < handle
    properties
        Robots          
        MapWalls        
        Targets         
        StateDim        
        ActionDim       
        PrevDistToTarget % Track previous distances to compute dense progress metrics
    end
    
    methods
        function obj = MARLEnvWrapper()
            obj.MapWalls = [-10.0, 10.0, -10.0, 10.0];
            obj.StateDim = 13;  
            obj.ActionDim = 2;  
            obj.reset();
        end
        
        function states = reset(obj)
            obj.Robots = [ ...
                DifferentialRobot(1, -6.0, 0.0, 0.0), ...   
                DifferentialRobot(2,  6.0, 0.0, pi) ...    
            ];
            obj.Targets = [ 6.0, 0.0; -6.0, 0.0 ];
            obj.PrevDistToTarget = [12.0, 12.0];            
            states = cell(1, 2);
            for i = 1:2
                states{i} = obj.getObservation(i);
            end
        end
        
        function obs = getObservation(obj, agent_idx)
            bot = obj.Robots(agent_idx);
            target = obj.Targets(agent_idx, :);
            lidar_readings = bot.readLiDAR(obj.Robots, obj.MapWalls);
            dx_target = target(1) - bot.x;
            dy_target = target(2) - bot.y;
            dist_to_target = sqrt(dx_target^2 + dy_target^2);
            angle_to_target = atan2(dy_target, dx_target);
            heading_error = wrapToPi(angle_to_target - bot.theta);
            
            obs = [lidar_readings, bot.v, bot.omega, dist_to_target, heading_error];
        end
        
        function [next_states, rewards, done] = step(obj, actions)
            for i = 1:2
                obj.Robots(i).updateKinematics(actions{i}(1), actions{i}(2));
            end  
            next_states = cell(1, 2);
            for i = 1:2
                next_states{i} = obj.getObservation(i);
            end            
            rewards = zeros(1, 2);
            done = false;    
            dist_t1 = next_states{1}(12); 
            dist_t2 = next_states{2}(12);
             for i = 1:2
                bot = obj.Robots(i);
                r_progress = 15.0 * (obj.PrevDistToTarget(i) - next_states{i}(12));                
                % Update historical baseline tracking distance
                obj.PrevDistToTarget(i) = next_states{i}(12);
                % If the robot is far from the goal and stops moving,
                % penalty
                if abs(bot.v) < 0.05 && next_states{i}(12) > 0.5
                    r_lazy = -5.0; 
                else
                    r_lazy = 0.0;
                end
                % Safety Penalty Component
                current_min_d_surface = min(next_states{i}(1:9));
                if current_min_d_surface < 0.8
                    r_collision = -30.0 * (0.8 - current_min_d_surface);
                else
                    r_collision = 0.0;
                end
                % Actuator Smoothness Penalty
                r_energy = -0.1 * (bot.v^2 + bot.omega^2);                
                % Composite Step Reward Formulation
                rewards(i) = r_progress + r_lazy + r_collision + r_energy;                
                % Terminal Check: Spatial crash validation
                if current_min_d_surface <= 0.02
                    rewards(i) = rewards(i) - 500.0; 
                    done = true;
                end
             end
            % Terminal Check: Both agents reached destinations safely
            if dist_t1 < 0.4 && dist_t2 < 0.4
                rewards = rewards + 100.0; 
                done = true;
            end
        end
    end
end
