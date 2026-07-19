% main.m
% Deep MARL Orchestrator with Policy Step Clipping Constraints

clear; clc; close all;

%% 1. Initialize Environment and Architecture Dimensions
env = MARLEnvWrapper();
num_episodes = 500;          
max_steps_per_episode = 200;  
state_dim = env.StateDim;   
action_dim = env.ActionDim; 
hidden_dim = 64;            

% Stable high-agility online configurations
lr_actor = 0.0008;   
lr_critic = 0.0005;  
discount_factor = 0.99;
clip_epsilon = 0.2; % PPO-style gradient adjustment clip boundary

%% 2. Initialize Deep Actor-Critic Network Weights
W1_layer1 = randn(hidden_dim, state_dim) * sqrt(2/state_dim) * 0.1;
W1_layer2 = randn(action_dim, hidden_dim) * 0.01;
W2_layer1 = randn(hidden_dim, state_dim) * sqrt(2/state_dim) * 0.1;
W2_layer2 = randn(action_dim, hidden_dim) * 0.01;

sigma1 = ones(action_dim, 1) * 0.3;
sigma2 = ones(action_dim, 1) * 0.3;

joint_state_dim = state_dim * 2; 
W_critic_layer1 = randn(hidden_dim, joint_state_dim) * sqrt(2/joint_state_dim) * 0.1;
W_critic_layer2 = randn(1, hidden_dim) * 0.01;

episode_rewards_history = zeros(num_episodes, 2);

fprintf('--- Starting Constrained Deep Actor-Critic Training Loop ---\n');

%% 3. The Global Training Loop
for ep = 1:num_episodes
    states = env.reset(); 
    episode_rewards = zeros(1, 2);
    
    ep_states1 = []; ep_states2 = [];
    ep_h1 = [];      ep_h2 = [];
    ep_actions1 = []; ep_actions2 = [];
    ep_mus1 = [];     ep_mus2 = [];
    ep_rewards1 = []; ep_rewards2 = [];
    
    for step_idx = 1:max_steps_per_episode
        % State Space Normalization Filter
        s1 = states{1}; s2 = states{2};
        s1(1:9) = s1(1:9) / 5.0; s1(12)  = s1(12) / 12.0;   
        s2(1:9) = s2(1:9) / 5.0; s2(12)  = s2(12) / 12.0;   
        
        ep_states1 = [ep_states1; s1];
        ep_states2 = [ep_states2; s2];
        
        % ACTOR NETWORK FORWARD PASS
        net1 = W1_layer1 * s1'; h1 = max(0.01 * net1, net1); mu1 = W1_layer2 * h1;
        ep_h1 = [ep_h1; h1']; ep_mus1 = [ep_mus1; mu1'];
        
        net2 = W2_layer1 * s2'; h2 = max(0.01 * net2, net2); mu2 = W2_layer2 * h2;
        ep_h2 = [ep_h2; h2']; ep_mus2 = [ep_mus2; mu2'];
        
        raw_act1 = mu1 + sigma1 .* randn(action_dim, 1);
        raw_act2 = mu2 + sigma2 .* randn(action_dim, 1);
        
        ep_actions1 = [ep_actions1; raw_act1'];
        ep_actions2 = [ep_actions2; raw_act2'];
        
        act1 = [tanh(raw_act1(1)) * 1.5, tanh(raw_act1(2)) * 1.0];
        act2 = [tanh(raw_act2(1)) * 1.5, tanh(raw_act2(2)) * 1.0];
        
        [next_states, rewards, done] = env.step({act1, act2});
        
        rewards = rewards / 10.0;
        ep_rewards1 = [ep_rewards1; rewards(1)];
        ep_rewards2 = [ep_rewards2; rewards(2)];
        episode_rewards = episode_rewards + rewards;
        
        states = next_states;
        if done, break; end
    end
    
    % Returns-to-go calculations
    T = length(ep_rewards1);
    returns1 = zeros(T, 1); returns2 = zeros(T, 1);
    g1 = 0; g2 = 0;
    for t = T:-1:1
        g1 = ep_rewards1(t) + discount_factor * g1;
        g2 = ep_rewards2(t) + discount_factor * g2;
        returns1(t) = g1; returns2(t) = g2;
    end
    
    % Centralized Advantage Calculation
    joint_states = [ep_states1, ep_states2];
    net_critic = W_critic_layer1 * joint_states';
    h_critic = max(0.01 * net_critic, net_critic);
    state_values = (W_critic_layer2 *h_critic)';      
    
    advantages1 = returns1 - state_values;
    advantages2 = returns2 - state_values;
    
    if T > 1
        advantages1 = (advantages1 - mean(advantages1)) / (std(advantages1) + 1e-5);
        advantages2 = (advantages2 - mean(advantages2)) / (std(advantages2) + 1e-5);
    end
    
    % --- ONLINE STEP-BY-STEP CONSTRAINED UPDATES ---
    for t = 1:T
        % 1. Centralized Critic Update
        critic_error = (returns1(t) + returns2(t))/2 - state_values(t);
        d_critic_layer2 = critic_error * h_critic(:, t)';
        leaky_grad_critic = double(net_critic(:, t) > 0) + 0.01 * double(net_critic(:, t) <= 0);
        d_critic_layer1 = (W_critic_layer2' * critic_error) .* leaky_grad_critic * joint_states(t, :);
        
        W_critic_layer2 = W_critic_layer2 + lr_critic * max(-1.0, min(1.0, d_critic_layer2));
        W_critic_layer1 = W_critic_layer1 + lr_critic * max(-1.0, min(1.0, d_critic_layer1));
        
        % 2. Agent 1 Actor Update (With Ratio Clipping Bounds)
        loss_grad1 = ((ep_actions1(t, :) - ep_mus1(t, :))' ./ (sigma1.^2)) * advantages1(t);
        
        % Clip individual policy parameter adjustments directly
        loss_grad1 = max(-clip_epsilon, min(clip_epsilon, loss_grad1));
        
        dW1_layer2 = loss_grad1 * ep_h1(t, :);
        state_t1 = ep_states1(t, :);
        net_t1 = W1_layer1 * state_t1'; 
        leaky_grad1 = double(net_t1 > 0) + 0.01 * double(net_t1 <= 0);
        dW1_layer1 = (W1_layer2' * loss_grad1) .* leaky_grad1 * state_t1;
        
        W1_layer2 = W1_layer2 + lr_actor * max(-1.0, min(1.0, dW1_layer2));
        W1_layer1 = W1_layer1 + lr_actor * max(-1.0, min(1.0, dW1_layer1));
        
        % 3. Agent 2 Actor Update (With Ratio Clipping Bounds)
        loss_grad2 = ((ep_actions2(t, :) - ep_mus2(t, :))' ./ (sigma2.^2)) * advantages2(t);
        
        % Clip individual policy parameter adjustments directly
        loss_grad2 = max(-clip_epsilon, min(clip_epsilon, loss_grad2));
        
        dW2_layer2 = loss_grad2 * ep_h2(t, :);
        state_t2 = ep_states2(t, :);
        net_t2 = W2_layer1 * state_t2';
        leaky_grad2 = double(net_t2 > 0) + 0.01 * double(net_t2 <= 0);
        dW2_layer1 = (W2_layer2' * loss_grad2) .* leaky_grad2 * state_t2;
        
        W2_layer2 = W2_layer2 + lr_actor * max(-1.0, min(1.0, dW2_layer2));
        W2_layer1 = W2_layer1 + lr_actor * max(-1.0, min(1.0, dW2_layer1));
    end
    
    % Decay exploration standard deviation smoothly
    sigma1 = max(sigma1 * 0.996, 0.08);
    sigma2 = max(sigma2 * 0.996, 0.08);
    
    episode_rewards_history(ep, :) = episode_rewards;
end
figure('Name', 'Optimized Deep MARL Convergence Profile');
plot(smooth(episode_rewards_history(:, 1), 20), 'LineWidth', 2, 'DisplayName', 'Agent 1 Policy');
hold on;
plot(smooth(episode_rewards_history(:, 2), 20), 'LineWidth', 2, 'DisplayName', 'Agent 2 Policy');
grid on;
title('Deep AC Convergence (Policy Step Clipping Constraints)');
xlabel('Training Epoch Number');
ylabel('Scaled Cumulative Return');
legend('Location', 'best');
