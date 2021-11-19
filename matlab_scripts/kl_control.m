%% Load data
clear all;
close all;

%% Load files if exist
% ctrl_desired = csvread("cartpole_ctrl_check_KL13.csv");
% ctrl_pi_current = csvread("current_ctrl.csv");
% ctrl_sample_time = csvread("ctrl_sample_time.csv");
% ctrl_sample_cost = csvread("traj_cost.csv");
% reg_traj = csvread("_hessian_.csv");
% [K, T] = size(ctrl_sample_time);

%% KL Control example
K=20; T = 50;
reg_traj = ones(1, T) * 400;
mean = 0; variance = 0.1;
hess_reg_diag = diag(reg_traj);
sample_reg = eye(T) * 1/variance;
ctrl_desired = sin((1:T)./2)' + 0.5 * sin((1:T)./2 - 0.75)';

%% Compute the cost from trajectory
lambda = 0.5;
traj_cost = zeros(K, 1);
guess_ctrl = zeros(T, 1);
current_ctrl = zeros(1, T);

fig_1 = figure();
error = 100; iteration = 1;
while iteration < 100
    random_ctrl = normrnd(0,sqrt(variance),[K,T]);
    for sample = 1:1:K
        new_ctrl = current_ctrl + random_ctrl(sample, :);
%         new_ctrl = current_ctrl + ctrl_sample_time(sample, :);
        traj_cost(sample, 1) = (ctrl_desired' - new_ctrl)... 
            * hess_reg_diag * (ctrl_desired' - new_ctrl)'...
            + (new_ctrl - current_ctrl) * sample_reg ...
            * (new_ctrl - current_ctrl)';
    end
    traj_cost = traj_cost * 0.5 * lambda;
    min_cost = min(traj_cost);
    exp_cost = exp(-1/lambda * (traj_cost - min_cost));
    norm_const = sum(exp_cost);
    weights = exp_cost ./ norm_const;
%     current_ctrl = current_ctrl +(weights' * ctrl_sample_time);
    current_ctrl = current_ctrl +(weights' * random_ctrl);
    clf(fig_1);
    subplot(2, 1, 1)
    plot(ctrl_desired);
    hold on 
    plot(current_ctrl);
%     subplot(2, 1, 2)
%     plot(ctrl_desired);
%     hold on
%     plot(ctrl_pi_current);
    drawnow;
    error = rms(current_ctrl - ctrl_desired');
    fprintf("Error: %d error at iteration: %i \n", error, iteration);
    variance = variance * 95/100;
    iteration = 1 + iteration;
end