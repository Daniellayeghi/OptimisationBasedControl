addpath("../data");

set(groot,'defaultLineLineWidth',1.25)
close all;
clear all; clc;

%% Total Ctrl Effort plots
task_name = "Finger Spinner";
file_name_pi = "finger_ng_ctrl_pi_ddp0.csv";
file_name_ddp = "finger_ng_ctrl_ddp0.csv";
file_name_pi_ddp = "finger_ng_ctrl_pi_ddp1.csv";
legends = ["PI", "ILQG", "PI-ILQG"];

mkdir results;
copyfile  ../data/finger_ng_ctrl_* results; 


fig_ctrl_effort = figure();
ctrl_pi = csvread(file_name_pi);
[row, col] = size(ctrl_pi);
ctrl_pi = ctrl_pi(:, 1:col-1); 
ctrl_ilqg = csvread(file_name_ddp); ctrl_ilqg = ctrl_ilqg(:, 1:col-1);
ctrl_pi_ilqg = csvread(file_name_pi_ddp ); ctrl_pi_ilqg = ctrl_pi_ilqg(:, 1:col-1);
[it_pi, ~] = size(ctrl_pi); [it_ilqg, ~] = size(ctrl_ilqg); [it_pi_ilqg, ~] = size(ctrl_pi_ilqg);


[total_ctrl_eff, ind] = sort([sum(sum(abs(ctrl_pi)')') sum(sum(abs(ctrl_ilqg)')') sum(sum(abs(ctrl_pi_ilqg)')')]);
X = categorical({'Controllers'});
bar(X, total_ctrl_eff);
title(task_name + ' Total Control Effor Comparison','interpreter','latex', 'FontSize', 12);
ylabel('Total Control Effort', 'interpreter', 'latex', 'FontSize', 12);
legend(legends(ind(1)), legends(ind(2)), legends(ind(3)));

fig_iteration = figure();
total_it = sort([it_pi it_ilqg it_pi_ilqg]);
bar(X, total_it);
title(task_name + ' Total MPC Iteration','interpreter','latex', 'FontSize', 12);
ylabel('MPC Iteration', 'interpreter', 'latex', 'FontSize', 12);
legend("PI-ILQG", "PI", "ILQG");

print(fig_ctrl_effort,task_name + 'ctrl_eff.png','-dpng','-r600');
print(fig_iteration,task_name + 'mpc_iteration.png','-dpng','-r600');


%% Cost
task_name = "Finger Spinner";
file_name_pi = "finger_ng_cost_mpc_pi_ddp0.csv";
file_name_ddp = "finger_ng_cost_mpc_ddp0.csv";
file_name_pi_ddp = "finger_ng_cost_mpc_pi_ddp1.csv";

mkdir results;
copyfile  ../data/finger_ng_cost* results; 

fig_cost = figure();
cost_pi = csvread(file_name_pi); cost_pi = cost_pi(:, 1); 
cost_ilqg = csvread(file_name_ddp); cost_ilqg = cost_ilqg(:, 1);
cost_pi_ilqg = csvread(file_name_pi_ddp ); cost_pi_ilqg = cost_pi_ilqg(:, 1);
[it_pi, ~] = size(cost_pi); [it_ilqg, ~] = size(cost_ilqg); [it_pi_ilqg, ~] = size(cost_pi_ilqg);
total_it = max([it_pi, it_ilqg, it_pi_ilqg]);

[total_cost, ind] = sort([sum(abs(cost_pi)) sum(abs(cost_ilqg)) sum(abs(cost_pi_ilqg))]);
X = categorical({'Costs'});
bar(X, total_cost);
title(task_name + ' Total Control Cost Comparison','interpreter','latex', 'FontSize', 12);
ylabel('Total Control Cost', 'interpreter', 'latex', 'FontSize', 12);
legend(legends(ind(1)), legends(ind(2)), legends(ind(3)));

fig_cost_traj= figure();
plot(linspace(1, it_pi, it_pi)*0.01, cost_pi)
hold on;
plot(linspace(1, it_ilqg, it_ilqg)*0.01, cost_ilqg)
hold on;
plot(linspace(1, it_pi_ilqg, it_pi_ilqg)*0.01, cost_pi_ilqg)
title(task_name + " Cost",'interpreter','latex', 'FontSize', 12);
xlabel('Time (s)','interpreter','latex', 'FontSize', 12);
ylabel('Cost', 'Interpreter', 'latex', 'FontSize', 12);
legend("PI", "ILQG", "PI-ILQG");

print(fig_cost,task_name + 'cost.png','-dpng','-r600');
print(fig_cost_traj, task_name + 'cost_traj.png','-dpng','-r600');
