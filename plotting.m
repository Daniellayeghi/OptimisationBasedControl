clear all
clc

%% Plot Acrobot iLQR
SAVE_AR = true;

fig_ilqr_ar_1 = figure();
ar_ctrl = csvread("acrobot_ctrl.csv");
[row, ~] = size(ar_ctrl);
iteration = linspace(1, row, row)/0.01;
plot(iteration , ar_ctrl(:, 1))
title('Cartpole control trajectory','interpreter','latex', 'FontSize', 12);
xlabel('time (s)','interpreter','latex', 'FontSize', 12);
ylabel('$u$', 'Interpreter', 'latex', 'FontSize', 12);

fig_ilqr_ar_2 = figure();
ar_pos = csvread("acrobot_pos.csv");
ar_vel = csvread("acrobot_vel.csv");
plot(ar_pos(:, 2), ar_vel(:, 2))
title('Acrobot phase plane','interpreter','latex', 'FontSize', 12);
xlabel('$x$','interpreter','latex', 'FontSize', 12);
ylabel('$\dot{x}$', 'interpreter', 'latex', 'FontSize', 12);

fig_ilqr_ar_3 = figure();
ar_cost = csvread("acrobot_cost_mpc.csv");
[row, ~] = size(ar_cost);
iteration = linspace(1, row, row);
plot(iteration , ar_cost)
title('Acrobot MPC cost - iteration','interpreter','latex', 'FontSize', 12);
xlabel('MPC iteration','interpreter','latex', 'FontSize', 12);
ylabel('Cost', 'Interpreter', 'latex', 'FontSize', 12);

print(fig_ilqr_ar_1,'ar_ctrl_ilqr.png','-dpng','-r600');
print(fig_ilqr_ar_2,'ar_pp_ilqr.png','-dpng','-r600');
print(fig_ilqr_ar_3,'ar_cost_ilqr.png','-dpng','-r600');

if(SAVE_AR)
    save("ar_cost.mat", 'ar_cost');
    save("ar_pos.mat", 'ar_pos');
    save("ar_pos.mat", 'ar_vel');
end

%% Plot Cartpole iLQR
SAVE_CP = true;

fig_ilqr_cp_1 = figure();
cp_ctrl = csvread("cartpole_ctrl.csv");
[row, ~] = size(cp_ctrl);
iteration = linspace(1, row, row)/0.01;
plot(iteration , cp_ctrl(:, 1))
title('Cartpole control trajectory','interpreter','latex', 'FontSize', 12);
xlabel('time (s)','interpreter','latex', 'FontSize', 12);
ylabel('$u$', 'Interpreter', 'latex', 'FontSize', 12);

fig_ilqr_cp_2 = figure();
cp_pos = csvread("cartpole_pos.csv");
cp_vel = csvread("cartpole_vel.csv");
plot(cp_pos(:, 2), cp_vel(:, 2))
title('Cartpole phase plane','interpreter','latex', 'FontSize', 12);
xlabel('$x$','interpreter','latex', 'FontSize', 12);
ylabel('$\dot{x}$', 'interpreter', 'latex', 'FontSize', 12);

fig_ilqr_cp_3 = figure();
cp_cost = csvread("cartpole_cost_mpc.csv");
[row, ~] = size(cp_cost);
iteration = linspace(1, row, row);
plot(iteration , cp_cost)
title('Cartpole MPC cost - iteration','interpreter','latex', 'FontSize', 12);
xlabel('MPC iteration','interpreter','latex', 'FontSize', 12);
ylabel('Cost', 'Interpreter', 'latex', 'FontSize', 12);

print(fig_ilqr_cp_1,'cp_ctrl_ilqr.png','-dpng','-r600');
print(fig_ilqr_cp_2,'cp_pp_ilqr.png','-dpng','-r600');
print(fig_ilqr_cp_3,'cp_cost_ilqr.png','-dpng','-r600');

if(SAVE_CP)
    save("cp_cost.mat", 'cp_cost');
    save("cp_vel.mat", 'cp_vel');
    save("cp_pos.mat", 'cp_pos');
end


%% Plot Cartpole MPPI
SAVE_CP_MPPI = true;

fig_mppi_1 = figure();
cp_ctrl_mppi = csvread("cartpole_ctrl_mppi.csv");
[row, ~] = size(cp_ctrl_mppi);
iteration = linspace(1, row, row)/0.01;
plot(iteration , cp_ctrl_mppi(:, 1))
title('Cartpole control trajectory','interpreter','latex', 'FontSize', 12);
xlabel('time (s)','interpreter','latex', 'FontSize', 12);
ylabel('$u$', 'Interpreter', 'latex', 'FontSize', 12);

fig_mppi_2 = figure();
cp_pos_mppi = csvread("cartpole_pos_mppi.csv");
cp_vel_mppi = csvread("cartpole_vel_mppi.csv");
plot(cp_pos_mppi(:, 2), cp_vel_mppi(:, 2))
title('Cartpole phase plane','interpreter','latex', 'FontSize', 12);
xlabel('$x$','interpreter','latex', 'FontSize', 12);
ylabel('$\dot{x}$', 'interpreter', 'latex', 'FontSize', 12);

fig_mppi_3 = figure();
cp_cost_mppi = csvread("cartpole_cost_mpc_mppi.csv");
[row, ~] = size(cp_cost_mppi);
iteration = linspace(1, row, row);
plot(iteration , cp_cost_mppi)
title('Cartpole MPC cost - iteration','interpreter','latex', 'FontSize', 12);
xlabel('MPC iteration','interpreter','latex', 'FontSize', 12);
ylabel('Cost', 'Interpreter', 'latex', 'FontSize', 12);

print(fig_mppi_1,'cp_ctrl_mppi.png','-dpng','-r600');
print(fig_mppi_2,'cp_pp_mppi.png','-dpng','-r600');
print(fig_mppi_3,'cp_cost_mppi.png','-dpng','-r600');

if(SAVE_CP_MPPI)
    save("cp_cost_mppi.mat", 'cp_cost_mppi');
    save("cp_vel_mppi.mat", 'cp_vel_mppi');
    save("cp_pos_mppi.mat", 'cp_pos_mppi');
end
