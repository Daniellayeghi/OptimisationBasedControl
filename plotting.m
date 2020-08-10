%% Plot Acrobot

SAVE_AR = true;

figure()
ar_ctrl = csvread("acrobot_ctrl.csv");
[row, ~] = size(ar_ctrl);
iteration = linspace(1, row, row)/0.01;
plot(iteration , ar_ctrl(:, 1))
title('Cartpole control trajectory','interpreter','latex', 'FontSize', 12);
xlabel('time (s)','interpreter','latex', 'FontSize', 12);
ylabel('$u$', 'Interpreter', 'latex', 'FontSize', 12);

figure()
ar_pos = csvread("acrobot_pos.csv");
ar_vel = csvread("acrobot_vel.csv");
plot(ar_pos(:, 2), ar_vel(:, 2))
title('Acrobot phase plane','interpreter','latex', 'FontSize', 12);
xlabel('$x$','interpreter','latex', 'FontSize', 12);
ylabel('$\dot{x}$', 'interpreter', 'latex', 'FontSize', 12);

figure()
ar_cost = csvread("acrobot_cost_mpc.csv");
[row, ~] = size(ar_cost);
iteration = linspace(1, row, row);
plot(iteration , ar_cost)
title('Acrobot MPC cost - iteration','interpreter','latex', 'FontSize', 12);
xlabel('MPC iteration','interpreter','latex', 'FontSize', 12);
ylabel('Cost', 'Interpreter', 'latex', 'FontSize', 12);

if(SAVE_AR)
    save("ar_cost.mat", 'ar_cost');
    save("ar_pos.mat", 'ar_pos');
    save("ar_pos.mat", 'ar_vel');
end

%% Plot Cartpole
SAVE_CP = true;

figure()
cp_ctrl = csvread("cartpole_ctrl.csv");
[row, ~] = size(cp_ctrl);
iteration = linspace(1, row, row)/0.01;
plot(iteration , cp_ctrl(:, 1))
title('Cartpole control trajectory','interpreter','latex', 'FontSize', 12);
xlabel('time (s)','interpreter','latex', 'FontSize', 12);
ylabel('$u$', 'Interpreter', 'latex', 'FontSize', 12);

figure()
cp_pos = csvread("cartpole_pos.csv");
cp_vel = csvread("cartpole_vel.csv");
plot(cp_pos(:, 2), cp_vel(:, 2))
title('Cartpole phase plane','interpreter','latex', 'FontSize', 12);
xlabel('$x$','interpreter','latex', 'FontSize', 12);
ylabel('$\dot{x}$', 'interpreter', 'latex', 'FontSize', 12);

figure()
cp_cost = csvread("cartpole_cost_mpc.csv");
[row, ~] = size(cp_cost);
iteration = linspace(1, row, row);
plot(iteration , cp_cost)
title('Cartpole MPC cost - iteration','interpreter','latex', 'FontSize', 12);
xlabel('MPC iteration','interpreter','latex', 'FontSize', 12);
ylabel('Cost', 'Interpreter', 'latex', 'FontSize', 12);

if(SAVE_CP)
    save("cp_cost.mat", 'cp_cost');
    save("cp_vel.mat", 'cp_vel');
    save("cp_pos.mat", 'cp_pos');
end