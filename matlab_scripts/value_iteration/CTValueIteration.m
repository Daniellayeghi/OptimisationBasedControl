%% Generate data
clc;
clear all;
close all;
params   = [5; 10];
x        = [5; 0];
step     = 0.5;
min_pos  = -2*pi; max_pos = 2*pi;
min_vel  = -2*pi; max_vel = 2*pi;
min_ctrl = -1; max_ctrl = 1;
pos_arr  = min_pos:step:max_pos;
vel_arr  = min_vel:step:max_vel;
values   = zeros(length(pos_arr), length(vel_arr));
time_step = 0.05;
time      = 0 :time_step:10;
[P, V]    = meshgrid(pos_arr, vel_arr);

%% Cost params
Q = diag([100, 100, 0.05*100]);
ref_state = [0; 1; 0];
R = diag(1);

%% Simulate system Pendulum
[t_p, y_p] = ode45(@(t, state)Pendulum(state, 0), time, [0+0.005; 0]);
cart = [sin(y_p(:, 1)), cos(y_p(:, 1))];

%% Value iteration
for iteration = 1:5
    values_diff = values;
    for pos_it = 1:1:length(pos_arr)
        for vel_it = 1:1:length(vel_arr)
            curr_state = [pos_arr(pos_it); vel_arr(vel_it)];
            for ctrl = linspace(min_ctrl, max_ctrl, 25)
                curr_state_d = Pendulum(curr_state, ctrl);
                curr_state   = curr_state + curr_state_d * time_step;
                curr_state_t = [sin(curr_state(1,1)); cos(curr_state(1,1)); curr_state(2,1)];
                error = ref_state - curr_state_t;
                next_val = interp2(P, V, values, curr_state(1,1), curr_state(2,1));
                if isnan(next_val)
                    next_val = 0;
                end
                current_val = (error' * Q * error) + ctrl' * R * ctrl + next_val;
                if ctrl == min_ctrl && iteration == 1
                    values(pos_it, vel_it) = current_val;
                end
                if current_val < values(pos_it, vel_it)
                    values(pos_it, vel_it) = current_val;
                end
            end
        end
    end
end

%% Plot trajectory
hold on
plot(time, cart);
xlabel("Time s");
ylabel("State");
title("State Trajectory")

%% Plot value function
figure()
surf(P, V, values);
xlabel("Pos");
ylabel("Vel");
zlabel("Value");
title("Value Function")
