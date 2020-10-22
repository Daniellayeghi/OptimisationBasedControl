%% Generate data
clc;
clear all;
params   = [5; 10];
x        = [5; 0];
[A, B]   = SpringMassStateSpace(params);
step     = 0.1;
min_pos  = -7; max_pos  = 7;
min_vel  = -7; max_vel  =7;
min_ctrl = -1; max_ctrl = 1;
pos_arr  = min_pos:step:max_pos;
vel_arr  = min_vel:step:max_vel;
values   = -inf((max_pos - min_pos)/step+1, (max_vel - min_vel)/step+1);

%% Cost params SpringMass
Q = diag([100, 0.05*10]);
R = diag(10);

%% Simulate system SpringMass
time_step = 0.05;
time      = 0 :time_step:10;
[t, y]    = ode45(@(t, state)SpringMass(state, 0, params), time, x);

%% Cost params SpringMass
Q = diag([100, 100, 0.05*10]);
R = diag(10);

%% Simulate system Pendulum
[t_p, y_p] = ode45(@(t, state)Pendulum(state, 0), time, [0; 0]);
cart = [sin(y_p(:, 1)), cos(y_p(:, 2))];

%% Value iteration
for pos_it = 1:1:length(pos_arr)
    for vel_it = 1:1:length(vel_arr)
        curr_state = [pos_arr(pos_it); vel_arr(vel_it)];
        for ctrl = linspace(min_ctrl, max_ctrl, 1000)
            curr_state_d = Pendulum(curr_state, ctrl);
            curr_state   = curr_state + curr_state_d * time_step;
            curr_state_t = [sin(curr_state(1, 1)); cos(curr_state(1, 1)); curr_state(2,1)];
            cost = curr_state_t' * Q * curr_state_t + ctrl * R * ctrl;                
            if values(pos_it, vel_it) < cost
               values(pos_it, vel_it) = cost;
            end
        end
    end
end

%% Plot trajectory
hold on
plot(time, y_p);
xlabel("Time s");
ylabel("State");
title("State Trajectory")

%% Plot value function
figure()
[P, V] = meshgrid(pos_arr, vel_arr);
surf(P, V, values);
xlabel("Pos");
ylabel("Vel");
zlabel("Value");
title("Value Function")
