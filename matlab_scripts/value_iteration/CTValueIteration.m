%% Generate data
clc;
clear all;
params   = [5; 0.00001];
x        = [5; 0];
[A, B]   = SpringMassStateSpace(params);
step     = 0.1;
min_pos  = -5; max_pos  = 5;
min_vel  = -5; max_vel  = 5;
min_ctrl = -1; max_ctrl = 1;
pos_arr  = min_pos:step:max_pos;
vel_arr  = min_vel:step:max_vel;
values   = -inf((max_pos - min_pos)/step+1, (max_vel - min_vel)/step+1);

%% Cost params
Q = diag([100, 0.05*10]);
R = diag(10);

%% Simulate system
time_step = 0.05;
time      = 0 :time_step:10;
[t, y]    = ode45(@(t, state)SpringMass(state, 0, params), time, x);

%% Value iteration
for pos_it = 1:1:length(pos_arr)
    for vel_it = 1:1:length(vel_arr)
        curr_state = [pos_arr(pos_it); vel_arr(vel_it)];
        for ctrl = linspace(min_ctrl, max_ctrl, 50)
            curr_state_d = SpringMass(curr_state, ctrl, params);
            curr_state   = curr_state + curr_state_d * time_step;
            cost = curr_state' * Q * curr_state + ctrl * R * ctrl;                
            if values(pos_it, vel_it) < cost
                values(pos_it, vel_it) = cost;
            end
        end
    end
end
 

%% Plot trajectory
plot(time, y);
xlabel("Time s");
ylabel("State");
title("State Trajectory")

[P, V] = meshgrid(pos_arr, vel_arr);
surf(P, V, values);
xlabel("Pos");
ylabel("Vel");
zlabel("Value");
title("Value Function")