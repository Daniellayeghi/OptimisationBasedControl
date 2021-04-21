clear all; 

%% Point mass model
force = [0,0,0];
state_i = [0; 0; 0; 1; 0; 0];
model.mass = 1;

%% ODE integeration
% time = linspace(0, 1, 2000);
% [t, state] = ode45(@(t, state) point_mass(t, state, force), time, state_i);

%% Euler integration
x = zeros(2000, 6);
for i= 1:2000
   state_i = euler_integration(@point_mass, state_i, force, 0.0005);
   x(i, :) = state_i';
end

plot3(x(:, 1), x(:, 2), x(:, 3))
