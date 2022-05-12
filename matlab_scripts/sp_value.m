%% Value function learning with sparse ID
clear all;
close all;
t = 0:0.01:10;
x_0 = [1; 0];
u = 0;
v = [];
sv = [];
sj = [];
sj_c = [];
running_cost = @(x, u)(quad_cost(x(1)) + quad_cost(x(2)) + quad_cost(u^2));

for i = 1:100
    x_0 = [rand; 0];
    [t, x] = ode45(@point_mass, t, x_0);
    u1 = arrayfun(@(xi) -(xi(1)), x(:, 1));
    u2 = arrayfun(@(xi) -(sqrt(3) * x(2)), x(:, 2));
    u = u1 + u2;
    v = compute_values(x, u); 
    j = compute_cst2go(x);
    temp = [x, j];
    temp = remove_mid_elems(temp, 3);
    sj_c = [sj_c; temp];
    sv = [sv, [x, v/max(v)]];
    sj = [sj, [x, j/max(j)]];
end

sj_c  = sortrows(sj_c, 3);

figure();
plot3(sv(:, 1:3:end), sv(:, 2:3:end), sv(:, 3:3:end));

figure();
plot3(sj(:, 1:3:end), sj(:, 2:3:end), sj(:, 3:3:end));

parent = '/home/daniel/Repos/OptimisationBasedControl/data/';
dlmwrite('di_cst2go.csv', sj_c, 'delimiter', ',', 'precision', 15);

%% Function Defs
function xd = point_mass(t, x)
    u = ctrl_cb(x);
    xd = [x(2); u];
end

function u = ctrl_cb(x)
    u = -(x(1) + sqrt(3) * x(2));
end

function cst = quad_cost(x)
    cst = x^2;
end

function v = compute_values(x, u)
    quad_cost = @(x)(x^2);
    pos_c = arrayfun(@quad_cost, x(:, 1));
    vel_c = arrayfun(@quad_cost, x(:, 2));
    ctr_c = arrayfun(@quad_cost, u(:, 1));
    
    v= [];
    
    for i = 1:length(x)
        cst = sum(pos_c(i:end, 1)) + sum(vel_c(i:end, 1));
        v = [v; cst];
    end
end


function cst2go = compute_cst2go(x)
    J = @(x)(sqrt(3)*x(1)^2 + 2*x(1)*x(2) + sqrt(3)*x(2)^2);
    cst2go = [];
    for i = 1:length(x)
        xi = x(i, :);
        cst2go = [cst2go; J(xi)]; 
    end 
end

function arr = remove_mid_elems(arr, times)
    for sparse = 1:times
        arr(1:2:end-1, :) = []; 
    end
end