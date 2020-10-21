%% Spring damper mass eom
% x = [pos; vel], u = ctrls, parms = [k; b]

function x_dot = SpringMass(x, u, params)
    acc   = -params(1,1) * x(1,1) - params(2,1) * x(2,1) + u; 
    x_dot = [x(2,1); acc];
end