%% Spring damper mass A B

function x_dot = Pendulum(state, ctrl)
    damping = 0.5; l = 1; mass = 1;
    acc   = 9.81/l*sin(state(1,1)) + (-damping * state(2,1) + ctrl)/(mass*l^2);
    x_dot = [state(2,1); acc];
end