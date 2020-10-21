%% Spring damper mass A B

function [A, B] = SpringMassStateSpace(params)
    A = [0, 1; -params(1,1), -params(2,1)];
    B = [0;1];
end