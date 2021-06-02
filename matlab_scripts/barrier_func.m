%% Barrier Cost Function
close all
clear all

g_lo = @(control, lim_lo) lim_lo - control;
g_hi = @(control, lim_hi) control - lim_hi;
inv_barrier = @(ctrl, lim_hi, lim_lo) -(1/g_lo(ctrl, lim_lo) + 1/g_hi(ctrl, lim_hi));
log_barrier = @(ctrl, lim_hi, lim_lo) -(log_lo(ctrl, lim_lo, g_lo) + log_hi(ctrl, lim_hi, g_hi));
control = (-10:0.01:10);


for row = 1:length(control)
    res_inv(row) = (inv_barrier(control(row), 1, -1) + 0.1 * control(row)^4);
    res_log(row) = 1000*(log_barrier(control(row), 1, -1) + 0.5* control(row)^2);
end


% plot(control, res_inv);
% hold on
plot(control, res_log);




