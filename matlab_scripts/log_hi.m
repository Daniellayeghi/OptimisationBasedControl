function cost = log_hi(ctrl, lim_hi, g_hi)
if g_hi(ctrl, lim_hi) > 0
    disp(g_hi(ctrl, lim_hi))
    cost = log(g_hi(ctrl, lim_hi));
else
    cost = 0;
end
