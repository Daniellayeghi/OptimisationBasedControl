function cost = log_lo(ctrl, lim_lo, g_lo)
if g_lo(ctrl, lim_lo) > 0
    cost = log(g_lo(ctrl, lim_lo));
else
    cost = 0;
end

