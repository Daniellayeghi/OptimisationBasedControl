function  x = euler_integration(func,  x, control, time_step)
    t = 0;
    dx = func(t, x, control);
    x(1:3) = x(1:3) + dx(1: 3) * time_step;
    x(4:6) = dx(1:3);
end