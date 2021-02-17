function  x = euler_integration(func,  x, control, time_step)
    dx = func(x, control);
    x(1:3) = x(1:3) + dx(4: 6) * time_step;
    x(4:6) = dx(4:6);
end