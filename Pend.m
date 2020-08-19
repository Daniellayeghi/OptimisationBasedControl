function x_new = Pend(x, u)
    x_d = (u - 0.5*x(2, 1) - 9.81*sin(x(1,1)));
    x_new(2,1) = x(2,1) + x_d * 0.01;
    x_new(1,1) = wrapToPi(x(1,1) + x_new(2,1) * 0.01);
end