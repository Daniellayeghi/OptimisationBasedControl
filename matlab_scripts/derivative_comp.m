%% Compare the derivative information on test function
syms x real
gl_func = sin(10*pi*x)/(2*x)+(x-1)^4;
gl_func_grad = gradient(gl_func, x);
gl_func_hess = gradient(gl_func_grad, x);

subplot(1,3,1);
fplot(gl_func, [0.5 2.5]);
title("Value");
subplot(1,3,2);
fplot(gl_func_grad, [0.5 2.5]);
title("Gradient");
subplot(1,3,3);
fplot(gl_func_hess, [0.5 2.5]);
title("Hessian");

%% Compute minimum via gd at different points
obj_func = @(x)((x - 1)^4 + sin(10*pi*x)/(2*x));
fmincon(obj_func, 2.5, [], []);