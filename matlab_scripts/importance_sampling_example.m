clear al;
close all;

K = 100;
R = 10;
ctrl_sample = rand(K, 1);
% cost = ctrl_sample .* R .* ctrl_sample;
upper_mask = ctrl_sample < 0.25;
lower_mask = ctrl_sample > 0.2;
cost = ~(upper_mask .* lower_mask)*10000;
exp_cost = exp(-cost);
sum_exp_cost = sum(exp_cost);
weights = (exp_cost)/sum_exp_cost;
w_ctrl  = weights.* ctrl_sample;
ctrl_avg = sum(w_ctrl);
r = normrnd(sum(w_ctrl), 0.5, 1, K);
% ctrl_sample .- w_ctrl;

% Add covariance adaption
new_variance = sum((ctrl_sample - ctrl_avg).^2 .* w_ctrl);
y = gaussmf(-1:0.01:1, [sqrt(new_variance), ctrl_avg]);

subplot(1, 3, 1)
histfit(ctrl_sample, K/5)
xlabel("Ctrl")
ylabel("Samples")
subplot(1, 3, 2)
histfit(w_ctrl, K/5)
xlabel("Impartant Controls")
ylabel("Samples")
subplot(1, 3, 3)
histfit(r, K/5)
xlabel("Optimal Control")
ylabel("Samples")

figure();
plot(-1:0.01:1, y)
%% Distributoion Examples:
variance = 0.25;
mean = 0.5;
prob = zeros(100, 1);
iter = 1;
k = 0.0001;

for var = 0:0.01:1
    prob(iter, 1) = 1/sqrt(2*pi*variance^2/k^2)*exp(-(var - mean)^2/(2*variance^2));
    iter = iter + 1;
end

norm = sum(prob);
prob = prob./norm;

plot(0:0.01:1, prob)