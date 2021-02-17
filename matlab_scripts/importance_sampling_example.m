clear al;
close all;

K = 100;
R = 10;
ctrl = rand(K, 1);
% cost = ctrl * R .* ctrl;
upper_mask = ctrl < 0.5;
lower_mask = ctrl > 0.48;
cost = ~(upper_mask .* lower_mask)*10000;
exp_cost = exp(-cost);
sum_exp_cost = sum(exp_cost);
weights = (exp_cost)/sum_exp_cost;
w_ctrl  = weights.* ctrl;
r = normrnd(sum(w_ctrl), 0.5, 1, K);

subplot(1, 3, 1)
histfit(ctrl, K/5)
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