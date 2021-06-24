%% Temporal Averaging


vars    = linspace(100, 200, 100);
results = zeros(1, 100);
result  = 0;
weight_sum = 0;

for i = 1:length(vars)
    weight = (length(vars) - (i-1))/length(vars);
    weight_sum = weight_sum + weight; 
    result =  result + (weight * vars(i));
    vars(i) = result/weight_sum;
end
