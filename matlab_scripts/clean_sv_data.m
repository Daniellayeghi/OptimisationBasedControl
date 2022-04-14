%% Read all value functions
path = "~/Repos/OptimisationBasedControl/data/";
file_state = "doubleintegrator_state_";
file_value = "doubleintegrator_value_";
states = [];
values = [];

for i = 0:19
    fstate = file_state + int2str(i);
    fvalue = file_value + int2str(i);
    
    states = [states; {csvread(path + fstate + ".csv")}];
    values = [values; {csvread(path + fvalue + ".csv")}];
end


% plot3(states(:, 1), states(:, 2), values);
% 
% csvwrite(path + "di_states.csv", states);
% csvwrite(path + "di_values.csv", values);