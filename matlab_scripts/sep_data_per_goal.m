%% This script separates data files based on goals with the assumption that
% they are written in the following format ...start_#_#_goal...
clear all;
close all; 
clc; 

goal_sfix = "_goal";
start_pfix = "start_";
sep = "_";

f_ctrl_key = "doubleintegrator_ctrl";
f_state_key = "doubleintegrator_state";
path = "~/Repos/OptimisationBasedControl/data/";
u_files = dir(path + sprintf("*%s*", f_ctrl_key) + ".csv");
x_files = dir(path + sprintf("*%s*", f_state_key) + ".csv");

[~, idx] = find([u_files.bytes] > 0);
u_files = u_files(idx);

[~, idx] = find([x_files.bytes] > 0);
x_files = x_files(idx);

assert(length(x_files) == length(u_files));

%% Extract goals and init
file_names = string(length(u_files));
start_end = string(length(u_files));
goals = string(length(u_files));
start = string(length(u_files));
start_goal = [];
start_goal_s = string(ones(length(u_files), 2));


for i  = 1:length(u_files)
        file_names(i) = u_files(i).name;
        start_end = extractBefore(file_names(i), goal_sfix);
        start_end = extractAfter(start_end, start_pfix);
        goals(i) = extractAfter(start_end, sep);
        start(i) = extractBefore(start_end, sep);
        start_goal(i, :) = [str2double(start(i)), str2double(goals(i))];
        start_goal_s(i, :) = [start(i), goals(i)];
end

ctrl_data = [];
state_data = [];
[r, ~] = size(start_goal_s);
for i = 1:r
    ctrl_key = start_goal_s(i, 1) + "_" + start_goal_s(i, 2) + goal_sfix + "_" + f_ctrl_key;
    state_key = start_goal_s(i, 1) + "_" + start_goal_s(i, 2) +  goal_sfix + "_" + f_state_key;
    f_uname = match_file_to_name(ctrl_key, path);
    f_xname = match_file_to_name(state_key, path);
    new_ctrl = csvread(path + f_uname);
    new_state = csvread(path + f_xname);
    [ctrl_data, new_ctrl] = cp_to_match_zero(ctrl_data, new_ctrl);
    [state_data, new_state] = cp_to_match_final(state_data, new_state);
    ctrl_data = [ctrl_data, new_ctrl];
    state_data = [state_data, new_state];
end

f_u_name = "ctrl_files_di.csv";
f_d_name = "desc_files_di.csv";
f_x_name = "state_files_di.csv";

dlmwrite(path + f_u_name, ctrl_data, 'delimiter', ',', 'precision', 10);
dlmwrite(path + f_d_name, start_goal, 'delimiter', ',', 'precision', 10);
dlmwrite(path + f_x_name, state_data, 'delimiter', ',', 'precision', 10);

%% Utility functions
function name = match_file_to_name(key, path)
    f = dir(path + sprintf("*%s*", key) + ".csv");
    name = f.name;
end


function [arr1, arr2] = cp_to_match_zero(arr1, arr2)
    [r1, c1] = size(arr1);
    [r2, c2] = size(arr2);

    if r1 ~= r2 && ~isempty(arr2) && ~isempty(arr1) 
       extra = r1 - r2;
       if extra > 0
           arr2 = [arr2; zeros(abs(extra), c2)];
       elseif extra < 0
           arr1 = [arr1; zeros(abs(extra), c1)];
       end
    end
end


function [arr1, arr2] = cp_to_match_final(arr1, arr2)
    [r1, c1] = size(arr1);
    [r2, c2] = size(arr2);

    if r1 ~= r2 && ~isempty(arr2) && ~isempty(arr1) 
       extra = r1 - r2;
       if extra > 0
           arr2 = [arr2; ones(abs(extra), c2) .* arr2(end, :)];
       elseif extra < 0
           arr1 = [arr1; ones(abs(extra), c1) .* arr1(end, :)];
       end
    end
end
