%% This script separates data files based on goals with the assumption that
% they are written in the following format ...start_#_#_goal...
clear all;
close all; 
clc; 

goal_sfix = ".000000_goal";
start_pfix = "start_";
sep = "_";

path = "~/Repos/OptimisationBasedControl/data/";
all_files = dir(path + "*doubleintegrator_ctrl*.csv");
[~, idx] = find([all_files.bytes] > 0);
all_files = all_files(idx);


%% Extract goals and init
file_names = string(length(all_files));
start_end = string(length(all_files));
goals = string(length(all_files));
start = string(length(all_files));
start_goal = [];
start_goal_s = string(ones(length(all_files), 2));

for i  = 1:length(all_files)
        file_names(i) = all_files(i).name;
        start_end = extractBefore(file_names(i), goal_sfix);
        start_end = extractAfter(start_end, start_pfix);
        goals(i) = extractAfter(start_end, sep);
        start(i) = extractBefore(start_end, sep);
        start_goal(i, :) = [str2double(start(i)), str2double(goals(i))];
        start_goal_s(i, :) = [start(i), goals(i)];
end

ctrl_data = [];
for i = 1:length(start_goal_s)
    key = start_goal_s(i, 1) + "_" + start_goal_s(i, 2);
    fname = match_file_to_name(key, path);
    new_ctrl = csvread(path + fname);
    [ctrl_data, new_ctrl] = cp_to_match(ctrl_data, new_ctrl);
    ctrl_data = [ctrl_data, new_ctrl];
end

%% Utility functions
function name = match_file_to_name(key, path)
    f_wild_x = sprintf('*%s*', key);
    f = dir(fullfile(path, f_wild_x));
    name = f.name;
end


function [arr1, arr2] = cp_to_match(arr1, arr2)
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