%% Read all value functions
clear all;
path = "~/Repos/OptimisationBasedControl/data/";
%% Clean up
% This script assumes file numbering is in order with goal and start 
% posisiton

st_files = dir(path + "*doubleintegrator_state*.csv");
vi_files = dir(path + "*doubleintegrator_value*.csv");

st_files = sort_by_date(st_files);
vi_files = sort_by_date(vi_files);

states = [];
values = [];
st_val = [];
n_goal = 19;
n_init = length(st_files)/n_goal;
n_states = 2;
id_value = n_states + 1;

for goal = 1:n_goal
    temp_sv = [];
    i = n_init * (goal-1) + 1;
    j = n_init * (goal-1) + n_init;
    idx = i:j;
    for init = idx
        s = csvread(path + st_files(init).name); s(1,:)= [];
        v = csvread(path + vi_files(init).name); v(1,:)= [];
        sv = [s, v]; 
        states = [states; {s}]; 
        values = [values; {v}];
        temp_sv = [temp_sv; sv];
    end
    [r, c] = size(temp_sv);
    temp_sv = sortrows(temp_sv, c);
    
    % Match dimensions
    [temp_sv, st_val] = cp_to_match(temp_sv, st_val);
    
    % Add new dataset
    st_val = [st_val, temp_sv];
end 

st_val = remove_mid_elems(st_val, 5);

% Normalise values 
st_val(:, id_value:id_value:end) = st_val(:, id_value:id_value:end) ./ ...
                                   max(st_val(:, id_value:id_value:end));


%% Reconstruct Data for GANs
data_segments = length(states) / length(st_files);
fname = sprintf("di_data_size_%d_data_num_%d.csv", length(st_val), n_goal);
csvwrite(path + fname, st_val);


function s = sort_by_date(s)
    s(~[s.isdir]);
    [~,idx] = sort([s.datenum]);
    s = s(idx);
end


function [arr1, arr2] = cp_to_match(arr1, arr2)
    [r1, c1] = size(arr1);
    [r2, c2] = size(arr2);

    if r1 ~= r2 && ~isempty(arr2)
       extra = r1 - r2;
       if extra > 0
           arr2 = [ones(abs(extra), c2) .* arr2(1, :); arr2];
       elseif extra < 0
           arr1 = [ones(abs(extra), c1) .* arr1(1, :); arr1];
       end
    end
end


function arr = remove_mid_elems(arr, times)

    for sparse = 1:times
        arr(1:2:end-1, :) = []; 
    end
end