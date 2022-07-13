%% Read all value functions
clear all;
close all;
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
n_goal = 1;
n_init = length(st_files)/n_goal;
n_states = 2;
id_value = n_states + 1;

for goal = 1:n_goal
    temp_sv = [];
    sv_cols = [];
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
        sv = sortrows(sv, id_value);
        [sv, sv_cols] = cp_to_match(sv, sv_cols);
        sv_cols = [sv_cols, sv];
    end
    
    temp_sv(:, id_value:id_value:end) = temp_sv(:, id_value:id_value:end) ./ ...
                                        max(temp_sv(:, id_value:id_value:end));    
    [r, c] = size(temp_sv);
    temp_sv = sortrows(temp_sv, c);
    
    % CSV per goal
    fname = sprintf("di_%d_data_size_%d_data_num_%d.csv", goal, length(temp_sv), goal);
    dlmwrite(path + fname, temp_sv, 'delimiter', ',', 'precision', 10);
    
    % Match dimensions
    [temp_sv, st_val] = cp_to_match(temp_sv, st_val);
    
    % Add new dataset
    st_val = [st_val, temp_sv];
end 

st_val = remove_mid_elems(st_val, 3);

% Normalise values per goal 
st_val(:, id_value:id_value:end) = st_val(:, id_value:id_value:end) ./ ...
                                   max(st_val(:, id_value:id_value:end));
figure(4);                               
plot3(sv_cols(:,1:3:end), sv_cols(:,2:3:end), sv_cols(:,3:3:end)); 

%% Reconstruct Data for GANs
data_segments = length(states) / length(st_files);
fname = sprintf("di_data_size_%d_data_num_%d.csv", length(st_val), n_goal);
dlmwrite(path + fname, st_val, 'delimiter', ',', 'precision', 10);


%% New clean



%% Helper functions
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


function files = match_file_to_name(name, dir)
    f_name = sprintf('*%f*', name);
    f = dir(fullfile(dir, f_name));
    files = imreads(fullfile(dir, f,name));
end