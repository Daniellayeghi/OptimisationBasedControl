%% Read all value functions
path = "~/Repos/OptimisationBasedControl/data/";
%%
st_files = dir(path + "*doubleintegrator_state*.csv");
vl_files = dir(path + "*doubleintegrator_value*.csv");

states = [];
values = [];

for i = 1:length(st_files)
   
    states = [states; {csvread(path + st_files(i).name)}];
    values = [values; {csvread(path + vl_files(i).name)}];
end 

state_size = length(states{1}(end, :));
value_size = length(values{1}(end, :));

rows = [];
for cell = 1:length(states)
    [row, ~] = size(states{cell});
    rows = [rows; row];
end


max_rows = max(rows);
row_diff = abs(rows - max_rows);

for cell = 1:length(states)
    s_vec = states{1}(end, :);
    s_vec = ones(row_diff(cell), state_size) .* s_vec;
    states{cell} = [states{cell}; s_vec];
    
    v_vec = values{1}(end, :);
    v_vec = ones(row_diff(cell), value_size) .* v_vec;
    values{cell} = [values{cell}; v_vec];
end

num_data = 9;
data_size = length(values{1})*num_data;

states = cell2mat(states);
values = cell2mat(values);
st_val = [states, values];
plot3(states(:, 1), states(:, 2), values);
fname = sprintf("di_data_size_%d_data_num_%d.csv", data_size, num_data);
csvwrite(path + fname, st_val);