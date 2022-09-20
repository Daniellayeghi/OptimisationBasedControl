function state_d= point_mass(t, state, force)
    state_d = zeros(6, 1);
    fri_coeff = 0;
    state_d(4:6, 1) = eye(3) * 1 * (force' - eye(3) * fri_coeff* state(4:6));
    state_d(1:3, 1) = state(4:6) + state_d(4:6, 1) * 0.0005;
end
