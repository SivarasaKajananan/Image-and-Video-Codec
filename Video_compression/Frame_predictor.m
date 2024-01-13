function predicted_frame = Frame_predictor(previous_frame , motion_vectors , N)

[rows,cols] = size(previous_frame);
predicted_frame = zeros(rows,cols);

% Predict frames using reference frame and motion vectors (By adding motion vector with reference frame)
for i = 1 : N : rows
    for j = 1 : N : cols
        motion_vect = motion_vectors{ceil(i/N) , ceil(j/N)};
        vect_row = motion_vect(1,1);
        vect_col = motion_vect(1,2);
        predicted_frame(i:i+N-1 , j : j + N - 1) = previous_frame(vect_row:vect_row+N-1 , vect_col:vect_col + N - 1);
    
    end
end
end