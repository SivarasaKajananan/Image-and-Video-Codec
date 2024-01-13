function [motion_vectors,SAD_vals] = SAD(frame1 , frame2 ,N)

[rows,cols] = size(frame1);
motion_vectors = cell(rows/N , cols/N);
SAD_vals = cell(rows/N , cols/N);
initial_SAD = 26320;
motion_vect = zeros(1,2);

%Get the MinimumAD values after comparing each block in current frame with the every block in the reference frame
for i = 1 : N : rows
    for j = 1 : N : cols
        frame2_MB = frame2(i: i +N-1 , j : j + N-1);
        for k = 1 : N : rows
            for m = 1 : N : cols
                frame1_MB = frame1(k : k + N-1 , m : m + N-1);
                difference = abs(double(frame2_MB) - double(frame1_MB)) ;
                sum_abs_diff = sum(difference(:));
                
                if (sum_abs_diff < initial_SAD)
                initial_SAD = sum_abs_diff ; 
                motion_vect(1,1) = k ;
                motion_vect(1,2) = m ;
                end 
            end
        end
        SAD_vals{ceil(i/N) , ceil(j/N)} = initial_SAD ; 
        motion_vectors{ceil(i/N) , ceil(j/N)} = motion_vect;

         % Reset initial_SAD and motion_vect for the next block
        initial_SAD = 26320;
        motion_vect = zeros(1, 2);
    end
end
end