function DCT_frame = DCT(Gray_frame , N)

[rows , cols ] = size(Gray_frame);
DCT_frame = zeros(rows , cols);

% Divide the Gray frame into marco blocks (8*8) and do the DCT transformation
for i= 1 : N :rows
    for j = 1:N:cols
        DCT_frame(i:i+N-1 , j:j+N-1) = dct2(Gray_frame(i:i+N-1 , j :j+N-1));
    end
end

end

