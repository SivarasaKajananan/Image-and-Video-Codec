function IDCT_frame_uint8 = IDCT(Gray_frame,N)

[rows , cols ] = size(Gray_frame);
IDCT_frame = zeros(rows , cols);

% Divide the dequantized frame into marco blocks (8*8) and do the IDCT transformation
for i= 1 : N :rows
    for j = 1:N:cols
        IDCT_frame(i:i+N-1 , j:j+N-1) = idct2(Gray_frame(i:i+N-1 , j :j+N-1));
    end
end
IDCT_frame_uint8 = uint8(IDCT_frame);
end

