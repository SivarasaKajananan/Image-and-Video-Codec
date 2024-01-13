function Quantized_frame = Quantizer(N ,DCT_frame,Quantizing_table)

[rows,cols] = size(DCT_frame);
Quantized_frame = zeros(rows,cols);
ans = zeros(N,N ) ;

% Divide the DCT frame into Macro blocks (8*8) and divide each MB by the default Quantizing table
for k = 1 : N : rows
    for p = 1 :N: cols
        Macro_blk  = DCT_frame(k:k+N-1 , p:p+N-1);
        for i = 1 : N
            for j = 1 :N
                 ans(i,j) = Macro_blk(i,j)/Quantizing_table(i,j);  
            end
        end
        Quantized_frame(k:k+N-1 ,p:p+N-1) = round(ans) ;
    end
end
disp(Macro_blk)
end