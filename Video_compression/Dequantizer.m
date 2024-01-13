function Dequantized = Dequantizer(N , decoded, Quantizing_table)

   [rows,cols] = size(decoded);
   Dequantized = zeros(rows,cols);
   ans = zeros(N,N ) ;

 % Divide the huffman decoded frame into Macro blocks (8*8) and divide each MB by the default Quantizing table
for k = 1 : N : rows
    for p = 1 :N: cols
        Macro_blk  = decoded(k:k+N-1 , p:p+N-1);
        for i = 1 : N
            for j = 1 :N
                 ans(i,j) = Macro_blk(i,j) * Quantizing_table(i,j);    
            end
        end
        Dequantized(k:k+N-1 ,p:p+N-1) = ans ;
    end
end

end