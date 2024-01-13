function decoded = Huffman_decoder(rows ,cols ,N , encoded, dict)
decoded = zeros(rows , cols);

for i = 1 : rows/N
    for j = 1 : cols/N
        input_dict = dict{i,j};
        input_code = encoded{i,j} ;
        decoded((i-1)*8 + 1 : (i-1)*8+8 , (j-1)*8 + 1 : (j-1)*8+8) =  reshape(huffmandeco(input_code , input_dict),[N,N]);
    
    end
end
end

