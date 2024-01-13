function [dict , encoded_bitseq ] = Huffman_encoder(Quantized_frame, N)

[rows,cols] = size(Quantized_frame);
encoded_bitseq = cell(rows/N , cols/N);
dict = cell(rows/N , cols/N);

% Divide the quantized frame into macro blocks(8*8) and do huffman encoding to each block
for i = 1 : N : rows
    for j = 1 : N : cols
       
        Macro_blk = Quantized_frame(i:i+N-1 , j:j+N-1);
        
        if nnz(Macro_blk) == 0
            Macro_blk(1,1) = 1;
            [g , ~ , Intensity_val] = grp2idx(Macro_blk(:));
            Frequency = accumarray(g,1);
            probability = Frequency./(N*N) ;
            [dict1,avglen] = huffmandict(Intensity_val,probability);
            encoded_bitseq{floor(i/N)+1,floor(j/N)+1} = huffmanenco(Macro_blk(:) , dict1);
            dict{floor(i/N)+1,floor(j/N)+1} =dict1;
        else
            [g , ~ , Intensity_val] = grp2idx(Macro_blk(:));
            Frequency = accumarray(g,1);
            probability = Frequency./(N*N) ;
            [dict1,avglen] = huffmandict(Intensity_val,probability);
            encoded_bitseq{floor(i/N)+1,floor(j/N)+1} = huffmanenco(Macro_blk(:) , dict1);
            dict{floor(i/N)+1,floor(j/N)+1} =dict1;   
        end
    end   
end
end