function Reconstructed_frame = frame_decoder(height , width ,N , encoded_frame, dict_frame , Quantizing_table)

%Huffman decoding
Decoded_frame = Huffman_decoder(height , width ,N , encoded_frame, dict_frame);

%Dequnatizing 
Dequantized_frame = Dequantizer(N , Decoded_frame,Quantizing_table) ;

% IDCT
Reconstructed_frame = IDCT(Dequantized_frame , N);
end