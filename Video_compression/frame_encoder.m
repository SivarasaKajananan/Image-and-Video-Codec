function [dict , encoded_frame] = frame_encoder(Gray_frame ,N , Quantizing_table)

% DCT transform the Gray frame
DCT_frame = DCT(Gray_frame , N);

% Quantizing the Gray frame
quantized_frame = Quantizer(N , DCT_frame, Quantizing_table);

% Huffman encoding
[dict , encoded_frame] = Huffman_encoder(quantized_frame, N);
end