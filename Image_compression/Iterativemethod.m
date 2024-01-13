InputImage = imread('wallpaper.jpg');
GrayImage = rgb2gray(InputImage);
quantization_matrix = adjust_quantization_matrix(GrayImage, 700 )

function quantization_matrix = adjust_quantization_matrix(gray_img, desired_rate )
    % Initialize variables    
    initial_matrix=[16, 11, 10, 16, 24, 40, 51, 61;12, 12, 14, 19, 26, 58, 60, 55;14, 13, 16, 24, 40, 57, 69, 56;14, 17, 22, 29, 51, 87, 80, 62; 18, 22, 37, 56, 68, 109, 103, 77;24, 35, 55, 64, 81, 104, 113, 92; 49, 64, 78, 87, 103, 121, 120, 101;
                  72, 92, 95, 98, 112, 100, 103, 99];
     high_matrix =initial_matrix*16;
     low_matrix =initial_matrix/4;
     binaryDataO = de2bi(gray_img);
    desired_ratio=(numel(binaryDataO)/1024)/desired_rate;
    current_ratio = get_compression_ratio(gray_img, initial_matrix);
    quantization_matrix = initial_matrix;
    max_ratio = get_compression_ratio(gray_img, high_matrix);
    min_ratio = get_compression_ratio(gray_img, low_matrix);

    % Check if desired ratio is within bounds
    if desired_ratio >= max_ratio
        quantization_matrix = high_matrix;
    elseif desired_ratio <= min_ratio
        quantization_matrix = low_matrix;
    else
        % Iterate to find the closest compression ratio
        while abs(current_ratio - desired_ratio) > 0.01
            % Increase or decrease quantization matrix based on the current ratio
            if current_ratio < desired_ratio
                quantization_matrix = quantization_matrix * 1.1; % Increase matrix by scalar
            else
                quantization_matrix = quantization_matrix * 0.9; % Decrease matrix by scalar
            end
            
            % Calculate the new compression ratio
            current_ratio = get_compression_ratio(gray_img, quantization_matrix);
        end
    end
quantization_matrix=round(quantization_matrix)
end

function compression_ratio = get_compression_ratio(gray_img, quantization_matrix)
 % Convert to grayscale if needed
DGrayImage = im2double(gray_img);
[width,length] = size(DGrayImage);

% Parameters
blockSize = 8; % Size of the DCT block
quant=quantization_matrix;

% Forward Transform (DCT)
dctImage = blockproc(DGrayImage, [blockSize blockSize], @(block_struct) dct2(block_struct.data));
dctImage = ceil(dctImage * 1000);

% Quantization
quantizedImage = blockproc(dctImage, [blockSize blockSize], @(block_struct) round(block_struct.data ./ quant));

% Entropy Encoding (Huffman coding)
[g,~,intensity_val] = grp2idx(quantizedImage(:));
 Frequency = accumarray(g,1);
 probability = Frequency./(width * length);
 T = table(intensity_val,Frequency,probability);%table(element | count| prob
 dict=huffmandict(intensity_val,probability);
 huffmanEncoded = huffmanenco(quantizedImage(:),dict);
binaryData = de2bi(huffmanEncoded);
encodedSize = numel(binaryData)/1024;
binaryDataO = de2bi(gray_img);
orignalsize = numel(binaryDataO)/1024;
compression_ratio = orignalsize / encodedSize;
 end
