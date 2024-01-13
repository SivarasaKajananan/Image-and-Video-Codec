clc;
clear all;

% Load an input image
InputImage = imread('wallpaper.jpg');
GrayImage = rgb2gray(InputImage); % Convert to grayscale if needed
DGrayImage = im2double(GrayImage);
[width,length] = size(DGrayImage);
imwrite(GrayImage,'gray_image.jpg')
 

qtable=generateQuantizationMatrix(780, [width,length]);

% Parameters
blockSize = 8; % Size of the DCT block

qLow = [ 32, 24, 24, 32, 48, 80, 102, 122; 24, 24, 28, 38, 52, 116, 120, 110;28, 26, 32, 48, 80, 114, 138, 112; 28, 34, 44, 58, 102, 174, 160, 124;36, 44, 74, 112, 136, 218, 206, 154;48, 70, 110, 128, 162, 208, 226, 184;98, 128, 156, 174, 206, 242, 240, 202; 144, 184, 190, 196, 224, 200, 206, 198;]; % Quantization matrix for low quality
qMed = [16, 11, 10, 16, 24, 40, 51, 61; 12, 12, 14, 19, 26, 58, 60, 55; 14, 13, 16, 24, 40, 57, 69, 56; 14, 17, 22, 29, 51, 87, 80, 62; 18, 22, 37, 56, 68, 109, 103, 77; 24, 35, 55, 64, 81, 104, 113, 92; 49, 64, 78, 87, 103, 121, 120, 101; 72, 92, 95, 98, 112, 100, 103, 99]; % Quantization matrix for Medium quality
qHigh = [8, 6, 5, 8, 12, 20, 26, 31; 6, 6, 7, 10, 13, 29, 30, 27; 7, 6, 8, 12, 20, 28, 35, 28; 7, 9, 11, 15, 26, 44, 40, 31; 9, 11, 19, 28, 34, 55, 52, 39; 12, 18, 28, 32, 41, 52, 57, 46; 24, 32, 39, 44, 52, 61, 60, 51; 36, 46, 48, 49, 56, 50, 51, 50]; % Quantization matrix for High quality

quant=qtable

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
 %Save the compressed image

file3 = fopen('compressed image data.txt','w');
[r,~]=size(huffmanEncoded);
for c=1:r
    fprintf(file3, '%d',huffmanEncoded(c));
end
fclose(file3);
 
 
% Entropy Decoding (Huffman decoding)
huffmanDecoded=huffmandeco(huffmanEncoded,dict);
huffmanDecoded = reshape(huffmanDecoded , [width ,length]);

% Reconstruction of Quantized Data
reconstructedQuantized = blockproc(huffmanDecoded, [blockSize blockSize], @(block_struct) block_struct.data .* quant);

% Inverse Transform (IDCT)
reconstructedQuantized = reconstructedQuantized/1000;
reconstructedImage = (blockproc(reconstructedQuantized, [blockSize blockSize], @(block_struct) idct2(block_struct.data)));
imwrite(reconstructedImage,'Reconstructed_image.jpg');
% Display the results
binaryDataO = de2bi(GrayImage);
binaryData = de2bi(huffmanEncoded);
orignalsize = numel(binaryDataO)/1024;
encodedSize = numel(binaryData)/1024;
compressionRatio = orignalsize / encodedSize; % Compression ratio
PSNR_DECODE_IMAGE = psnr(reconstructedImage,DGrayImage); %30dB-50dB is better less is not acceptable   

fprintf('Original image trans size: %d kbits \n', orignalsize);
fprintf('Encoded image size: %d kbits \n', ceil(encodedSize));
fprintf('Compression ratio: %.2f\n', compressionRatio);
fprintf('PSNR: %.2f dB\n', PSNR_DECODE_IMAGE);
% imshow(reconstructedImage)
% figure;
% 
% subplot(1, 2, 1); imshow(DGrayImage); title('Original Image');
% subplot(1, 2, 2); imshow(reconstructedImage); title(' Reconstructed image' );
% figure;
% imshow(DGrayImage)

function quantizationMatrix = generateQuantizationMatrix(x, imageSize)
    % Define the base quantization matrix
    %x=1/x;
    targetBitRate =((1*x^8)+(0*x^7)+(1*x^6)+(0*x^5)+(1*x^4)+(-45*x^3)+(16703*x^2)+(-3368296*x^1)+289655957)*(1.0e-17)   %targetBitRate=round(targetBitRate)
   
    baseMatrix = [16, 11, 10, 16, 24, 40, 51, 61;
                  12, 12, 14, 19, 26, 58, 60, 55;
                  14, 13, 16, 24, 40, 57, 69, 56;
                  14, 17, 22, 29, 51, 87, 80, 62;
                  18, 22, 37, 56, 68, 109, 103, 77;
                  24, 35, 55, 64, 81, 104, 113, 92;
                  49, 64, 78, 87, 103, 121, 120, 101;
                  72, 92, 95, 98, 112, 100, 103, 99];
    % Calculate the scale factor based on the target bit rate
    
    imageSize = prod(imageSize);
    scale = sqrt(targetBitRate / (imageSize/64))
    
    % Scale the base quantization matrix
    quantizationMatrix = round(baseMatrix * scale);
end
