
v = VideoReader('sunset.mp4');

%Break video to frames
num_frames = 20;
N = 8 ; % Macro block size
Quantizing_table = [17,18,24,47,99,99,99,99 ; 18,21,26,66,99,99,99,99 ; 24,26,56,99,99,99,99,99; 47,66,99,99,99,99,99,99; 99,99,99,99,99,99,99,99;99,99,99,99,99,99,99,99;99,99,99,99,99,99,99,99;99,99,99,99,99,99,99,99];


for i = 1:20;
    filename=strcat('frame',num2str(i),'.jpg');
    frame = read(v, i);
    Gray_frame = rgb2gray(frame);
    final_frame =imresize(Gray_frame,[144 256]);
    imwrite(final_frame,filename);
end

frames  = cell(1,20);
frame_names = cell(1,20);

for i = 1 : num_frames
    frame_names{1,i} = strcat('frame',num2str(i),'.jpg');
end
for j = 1 : num_frames
    frames{1,j} = imread(frame_names{1,j}); 
end

outputVideo1 = VideoWriter('Initial_video');
outputVideo1.FrameRate = v.FrameRate;
open(outputVideo1);

for ii = 1:num_frames
   img = uint8(frames{1,ii}) ;
  writeVideo(outputVideo1,img);
end
close(outputVideo1)
% dimensions of a frame
[height, width] = size(frames{1,1});
 
% Encoding and Decoding the 1st frame and Calculate all motion vectors and Predict all frames
[dict1 , encoded1] = frame_encoder(frames{1,1}, N , Quantizing_table);
reconstructed_frame1 = frame_decoder(height , width , N , encoded1, dict1 , Quantizing_table);

[predicted_frames , motion_vectors] = Predictor(reconstructed_frame1,num_frames , N);

figure , imshow(frames{1,2}) , title('Original frame 2');
figure , imshow(reconstructed_frame1) , title('Reconstructed frame 1');

 

%Find the difference between Original Frame and Motion predicted Frame
p_residual = cell(1,num_frames);

for i = 1 : num_frames
    p_residual{1,i} = double(frames{1,i}) - double(predicted_frames{1,i});
end

% Encoding Residual of all P frames
dict_p_residuals = cell(1,20);
encoded_p_residual = cell(1,20);

for i = 2 : num_frames
    [dict_res , enc_res] = frame_encoder(p_residual{1,i} ,N , Quantizing_table);
    dict_p_residuals{1,i} = dict_res ; 
    encoded_p_residual{1,i} = enc_res ;
end

dict_p_residuals{1,1} = dict1;
encoded_p_residual{1,1} = encoded1;

% write text file
fid1 = fopen('Residual_frames_including_frame1.txt','w');

for k = 1 : num_frames
    pres = encoded_p_residual{1,k};
     for i = 1 : height/N
        for j = 1 :width/N
        
            fprintf(fid1, num2str(pres{i,j}));
            fprintf(fid1,',');
        end
    end
    fprintf(fid1,'*');
end
fclose(fid1);

 figure , imshow(uint8(predicted_frames{1,2})) , title('predicted frame 2');
 figure , imshow(uint8(p_residual{1,2})) , title('Residual frame 2');

save('dict_p_residuals.mat', 'dict_p_residuals');
save('Motion_vectors.mat' , 'motion_vectors');

