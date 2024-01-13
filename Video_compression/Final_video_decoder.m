load('dict_p_residuals.mat');
load('motion_vectors.mat');
[rows, cols] = size(dict_p_residuals{1,2});    % r = rows/N
N = 8 ;
Quantizing_table = [17,18,24,47,99,99,99,99 ; 18 , 21 ,26, 66,99 ,99,99,99 ; 24 , 26 , 56 , 99 ,99,99,99,99; 47,66,99,99,99,99,99,99; 99,99,99,99,99,99,99,99;99,99,99,99,99,99,99,99;99,99,99,99,99,99,99,99;99,99,99,99,99,99,99,99];

%read data from text file
fid2 = fopen('residual_frames_including_frame1.txt');

% Read all lines & collect in cell array
txt = textscan(fid2,'%s','delimiter','*'); 
x = txt{1,1};

%%retrive residual_p_frame data
Final_retrived_code = cell(1,20);

for k = 1 : 20
encoded_code = retrive(rows,cols,x{k,1});
Final_retrived_code{1,k} = encoded_code ;
end

%Reconstructing P frames and first frame
Decoded_p_residual = cell(1,20);

for i = 1 : 20
    Decoded_p_residual{1,i} = frame_decoder(rows*N , cols*N ,N , Final_retrived_code{1,i}, dict_p_residuals{1,i} , Quantizing_table) ;
    
end

% get predicted frames from motion vectors
predicted_frames = cell(1,20);
predicted_frames{1,1} = Decoded_p_residual{1,1};

for i = 2 : 20
predicted_frames{1,i} = Frame_predictor(predicted_frames{1,i-1} , motion_vectors{1,i} , N);

end

Final_frames = cell(1,20);
Final_frames{1,1} = Decoded_p_residual{1,1} ;

for i = 2 : 20
    Final_frames{1,i} = double(Decoded_p_residual{1,i}) + double(predicted_frames{1,i});
    
end

figure , imshow(uint8(Final_frames{1,2})) , title('Reconstructed final frame 2');

%Save compressed video
Compressed_Video = VideoWriter('Compressed_video');
Compressed_Video.FrameRate = v.FrameRate;
open(Compressed_Video);

for ii = 1:20
   frame = uint8(Final_frames{1,ii});
   writeVideo(Compressed_Video,frame);
end

close(Compressed_Video)
