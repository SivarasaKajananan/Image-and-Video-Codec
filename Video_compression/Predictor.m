function [predicted_frames,motion_vectors] = Predictor(Reconstructed_frame1,num_frames , N)

motion_vectors = cell(1,num_frames);
frame_names = cell(1 , num_frames);
frames = cell(1,num_frames);
predicted_frames = cell(1 , num_frames);

for i = 1 : num_frames
    frame_names{1,i} = strcat('frame',num2str(i),'.jpg') ;
end
for j = 1 : num_frames
    frames{1,j} = imread(frame_names{1,j}); 
end

predicted_frames{1,1} = Reconstructed_frame1 ;
for k = 2: num_frames 

        if k == 2 
            Reference_frame = Reconstructed_frame1 ;
        else 
            Reference_frame = predicted_frames{1,k-1};
        end
        
        motion_vectors{1,k} = SAD( Reference_frame , frames{1,k} ,N);
        predicted_frames{1,k} =  Frame_predictor(Reference_frame , motion_vectors{1,k} , N); 
end
end