function filteredImage = deblockingFilter(inputImage, blockSize, strength)
    [height, width] = size(inputImage);
    filteredImage = inputImage;

    for i = 1:blockSize:height-blockSize+1
        for j = 1:blockSize:width-blockSize+1
            block = inputImage(i:i+blockSize-1, j:j+blockSize-1);
            
            % Calculate block boundary strength based on the differences between neighboring pixels
            horizontalStrength = mean(abs(diff(block, 1, 2)), 'all');
            verticalStrength = mean(abs(diff(block, 1, 1)), 'all');
            
            % Apply deblocking filter based on block boundary strength
            if horizontalStrength > strength
                filteredImage(i:i+blockSize-1, j+blockSize-1) = mean(block(:, blockSize));
            end
            
            if verticalStrength > strength
                filteredImage(i+blockSize-1, j:j+blockSize-1) = mean(block(blockSize, :));
            end
        end
    end
end
