%% Function to calculate the integral image for a single image
function int_image = integral_image(image)
    [row, col] = size(image); 
    int_image = zeros(row, col); 
    int_image(1, 1) = image(1, 1);

    for i = 2:row
        int_image(i, 1) = int_image(i-1, 1) + image(i, 1);
    end

    
    for i = 2:col
        int_image(1, i) = int_image(1, i-1) + image(1, i);
    end
    
    for i = 2:row
        for j = 2:col
            int_image(i, j) = image(i, j) - int_image(i-1, j-1) + int_image(i, j-1) + int_image(i-1, j);
        end
    end
end