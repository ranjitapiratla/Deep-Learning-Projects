function haarLikeFeat = haarLikeFeature(imgGray, rectMatrix)
% haarLikeFeature: computes Haar-like features of an image 
%   using the given rectangle matrix. Can only handle Haar-like features 
%   with no more than two rectangles
%
% inputs:
%   imgGray: grayscaled image matrix
%   rectMatrix: Hx8 matrix (where H is number of Haar-like features)
%       fmt: yt1, xl1, yb1, xr1, yt2, xl2, yb2, xr2
%       for a Haar-like feature with only one white rectangle, set the
%       values of 5th-8th index to -1
%
% output:
%   haarLikeFeat: 1xH double matrix (where H is number of Haar-like features
%       provided in rectMatrix input)
%
% dependencies:
%   integralImg(): custom integral image function

    % compute integral image and convert image matrix to double
    intImage = integralImg(double(imgGray));
    sumAll = double(intImage(end,end));

    % get image size and initiate the returned feature
    [height, width] = size(imgGray);
    haarLikeFeat = zeros(1, size(rectMatrix, 1));

    % loop through each Haar-like feature rectangle(s)
    for r = 1:length(rectMatrix)
        [y1, x1, y2, x2] = deal(rectMatrix(r, 1), rectMatrix(r, 2), rectMatrix(r, 3), rectMatrix(r, 4)); % Get the bounds
        
        sumW = intImage(y2, x2); % Get the bottom right pixel for the white box
        % Subtract black portion if white box is not on top of bound
        if y1 ~= 1
            sumW = sumW - intImage(y1 - 1, x2);
        end
        % Subtract black portion if white box is not on left of bound
        if x1 ~= 1
            sumW = sumW - intImage(y2, x1 - 1);
        end
        % Add back in the portion that is subtracted twice
        if x1 ~= 1 && y1 ~= 1
            sumW = sumW + intImage(y1 - 1, x1 - 1);
        end
        
        % Repeat the process if there is a second white box
        [y1, x1, y2, x2] = deal(rectMatrix(r, 5), rectMatrix(r, 6), rectMatrix(r, 7), rectMatrix(r, 8));
        if y1 ~= -1
            sumW = sumW + intImage(y2, x2);
            if y1 ~= 1
                sumW = sumW - intImage(y1 - 1, x2);
            end
            if x1 ~= 1
                sumW = sumW - intImage(y2, x1 - 1);
            end
            if x1 ~= 1 && y1 ~= 1
                sumW = sumW + intImage(y1 - 1, x1 - 1);
            end
        end

        sumB = sumAll - sumW; % Get the total black space
        % set current Haar-like feature to the difference of the white
        % rectangle(s) and black area, then divide by total number of
        % pixels
        haarLikeFeat(r) = (sumW - sumB)/(height*width);
    end
end