function euclidianDist = euclidianDistance(X,Y)
% euclidianDistance: computes the Euclidian distance of given feature X
%   and matrix of features Y (Note: X and Y must have the same width)
%
% inputs:
%   X: 1xM matrix to find the euclidean distance to Y
%   Y: NxM matrix to find the euclidean distance from X
%
% output:
%   euclidianDist: double matrix containing Euclidian distances

    m = size(X,1);
    n = size(Y,1);
    mOnes = ones(1,m);
    euclidianDist = zeros(m,n);
    
    for i = 1:n
        yi = Y(i,:);
        yiRep = yi(mOnes, :);
        euclidianDist(:,i) = sqrt(sum((X - yiRep).^2, 2));
    end
end