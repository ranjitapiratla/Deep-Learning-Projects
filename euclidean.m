function [dist] = euclidean(test, train)
    dist = 0;
    for i = 1:length(test)
        dist = dist + (train(i) - test(i))^2;
    end
    dist = sqrt(dist);
end