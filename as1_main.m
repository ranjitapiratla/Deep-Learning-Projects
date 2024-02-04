%% CPS 584 - Assignment 1
% Author - Connar Hite
clc; clear; close all;

%% Import images and format
% Set the path and general parameters
path = 'Assignment 1\Flowers\Flowers';
training_amount = 40;
testing_amount = 20;
categories = ["Rose","Tulip"];

% Set the images and labels arrays
training_images = cell(training_amount*length(categories), 1);
training_labels = zeros(training_amount*length(categories), 1);

testing_images = cell(testing_amount*length(categories), 1);
testing_labels = zeros(testing_amount*length(categories), 1);

% Set the training and test path
training_path = strcat(path, '\Training\');
testing_path = strcat(path, '\Testing\');

% Get the training images and format
for i = 1:length(categories)
    temp = strcat(training_path, categories(i));
    dCell = dir([strcat(temp, '\*.jpg')]);
    
    for d = 1:training_amount
        training_images{(training_amount * (i -1)) + d} = double(imresize(rgb2gray(imread([strcat(temp, '\', dCell(d).name)])), [50 50]));
        training_labels((training_amount * (i -1)) + d) = i-1;
    end
end

% Get the testing images and format
for i = 1:length(categories)
    temp = strcat(testing_path, categories(i));
    dCell = dir([strcat(temp, '\*.jpg')]);
    
    for d = 1:testing_amount
        testing_images{(testing_amount * (i -1)) + d} = double(imresize(rgb2gray(imread([strcat(temp, '\', dCell(d).name)])), [50 50]));
        testing_labels((testing_amount * (i -1)) + d) = i-1;
    end
end

%% Calculate the integral image
% Set integral image variable
training_int = zeros(training_amount*length(categories), 50, 50);
testing_int = zeros(testing_amount*length(categories), 50, 50);

% Calculate the integral image for all images
for i = 1:training_amount*length(categories)
    training_int(i, :, :) = integral_image(cell2mat(training_images(i, 1)));
end

for i = 1:testing_amount*length(categories)
    testing_int(i, :, :) = integral_image(cell2mat(testing_images(i, 1)));
end

%% Calculate the Rectangle sum w/ Haar-like features
% Set the bounds for the haar like features
haar_like = [3, 3, 45, 45, -1, -1, -1, -1; % rec feature 1
    4, 3, 18, 7, -1, -1, -1, -1; % rec feature 2
    15, 18, 19, 45, -1, -1, -1, -1; % rec feature 3
    4, 28, 7, 40, -1, -1, -1, -1; % ... 4
    4, 25, 40, 36, -1, -1, -1, -1; % ... 5
    8, 10, 36, 40, -1, -1, -1, -1; % ... 6
    17, 30, 22, 35, -1, -1, -1, -1; % ... 7
    28, 5, 40, 13, 28, 31, 40, 40; % ... 8
    16, 37, 40, 42, -1, -1, -1, -1; % ... 9
    22, 30, 31, 45, -1, -1, -1, -1; % ... 10
    10, 36, 18, 45, -1, -1, -1, -1; % ... 11
    13, 4, 18, 13, -1, -1, -1, -1; % ... 12
    18, 15, 26, 40, -1, -1, -1, -1; % ... 13
    4, 4, 35, 7, -1, -1, -1, -1; % ... 14
    4, 12, 28, 20, -1, -1, -1, -1; % ... 15
    7, 30, 10, 38, -1, -1, -1, -1; % ... 16
    5, 28, 7, 31, -1, -1, -1, -1; % ... 17
    36, 32, 38, 43, -1, -1, -1, -1; % ... 18
    27, 5, 30, 7, -1, -1, -1, -1; % ... 19
    27, 27, 30, 40, -1, -1, -1, -1; % ... 20
    40, 32, 42, 39, -1, -1, -1, -1; % ... 21
    5, 10, 11, 14, -1, -1, -1, -1; % ... 22
    6, 5, 14, 22, -1, -1, -1, -1; % ... 23
    26, 3, 28, 19, 26, 21, 28, 37; % ... 24
    29, 7, 37, 12, -1, -1, -1, -1; % ... 25
    22, 4, 25, 7, -1, -1, -1, -1; % ... 26
    8, 10, 42, 38, -1, -1, -1, -1; % ... 27
    20, 28, 24, 34, -1, -1, -1, -1; % ... 28
    25, 33, 39, 36, -1, -1, -1, -1; % ... 29
    18, 23, 24, 27, -1, -1, -1, -1; % ... 30
    ];

% Create the variables for the resulting rectangle sums
training_haar = zeros(training_amount*length(categories), length(haar_like));
testing_haar = zeros(testing_amount*length(categories), length(haar_like));

% Calculate the rectangle sums for the training set
for i = 1:training_amount*length(categories)
    for j = 1:length(haar_like)
        [y1, x1, y2, x2] = deal(haar_like(j, 1), haar_like(j, 2), haar_like(j, 3), haar_like(j, 4)); % Get the bounds
        sum_w = training_int(i, y2, x2); % Get the bottom right pixel for the white box
        % Subtract black portion if white box is not on top of bound
        if y1 ~= 1
            sum_w = sum_w - training_int(i, y1 - 1, x2);
        end
        % Subtract black portion if white box is not on left of bound
        if x1 ~= 1
            sum_w = sum_w - training_int(i, y2, x1 - 1);
        end
        % Add back in the portion that is subtracted twice
        if x1 ~= 1 && y1 ~= 1
            sum_w = sum_w + training_int(i, y1 - 1, x1 - 1);
        end
        % Repeat the process if there is a second white box
        [y1, x1, y2, x2] = deal(haar_like(j, 5), haar_like(j, 6), haar_like(j, 7), haar_like(j, 8));
        if y1 ~= -1
            sum_w = sum_w + training_int(i, y2, x2);
            if y1 ~= 1
                sum_w = sum_w - training_int(i, y1 - 1, x2);
            end
            if x1 ~= 1
                sum_w = sum_w - training_int(i, y2, x1 - 1);
            end
            if x1 ~= 1 && y1 ~= 1
                sum_w = sum_w + training_int(i, y1 - 1, x1 - 1);
            end
        end
        sum_b = training_int(i, 50, 50) - sum_w; % Get the total black space
        training_haar(i, j) = (sum_w - sum_b) / (50*50); % Calcuate the rectangle sum
    end
end

% Calculate the rectangle sums for the testing set
for i = 1:testing_amount*length(categories)
    for j = 1:length(haar_like)
        [y1, x1, y2, x2] = deal(haar_like(j, 1), haar_like(j, 2), haar_like(j, 3), haar_like(j, 4)); % Get the bounds
        sum_w = testing_int(i, y2, x2); % Get the bottom right pixel for the white box
        % Subtract black portion if white box is not on top of bound
        if y1 ~= 1
            sum_w = sum_w - testing_int(i, y1 - 1, x2);
        end
        % Subtract black portion if white box is not on left of bound
        if x1 ~= 1
            sum_w = sum_w - testing_int(i, y2, x1 - 1);
        end
        % Add back in the portion that is subtracted twice
        if x1 ~= 1 && y1 ~= 1
            sum_w = sum_w + testing_int(i, y1 - 1, x1 - 1);
        end
        % Repeat the process if there is a second white box
        [y1, x1, y2, x2] = deal(haar_like(j, 5), haar_like(j, 6), haar_like(j, 7), haar_like(j, 8));
        if y1 ~= -1
            sum_w = sum_w + testing_int(i, y2, x2);
            if y1 ~= 1
                sum_w = sum_w - testing_int(i, y1 - 1, x2);
            end
            if x1 ~= 1
                sum_w = sum_w - testing_int(i, y2, x1 - 1);
            end
            if x1 ~= 1 && y1 ~= 1
                sum_w = sum_w + testing_int(i, y1 - 1, x1 - 1);
            end
        end
        sum_b = testing_int(i, 50, 50) - sum_w; % Get the total black space
        testing_haar(i, j) = (sum_w - sum_b) / (50*50); % Calcuate the rectangle sum
    end
end

%% Train and test K-nearest neighbor (KNN)
% Set the knn distance variable
knn = zeros(testing_amount*length(categories), training_amount*length(categories));

% Calculate the euclidean distance for each feature
for i = 1:testing_amount*length(categories)
    for j = 1:training_amount*length(categories)
        knn(i, j) = euclidean(testing_haar(i, :), training_haar(j, :));
    end
end

K = [1 3 5 7]; % Set the number of nearest neighbors
knn_output = zeros(testing_amount*length(categories), length(K)); % Set the output variable
temp = 0; % Used for formating output matrix

% Find the K nearest neighbors
for i = K
    temp = temp + 1;
    % For each testing image, get the K nearest neighbors
    for j = 1:testing_amount*length(categories)
        [~, idx] = mink(knn(j, :), i);
        results = zeros(i, 1);
        % Get the label for the K nearest neighbors
        for result = 1:i
            results(result) = training_labels(idx(result));
        end
        knn_output(j, temp) = mode(results); % Output is the label that appears the most
    end
end

% Get the accuracy for each KNN
accuracy = zeros(1, length(K));
accuracy(1) = sum(abs(knn_output(:, 1) - testing_labels));
accuracy(2) = sum(abs(knn_output(:, 2) - testing_labels));
accuracy(3) = sum(abs(knn_output(:, 3) - testing_labels));
accuracy(4) = sum(abs(knn_output(:, 4) - testing_labels));

accuracy = 1 - (accuracy ./ length(testing_labels));

%% Train and test Neural Network (NN)
test_amount = 1; % How many times will NN repeat testing

% Set the total accuracy variables
total_acc_rose = zeros(test_amount, 1);
total_acc_tulip = zeros(test_amount, 1);
total_acc_nn = zeros(test_amount, 1);

% Repeat testing 'test_amount' of times to get multiple results
for tests = 1:test_amount
    disp('Performing training');
    net = feedforwardnet([16]); % Set NN format
    net = train(net, training_haar', training_labels'); % Train NN
    
    disp('Performing testing');
    predicted = net(testing_haar'); % Get predicted output
    
    % Set accuracy variables
    accuracy_nn = 0;
    acc_rose = 0;
    acc_tulip = 0;
    
    % Test if prediction was correct
    for i = 1:size(testing_haar, 1)
        % Threshold prediction
        if predicted(i) >= 0.5
            predicted(i) = 1;
        else
            predicted(i) = 0;
        end
    
        % Compare prediction to actual label
        if(predicted(i) == testing_labels(i))
            accuracy_nn = accuracy_nn + 1;
            if predicted(i) == 0
                acc_rose = acc_rose + 1;
            else
                acc_tulip = acc_tulip + 1;
            end
        end
    end
    
    % Format accuracy
    accuracy_nn = accuracy_nn / length(testing_labels);
    acc_rose = acc_rose / testing_amount;
    acc_tulip = acc_tulip / testing_amount;

    % Store accuracy for each run
    total_acc_nn(tests, 1) = accuracy_nn;
    total_acc_tulip(tests, 1) = acc_tulip;
    total_acc_rose(tests, 1) = acc_rose;

    % Display result for each run
    disp(['The accuracy of NN: ' num2str(accuracy_nn * 100) '%']);
    disp(['The accuracy of the rose class: ' num2str(acc_rose * 100) '%']);
    disp(['The accuracy of the tulip class: ' num2str(acc_tulip * 100) '%']);
    disp('====================================')
end

% Display average accuracy accross all runs
disp(['The total accuracy of NN: ' num2str(sum(total_acc_nn) * 100 / test_amount) '%']);
disp(['The total accuracy of the rose class: ' num2str(sum(total_acc_rose) * 100 / test_amount) '%']);
disp(['The total accuracy of the tulip class: ' num2str(sum(total_acc_tulip) * 100 / test_amount) '%']);

% Create a box and whisker plot for all runs 
figure;
boxplot([total_acc_nn total_acc_rose total_acc_tulip], {'Total','Rose','Tulip'});
title('Accuracy Results for [16] NN w/ 70 Images');
ylabel('Accuracy (%)');