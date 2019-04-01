% Help for extracting the matched feature points
% (since code provided by discussion slides couldn't work on my Mac):
% https://www.mathworks.com/help/vision/ref/matchfeatures.html

clc; 
clear; 
close all;

% next 13 lines extract the matched feature points
img1 = imread('M01.jpg'); %tests first 2 images of UCSB
img2 = imread('M02.jpg');
I1 = rgb2gray(img1);
I2 = rgb2gray(img2);
% points1 = detectHarrisFeatures(I1);
% points2 = detectHarrisFeatures(I2);
points1 = detectSURFFeatures(I1);
points2 = detectSURFFeatures(I2);
[features1,valid_points1] = extractFeatures(I1,points1);
[features2,valid_points2] = extractFeatures(I2,points2);
indexPairs = matchFeatures(features1,features2);
matchedPoints1 = valid_points1(indexPairs(:,1),:); % matched points from 1st img
matchedPoints2 = valid_points2(indexPairs(:,2),:); % matched points from 2nd img
figure; 
ax = axes;
showMatchedFeatures(img1,img2,matchedPoints1,matchedPoints2,'montage','Parent',ax);


%%% RANSAC %%%
maxInliers = 0;
finalH = [0,0,0;0,0,0;0,0,0]; % for the best homography
for i = 1:100
    inliers = 0;
    
    fourPairs = zeros(4,4);
    select1 = randi([1, length(matchedPoints1)], 1, 1);
    select2 = randi([1, length(matchedPoints1)], 1, 1);
    select3 = randi([1, length(matchedPoints1)], 1, 1);
    select4 = randi([1, length(matchedPoints1)], 1, 1);
    fourPairs(1,1) = matchedPoints1.Location(select1,1); % 1st pair's 1st img's x-coord
    fourPairs(1,2) = matchedPoints1.Location(select1,2); % 1st pair's 1st img's y-coord
    fourPairs(1,3) = matchedPoints2.Location(select1,1); % 1st pair's 2nd img's x-coord
    fourPairs(1,4) = matchedPoints2.Location(select1,2); % 1st pair's 2nd img's y-coord
    fourPairs(2,1) = matchedPoints1.Location(select2,1); % 2nd pair's 1st img's x-coord
    fourPairs(2,2) = matchedPoints1.Location(select2,2); % 2nd pair's 1st img's y-coord
    fourPairs(2,3) = matchedPoints2.Location(select2,1); % 2nd pair's 2nd img's x-coord
    fourPairs(2,4) = matchedPoints2.Location(select2,2); % 2nd pair's 2nd img's y-coord
    fourPairs(3,1) = matchedPoints1.Location(select3,1); % 3rd pair
    fourPairs(3,2) = matchedPoints1.Location(select3,2);
    fourPairs(3,3) = matchedPoints2.Location(select3,1);
    fourPairs(3,4) = matchedPoints2.Location(select3,2);
    fourPairs(4,1) = matchedPoints1.Location(select4,1); % 4th pair
    fourPairs(4,2) = matchedPoints1.Location(select4,2);
    fourPairs(4,3) = matchedPoints2.Location(select4,1);
    fourPairs(4,4) = matchedPoints2.Location(select4,2);

    %%% Direct Linear Transformation algorithm %%%
    n = 4;
    matrix_A = zeros(2*n,9); % initializing the 2n x 9 matrix A
    for j = 1:n % going thru the 4 points
        % matrix_Ai = zeros(2,9); % 1 per point correspondence
        x1 = fourPairs(j,1);
        y1 = fourPairs(j,2);
        x2 = fourPairs(j,3);
        y2 = fourPairs(j,4);
        matrix_A((2*j)-1,:) = [-x1,-y1,-1,0,0,0,x2*x1,x2*y1,x2];
        matrix_A(2*j,:) = [0,0,0,-x1,-y1,-1,y2*x1,y2*y1,y2];
    end
    [U,D,V] = svd(matrix_A,0);
    h = V(:,end);
    preH = [h(1), h(2), h(3); h(4), h(5), h(6); h(7), h(8), h(9)];
    H = preH / h(9);
    %%% END DLT %%%
    
    for k = 1:length(matchedPoints1)
        
        % finding the distance between the calculated 2nd point (determined
        % by pi * H) and actual location of the 2nd point (p'i)
        firstPoint = zeros(3,1);
        secondPoint = zeros(3,1);
        firstPoint(1:2,1) = transpose(matchedPoints1.Location(k,:));
        secondPoint(1:2,1) = transpose(matchedPoints2.Location(k,:));
        firstPoint(3,1) = 1;
        secondPoint(3,1) = 1;
        
        %dimensions/d would be ri from the discussion slides (slide 19)
        dimensions = secondPoint - (H * firstPoint); 
        dx = dimensions(1,1);
        dy = dimensions(2,1);
        d = sqrt((dx*dx)+(dy*dy));
        % if the distance between estimated location and actual location
	% of 2nd point is larger than it's supposed to be, the actual
        % 2nd point is an outlier
        if (d < 2) % if ||ri|| < epsilon
           inliers = inliers + 1; 
        end
    end
    if (maxInliers < inliers)
        maxInliers = inliers;
        finalH = H;
    end
end
%%% END RANSAC %%%

%Start of attempt of warping process
% transform = projective2d(finalH.');
% warpedI = imwarp(img1,transform);

%mapping the four corners of each image source
firstSize = size(I1);
secondSize = size(I2);
fourCorners1=[0,0;firstSize(1,1),0;0,firstSize(1,2);firstSize(1,1),firstSize(1,2)];
fourCorners2=[0,0;secondSize(1,1),0;0,secondSize(1,2);secondSize(1,1),secondSize(1,2)];





