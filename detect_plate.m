clc;
clear all;
close all;
imtool close all;
workspace;

% Taking the Image as Input
imagePath = 'CAR1.jpg'; % Use this or pass a different path
I = imread(imagePath);

% Display the original image
figure(1);
imshow(I);

% Call the main license plate detection function
detect_license_plate(imagePath);

% Extracting the Y component from image (Converting to Grayscale)
Igray = rgb2gray(I);
[rows, cols] = size(Igray);

% Dilate and Erode operations to remove noise
Idilate = Igray;
for i = 1:rows
    for j = 2:cols-1
        temp = max(Igray(i, j-1), Igray(i, j));
        Idilate(i, j) = max(temp, Igray(i, j+1));
    end
end
I = Idilate;

% Display intermediate results
figure(2);
imshow(Igray);
figure(3);
title('Dilated Image');
imshow(Idilate);
figure(4);
imshow(I);

% Processing horizontal and vertical edges
disp('Processing Edges Horizontally...');
horz = process_edges_horizontally(I, cols, rows);
vert = process_edges_vertically(I, cols, rows);

% Plot histograms
plot_histograms(horz, vert, cols, rows);

% Process histograms and find regions of interest
disp('Processing Histograms...');
[column, row, max_horz, max_vert] = process_histograms(I, horz, vert, cols, rows);

% Extract the region of interest (number plate)
ROI = extract_license_plate(I, column, row);

figure(8);
imshow(ROI);
title('Extracted Number Plate');

% Use OCR to read the number plate
results = ocr(ROI);
numberPlateText = results.Text;

% Display the recognized number plate
figure(9);
imshow(ROI);
title(['Recognized Number Plate: ', numberPlateText]);

% Save the processed image
imwrite(ROI, 'processed_image.jpg');

% Verify the image is saved
if exist('processed_image.jpg', 'file')
    disp('Processed image saved successfully.');
else
    disp('Error saving processed image.');
end

% Call the Python script using the system command
status = system('python download_and_compare.py');
if status == 0
    disp('Python script executed successfully.');
else
    disp('Error executing Python script.');
end

%% Function Definitions
function detect_license_plate(imagePath)
    % Main function to detect license plate from an image
    I = imread(imagePath); % Read the image
    figure; imshow(I); title('Original Image');

    % Preprocess the image
    Igray = preprocess_image(I);

    % Detect edges
    [horz, vert] = detect_edges(Igray);

    % Filter histograms
    horz = filter_histogram(horz);
    vert = filter_histogram(vert);

    % Extract regions of interest
    [column, row] = extract_regions(horz, vert);

    % Extract the license plate
    ROI = extract_license_plate(I, column, row);

    % Perform OCR
    numberPlateText = perform_ocr(ROI);

    % Display results
    display_results(ROI, numberPlateText);
end

function Igray = preprocess_image(I)
    % Convert to grayscale and apply morphological operations
    Igray = rgb2gray(I);
    Igray = imdilate(Igray, strel('rectangle', [3, 3])); % Dilation
    Igray = imerode(Igray, strel('rectangle', [3, 3])); % Erosion
end

function [horz, vert] = detect_edges(Igray)
    [rows, cols] = size(Igray);
    horz = zeros(1, cols);
    vert = zeros(1, rows);

    % Horizontal edge detection
    for i = 2:cols
        horz(i) = sum(abs(diff(Igray(:, i))) > 20);
    end

    % Vertical edge detection
    for i = 2:rows
        vert(i) = sum(abs(diff(Igray(i, :))) > 20);
    end
end

function filtered = filter_histogram(histogram)
    % Apply low-pass filter to smooth the histogram
    filtered = movmean(histogram, 41);
end

function [column, row] = extract_regions(horz, vert)
    % Extract probable segments for the number plate
    column = find(horz > mean(horz));
    row = find(vert > mean(vert));
end

function ROI = extract_license_plate(I, column, row)
    % Extract the region of interest for the license plate
    [rows, cols, ~] = size(I); % Get image dimensions

    % Calculate the center of the number plate region
    max_horz = max(column);
    max_vert = max(row);

    % Define the bounding box for the license plate
    top = max(1, max_vert - 15);         % Ensure the top is within bounds
    bottom = min(rows, max_vert + 15);   % Ensure the bottom is within bounds
    left = max(1, max_horz - 50);        % Ensure the left is within bounds
    right = min(cols, max_horz + 50);    % Ensure the right is within bounds

    % Extract the region of interest
    ROI = I(top:bottom, left:right, :);
end

function numberPlateText = perform_ocr(ROI)
    % Use OCR to read the number plate
    results = ocr(ROI);
    numberPlateText = results.Text;
end

function display_results(ROI, numberPlateText)
    % Display the extracted license plate
    figure; imshow(ROI);
    title(['Recognized Number Plate: ', numberPlateText]);
end

function horz = process_edges_horizontally(I, cols, rows)
    horz = zeros(1, cols);
    for i = 2:cols
        sum = 0;
        for j = 2:rows
            diff = abs(I(j, i) - I(j-1, i));
            if diff > 20
                sum = sum + diff;
            end
        end
        horz(i) = sum;
    end
end

function vert = process_edges_vertically(I, cols, rows)
    vert = zeros(1, rows);
    for i = 2:rows
        sum = 0;
        for j = 2:cols
            diff = abs(I(i, j) - I(i, j-1));
            if diff > 20
                sum = sum + diff;
            end
        end
        vert(i) = sum;
    end
end

function plot_histograms(horz, vert, cols, rows)
    figure(5);
    subplot(2, 1, 1);
    plot(horz);
    title('Horizontal Edge Processing Histogram');
    xlabel('Column Number');
    ylabel('Difference');

    subplot(2, 1, 2);
    plot(vert);
    title('Vertical Edge Processing Histogram');
    xlabel('Row Number');
    ylabel('Difference');
end

function [column, row, max_horz, max_vert] = process_histograms(I, horz, vert, cols, rows)
    horz = filter_histogram(horz);
    vert = filter_histogram(vert);

    column = find(horz > mean(horz));
    row = find(vert > mean(vert));

    max_horz = max(column);
    max_vert = max(row);
end


