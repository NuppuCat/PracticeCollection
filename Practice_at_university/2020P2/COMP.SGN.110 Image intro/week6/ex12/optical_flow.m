%%
% Constructs a multimedia reader object
vidReader = VideoReader('visiontraffic.avi');
% Initialize the optical flow object
opticFlow = opticalFlowLK('NoiseThreshold', 0.009);
disp('change the threshold of noise, if it is too big,movement will be recognized as noise')
while hasFrame(vidReader)
    % Read the rgb frame
    frameRGB  = readFrame(vidReader);
    % Convert rgb to grayscale
    frameGray = rgb2gray(frameRGB);
    % Compute optical flow
    flow = estimateFlow(opticFlow, frameGray);
    % Display rgb video frame with flow vectors
    imshow(frameRGB);
    hold on;
    % ScaleFactor control the length of arrows.DecimationFactor control the
    % precision or density of arrows.
    plot(flow, 'DecimationFactor', [10 10], 'ScaleFactor', 10);
    drawnow;
    hold off;
end
%%
% Constructs a multimedia reader object

vidReader = VideoReader('echo1.avi');
% Initialize the optical flow object

opticFlow = opticalFlowLK('NoiseThreshold', 0.02);

while hasFrame(vidReader)
    % Read the rgb frame
    frameRGB  = readFrame(vidReader);
    % Convert rgb to grayscale
    frameGray = rgb2gray(frameRGB);
    fim= wiener2(frameGray,[3,3]);
    % Compute optical flow
    flow = estimateFlow(opticFlow, fim);
    % Display rgb video frame with flow vectors
    
    imshow(fim);
    hold on;
    plot(flow, 'DecimationFactor', [5 5], 'ScaleFactor', 10);
    drawnow;
    hold off;
end
%%
% Constructs a multimedia reader object

vidReader = VideoReader('echo2.avi');
% Initialize the optical flow object

opticFlow = opticalFlowLK('NoiseThreshold', 0.02);

while hasFrame(vidReader)
    % Read the rgb frame
    frameRGB  = readFrame(vidReader);
    % Convert rgb to grayscale
    frameGray = rgb2gray(frameRGB);
    % Compute optical flow
    fim = wiener2(frameGray,[3,3]);
    flow = estimateFlow(opticFlow, fim);
    % Display rgb video frame with flow vectors
    
    imshow(fim);
    hold on;
    plot(flow, 'DecimationFactor', [5 5], 'ScaleFactor', 10);
    drawnow;
    hold off;
end
disp('echo2 is unhealthy heart, bcs it have less movement.')
disp('e have less noise, less disturb when we watch the arrows changes')