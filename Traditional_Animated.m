clc; clear; close all;
% Model Parameters
alpha = 0.33; % Capital share in output (typical ~1/3)
s = 0.2; % Savings rate
delta = 0.05; % Depreciation rate
n = 0.02; % Labor growth rate
g = 0.015; % Technology growth rate
A0 = 1; % Initial technology level
L0 = 100; % Initial labor population
K0 = 50; % Initial capital stock
T = 100; % Simulation time horizon (years)
dt = 0.1; % Time step

% Time Vector
t = 0:dt:T;
N = length(t);

% Preallocate Arrays
K = zeros(1, N);
L = zeros(1, N);
A = zeros(1, N);
Y = zeros(1, N);

% Initial Conditions
K(1) = K0;
L(1) = L0;
A(1) = A0;

% Euler's Method for Time Evolution
for i = 1:N-1
    % Compute Output using Cobb-Douglas function
    Y(i) = A(i) * (K(i)^alpha) * (L(i)^(1 - alpha));
    
    % Update Capital Stock
    K(i+1) = K(i) + dt * (s * Y(i) - delta * K(i));
    
    % Update Labor (Exponential Growth)
    L(i+1) = L(i) * exp(n * dt);
    
    % Update Technology (Exponential Growth)
    A(i+1) = A(i) * exp(g * dt);
end

% Compute Final Output
Y(N) = A(N) * (K(N)^alpha) * (L(N)^(1 - alpha));

% Create output directory for saving figures if it doesn't exist
output_dir = 'solow_model_output';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Create figure for animation and VideoWriter
figure('Position', [100, 100, 1000, 800]);
sgtitle('Solow Growth Model Simulation', 'FontSize', 16);

% Set up the VideoWriter
videoFileName = fullfile(output_dir, 'SolowModelAnimation.mp4');
videoObj = VideoWriter(videoFileName, 'MPEG-4');
videoObj.FrameRate = 60;  % Increased from 30 to 60 fps
videoObj.Quality = 100;
open(videoObj);

% Animation speed control (higher = faster animation, skips more frames)
animationControl = 10;  % Increased from 2 to 10 for faster animation

% Create full plots first (this will be our "background" that grows)
subplot(2,2,1);
p1_full = plot(t(1), K(1), 'b-', 'LineWidth', 2);
hold on;
h1 = plot(t(1), K(1), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 6);
xlabel('Time (years)'); ylabel('Capital Stock (K)');
title('Capital Accumulation'); grid on;
xlim([0 T]); ylim([0 max(K)*1.1]);

subplot(2,2,2);
p2_full = plot(t(1), L(1), 'r-', 'LineWidth', 2);
hold on;
h2 = plot(t(1), L(1), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
xlabel('Time (years)'); ylabel('Labor (L)');
title('Labor Growth'); grid on;
xlim([0 T]); ylim([0 max(L)*1.1]);

subplot(2,2,3);
p3_full = plot(t(1), A(1), 'g-', 'LineWidth', 2);
hold on;
h3 = plot(t(1), A(1), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 6);
xlabel('Time (years)'); ylabel('Technology Level (A)');
title('Technology Growth'); grid on;
xlim([0 T]); ylim([0 max(A)*1.1]);

subplot(2,2,4);
p4_full = plot(t(1), Y(1), 'm-', 'LineWidth', 2);
hold on;
h4 = plot(t(1), Y(1), 'mo', 'MarkerFaceColor', 'm', 'MarkerSize', 6);
xlabel('Time (years)'); ylabel('Output (Y)');
title('Economic Output (GDP)'); grid on;
xlim([0 T]); ylim([0 max(Y)*1.1]);

% Animation loop with growing trace instead of moving tail
for i = 2:animationControl:N
    % The key change: we always show the entire trace from the beginning up to the current point
    current_index = i;
    
    % Update plots with complete trace from beginning to current point
    set(p1_full, 'XData', t(1:current_index), 'YData', K(1:current_index));
    set(h1, 'XData', t(current_index), 'YData', K(current_index));
    
    set(p2_full, 'XData', t(1:current_index), 'YData', L(1:current_index));
    set(h2, 'XData', t(current_index), 'YData', L(current_index));
    
    set(p3_full, 'XData', t(1:current_index), 'YData', A(1:current_index));
    set(h3, 'XData', t(current_index), 'YData', A(current_index));
    
    set(p4_full, 'XData', t(1:current_index), 'YData', Y(1:current_index));
    set(h4, 'XData', t(current_index), 'YData', Y(current_index));
    
    drawnow;
    
    % Capture the frame for video
    frame = getframe(gcf);
    writeVideo(videoObj, frame);
end

% Close the video file
close(videoObj);

% Display completion message
fprintf('Animation saved as %s\n', videoFileName);

% Show a static plot of all data at the end and save it
fig_final = figure('Position', [100, 100, 1000, 800]);
sgtitle('Final Solow Model Results', 'FontSize', 16);

subplot(2,2,1);
plot(t, K, 'b', 'LineWidth', 2);
xlabel('Time (years)'); ylabel('Capital Stock (K)');
title('Capital Accumulation'); grid on;

subplot(2,2,2);
plot(t, L, 'r', 'LineWidth', 2);
xlabel('Time (years)'); ylabel('Labor (L)');
title('Labor Growth'); grid on;

subplot(2,2,3);
plot(t, A, 'g', 'LineWidth', 2);
xlabel('Time (years)'); ylabel('Technology Level (A)');
title('Technology Growth'); grid on;

subplot(2,2,4);
plot(t, Y, 'm', 'LineWidth', 2);
xlabel('Time (years)'); ylabel('Output (Y)');
title('Economic Output (GDP)'); grid on;

% Save the final figure
finalFigurePath = fullfile(output_dir, 'SolowModel_FinalResults.png');
saveas(fig_final, finalFigurePath);
fprintf('Final plot saved as %s\n', finalFigurePath);

% Save the data for future reference
dataFilePath = fullfile(output_dir, 'SolowModelData.mat');
save(dataFilePath, 't', 'K', 'L', 'A', 'Y', 'alpha', 's', 'delta', 'n', 'g');
fprintf('Data saved as %s\n', dataFilePath);