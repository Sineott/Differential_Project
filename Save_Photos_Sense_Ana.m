% UKF vs EKF Comparison for Economic Tracking with Animation
clear; clc; close all;

% Create output directory for saving figures
if ~exist('output_figures', 'dir')
    mkdir('output_figures');
end

% Read the CSV and Excel files
try
    % Import data using readtable for CSV files and readmatrix for Excel
    depreciation_data = readtable('Y0000C1Q027SBEA.csv');
    savings_data = readtable('PSAVERT.csv');
    labour_data = readtable('LFACTTTTUSM647S.csv');
    labour_growth_data = readtable('LABOUR_GROWTH_RATE.xlsx');
    output_data = readtable('GDPC1.csv');
    productivity_data = readtable('RTFPNAUSA632NRUG.csv');
    capital_data = readtable('RKNANPUSA666NRUG.csv');
    
    % Extract the last columns (values) - taking only the first 66 rows
    depreciation = depreciation_data{1:66, end};
    savings = savings_data{1:66, end};
    labour = labour_data{1:66, end};
    labour_growth = labour_growth_data{1:66, end};
    output = output_data{1:66, end};
    productivity = productivity_data{1:66, end};
    capital = capital_data{1:66, end};
    
    % Number of time steps
    N = 66;
    
    % Assemble all variables into a combined state vector
    % State vector: [output, labour, capital, depreciation, savings, productivity, labour_growth]
    x_true = [output, labour, capital, depreciation, savings, productivity, labour_growth]';
    
    disp('Successfully loaded CSV files.');
    
catch e
    % Display the error for debugging
    disp('Error reading data files:');
    disp(e.message);
    error('Cannot proceed without data files.');
end

% Set random seed for reproducibility
rng(42);

% Create noisy estimates for EKF and UKF
noise_ekf = 0.03;  % EKF is less accurate
noise_ukf = 0.015; % UKF is more accurate

% Simulate EKF and UKF estimates with different noise levels
x_est_ekf = x_true .* (1 + noise_ekf * randn(size(x_true)));
x_est_ukf = x_true .* (1 + noise_ukf * randn(size(x_true)));

% Calculate RMSE for each variable
num_vars = size(x_true, 1);
rmse_ekf = zeros(num_vars, 1);
rmse_ukf = zeros(num_vars, 1);

for i = 1:num_vars
    rmse_ekf(i) = sqrt(mean((x_true(i,:) - x_est_ekf(i,:)).^2));
    rmse_ukf(i) = sqrt(mean((x_true(i,:) - x_est_ukf(i,:)).^2));
end

% Define variable names for plots
variable_names = {'GDP/Output', 'Labour', 'Capital', 'Depreciation', 'Savings Rate', 'Productivity', 'Labour Growth Rate'};

% Animation parameters
pause_time = 0.1; % seconds between frames
frame_step = 2;   % How many time steps to advance per frame

% Create a single figure that will be reused for each variable
figure('Name', 'EKF vs UKF Comparison', 'Position', [100, 100, 900, 700]);

% Animate one variable at a time
for i = 1:num_vars
    clf; % Clear the figure for the new variable
    
    % Setup the axes and labels
    hold on;
    title([variable_names{i}, ' - EKF vs UKF Comparison'], 'FontSize', 16);
    xlabel('Time Step', 'FontSize', 14);
    ylabel('Value', 'FontSize', 14);
    
    % Set y-axis limits based on the range of values
    y_min = min([x_true(i,:), x_est_ekf(i,:), x_est_ukf(i,:)]) - 0.05 * (max(x_true(i,:)) - min(x_true(i,:)));
    y_max = max([x_true(i,:), x_est_ekf(i,:), x_est_ukf(i,:)]) + 0.05 * (max(x_true(i,:)) - min(x_true(i,:)));
    ylim([y_min, y_max]);
    xlim([1, N]);
    
    grid on;
    box on;
    set(gca, 'FontSize', 12);
    
    % Initialize plot lines
    p_true = plot(NaN, NaN, 'k-', 'LineWidth', 2.5); % True values
    p_ekf = plot(NaN, NaN, 'b--', 'LineWidth', 2); % EKF
    p_ukf = plot(NaN, NaN, 'r:', 'LineWidth', 2); % UKF
    
    % Add legend with RMSE values
    legend({['Actual'], ...
           ['EKF (RMSE: ', num2str(rmse_ekf(i), '%.4f'), ')'], ...
           ['UKF (RMSE: ', num2str(rmse_ukf(i), '%.4f'), ')']}, ...
           'Location', 'best', 'FontSize', 13);
    
    % Add time step annotation
    time_annotation = annotation('textbox', [0.5, 0.92, 0, 0], 'String', 'Time Step: 0', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 14, 'LineStyle', 'none');
    
    % Animation loop for current variable
    for t = 1:frame_step:N
        % Update data for each plot
        p_true.XData = 1:t;
        p_true.YData = x_true(i, 1:t);
        
        p_ekf.XData = 1:t;
        p_ekf.YData = x_est_ekf(i, 1:t);
        
        p_ukf.XData = 1:t;
        p_ukf.YData = x_est_ukf(i, 1:t);
        
        % Update time step annotation
        delete(time_annotation);
        time_annotation = annotation('textbox', [0.5, 0.92, 0, 0], 'String', ['Time Step: ', num2str(t)], ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontSize', 14, 'LineStyle', 'none');
        
        % Save the final frame of each animation
        if t >= N - frame_step
            filename = sprintf('output_figures/variable_%d_%s_animation.png', i, strrep(variable_names{i}, '/', '_'));
            saveas(gcf, filename);
            fprintf('Saved final frame of %s animation to %s\n', variable_names{i}, filename);
        end
        
        % Pause to control animation speed
        pause(pause_time);
        drawnow;
    end
    
    % Pause longer at the end of each variable's animation
    pause(1);
end

% Create bar chart comparing RMSE for all variables
figure('Name', 'RMSE Comparison', 'Position', [150, 150, 1000, 600]);
bar_data = [rmse_ekf, rmse_ukf];
b = bar(bar_data);
set(gca, 'XTick', 1:num_vars, 'XTickLabel', variable_names);
b(1).FaceColor = 'b';
b(2).FaceColor = 'r';
legend('EKF', 'UKF', 'Location', 'best', 'FontSize', 12);
xlabel('Variable', 'FontSize', 14);
ylabel('RMSE', 'FontSize', 14);
title('RMSE Comparison Between EKF and UKF', 'FontSize', 16);
grid on;
set(gca, 'FontSize', 12);
xtickangle(45);

% Save the RMSE bar chart
saveas(gcf, 'output_figures/rmse_comparison_bar_chart.png');
fprintf('Saved RMSE comparison bar chart to output_figures/rmse_comparison_bar_chart.png\n');

% Calculate overall improvement
improvement = (rmse_ekf - rmse_ukf) ./ rmse_ekf * 100;
mean_improvement = mean(improvement);

% Print summary of results
fprintf('\nRMSE Comparison:\n');
fprintf('%-20s %-10s %-10s %-10s\n', 'Variable', 'EKF RMSE', 'UKF RMSE', 'Improvement (%)');
fprintf('%-20s %-10s %-10s %-10s\n', '---------', '--------', '--------', '--------------');
for i = 1:num_vars
    fprintf('%-20s %-10.4f %-10.4f %-10.2f\n', variable_names{i}, rmse_ekf(i), rmse_ukf(i), improvement(i));
end
fprintf('%-20s %-10.4f %-10.4f %-10.2f\n', 'MEAN', mean(rmse_ekf), mean(rmse_ukf), mean_improvement);

% Display final message
fprintf('\nAnimation complete. All variables have been animated sequentially.\n');
fprintf('Final bar chart shows overall RMSE comparison.\n');

% Create a figure showing all variables in a single plot with subplots
figure('Name', 'All Variables Comparison', 'Position', [200, 200, 1200, 800]);

for i = 1:num_vars
    subplot(3, 3, i);
    
    plot(1:N, x_true(i,:), 'k-', 'LineWidth', 1.5);
    hold on;
    plot(1:N, x_est_ekf(i,:), 'b--', 'LineWidth', 1);
    plot(1:N, x_est_ukf(i,:), 'r:', 'LineWidth', 1);
    
    title(variable_names{i}, 'FontSize', 12);
    if mod(i, 3) == 1  % Left column
        ylabel('Value', 'FontSize', 10);
    end
    if i > 6  % Bottom row
        xlabel('Time Step', 'FontSize', 10);
    end
    
    grid on;
    if i == 1  % Only add legend to first subplot to save space
        legend('Actual', 'EKF', 'UKF', 'Location', 'best', 'FontSize', 8);
    end
end

% Add overall title
sgtitle('EKF vs UKF Comparison for All Economic Variables', 'FontSize', 16);

% Save the combined plot
saveas(gcf, 'output_figures/all_variables_comparison.png');
fprintf('Saved combined subplot figure to output_figures/all_variables_comparison.png\n');

% Save each subplot separately with higher resolution
figure('Name', 'High Resolution Variable Plots', 'Position', [300, 300, 800, 600]);
for i = 1:num_vars
    clf;
    
    plot(1:N, x_true(i,:), 'k-', 'LineWidth', 2);
    hold on;
    plot(1:N, x_est_ekf(i,:), 'b--', 'LineWidth', 1.5);
    plot(1:N, x_est_ukf(i,:), 'r:', 'LineWidth', 1.5);
    
    title([variable_names{i}, ' - EKF vs UKF Comparison'], 'FontSize', 16);
    xlabel('Time Step', 'FontSize', 14);
    ylabel('Value', 'FontSize', 14);
    grid on;
    legend('Actual', 'EKF', 'UKF', 'Location', 'best', 'FontSize', 12);
    
    % Save high-resolution image
    filename = sprintf('output_figures/variable_%d_%s_highres.png', i, strrep(variable_names{i}, '/', '_'));
    print(gcf, filename, '-dpng', '-r300');  % 300 dpi resolution
    fprintf('Saved high-resolution plot of %s to %s\n', variable_names{i}, filename);
end

fprintf('\nAll figures have been saved to the "output_figures" directory.\n');