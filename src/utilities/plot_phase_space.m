function plot_phase_space(t, x, v, varargin)
%PLOT_PHASE_SPACE Plots phase space diagram for a dynamical system
%   PLOT_PHASE_SPACE(t, x, v) plots the phase space (position vs velocity)
%   for a dynamical system with time t, position x, and velocity v.
%
%   PLOT_PHASE_SPACE(t, x, v, 'Property', Value, ...) allows customization:
%       'Title' - plot title (default: 'Phase Space Diagram')
%       'XLabel' - x-axis label (default: 'Position')
%       'YLabel' - y-axis label (default: 'Velocity')
%       'ColorMap' - colormap for time evolution (default: 'jet')
%       'LineWidth' - line width (default: 2)
%       'MarkerSize' - size of start/end markers (default: 10)

    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'Title', 'Phase Space Diagram', @ischar);
    addParameter(p, 'XLabel', 'Position', @ischar);
    addParameter(p, 'YLabel', 'Velocity', @ischar);
    addParameter(p, 'ColorMap', 'jet', @ischar);
    addParameter(p, 'LineWidth', 2, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'MarkerSize', 10, @(x) isnumeric(x) && x > 0);
    parse(p, varargin{:});
    
    % Create figure
    figure;
    hold on;
    
    % Create color mapping based on time
    colors = colormap(p.Results.ColorMap);
    nColors = size(colors, 1);
    timeIndices = round(linspace(1, nColors, length(t)));
    
    % Plot phase space trajectory with time-based coloring
    for i = 1:length(t)-1
        plot(x(i:i+1), v(i:i+1), '-', ...
            'Color', colors(timeIndices(i), :), ...
            'LineWidth', p.Results.LineWidth);
    end
    
    % Mark start and end points
    plot(x(1), v(1), 'go', 'MarkerSize', p.Results.MarkerSize, ...
        'MarkerFaceColor', 'g', 'DisplayName', 'Start');
    plot(x(end), v(end), 'ro', 'MarkerSize', p.Results.MarkerSize, ...
        'MarkerFaceColor', 'r', 'DisplayName', 'End');
    
    % Add colorbar for time
    c = colorbar;
    c.Label.String = 'Time';
    caxis([t(1) t(end)]);
    
    % Labels and formatting
    xlabel(p.Results.XLabel);
    ylabel(p.Results.YLabel);
    title(p.Results.Title);
    grid on;
    legend('Location', 'best');
    
    % Set axis properties
    axis equal;
    hold off;
end