%STARTUP MATLAB startup file for physics simulation project
%   This file is automatically executed when MATLAB starts in this directory

fprintf('Initializing MATLAB Physics Simulation Project...\n');

% Add project paths
projectRoot = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(projectRoot, 'src')));
addpath(genpath(fullfile(projectRoot, 'tests')));

% Set up project-specific preferences
set(0, 'DefaultFigureWindowStyle', 'docked');
set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultLineLineWidth', 1.5);

% Display project information
fprintf('\nProject paths added:\n');
fprintf('  - Source code: %s\n', fullfile(projectRoot, 'src'));
fprintf('  - Tests: %s\n', fullfile(projectRoot, 'tests'));

% Check MATLAB version
v = ver('MATLAB');
fprintf('\nMATLAB version: %s\n', v.Version);

if str2double(v.Version(1:4)) < 2023
    warning('This project is designed for MATLAB R2023b or later. Some features may not work correctly.');
end

% Check for Simulink
if license('test', 'Simulink')
    fprintf('Simulink is available.\n');
else
    warning('Simulink is not available. Simulink models will not be accessible.');
end

fprintf('\nProject initialized successfully!\n');
fprintf('Type "help <function_name>" for function documentation.\n');
fprintf('Run "buildfile" to execute build tasks.\n\n');