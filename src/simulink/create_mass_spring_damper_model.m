function create_mass_spring_damper_model()
%CREATE_MASS_SPRING_DAMPER_MODEL Creates a Simulink model for mass-spring-damper system
%   This function creates a Simulink model that simulates a mass-spring-damper
%   system with configurable parameters.

    % Model name
    modelName = 'mass_spring_damper_system';
    
    % Close model if it exists
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end
    
    % Create new model
    new_system(modelName);
    
    % Add blocks
    % Step input
    add_block('simulink/Sources/Step', [modelName '/Force Input']);
    set_param([modelName '/Force Input'], 'Position', [50, 100, 80, 130]);
    set_param([modelName '/Force Input'], 'Time', '1');
    set_param([modelName '/Force Input'], 'After', '10');
    
    % Gain for 1/mass
    add_block('simulink/Math Operations/Gain', [modelName '/1_over_Mass']);
    set_param([modelName '/1_over_Mass'], 'Position', [150, 100, 180, 130]);
    set_param([modelName '/1_over_Mass'], 'Gain', '1/mass');
    
    % First integrator (velocity)
    add_block('simulink/Continuous/Integrator', [modelName '/Velocity']);
    set_param([modelName '/Velocity'], 'Position', [250, 100, 280, 130]);
    set_param([modelName '/Velocity'], 'InitialCondition', '0');
    
    % Second integrator (position)
    add_block('simulink/Continuous/Integrator', [modelName '/Position']);
    set_param([modelName '/Position'], 'Position', [350, 100, 380, 130]);
    set_param([modelName '/Position'], 'InitialCondition', '0');
    
    % Spring gain
    add_block('simulink/Math Operations/Gain', [modelName '/Spring']);
    set_param([modelName '/Spring'], 'Position', [350, 200, 380, 230]);
    set_param([modelName '/Spring'], 'Gain', '-k');
    
    % Damper gain
    add_block('simulink/Math Operations/Gain', [modelName '/Damper']);
    set_param([modelName '/Damper'], 'Position', [250, 200, 280, 230]);
    set_param([modelName '/Damper'], 'Gain', '-b');
    
    % Sum block
    add_block('simulink/Math Operations/Add', [modelName '/Sum']);
    set_param([modelName '/Sum'], 'Position', [100, 95, 120, 135]);
    set_param([modelName '/Sum'], 'Inputs', '+++');
    
    % Scope for position
    add_block('simulink/Sinks/Scope', [modelName '/Position Scope']);
    set_param([modelName '/Position Scope'], 'Position', [450, 100, 480, 130]);
    
    % Scope for velocity
    add_block('simulink/Sinks/Scope', [modelName '/Velocity Scope']);
    set_param([modelName '/Velocity Scope'], 'Position', [300, 40, 330, 70]);
    
    % To Workspace blocks
    add_block('simulink/Sinks/To Workspace', [modelName '/Position Output']);
    set_param([modelName '/Position Output'], 'Position', [450, 160, 510, 190]);
    set_param([modelName '/Position Output'], 'VariableName', 'position_data');
    
    add_block('simulink/Sinks/To Workspace', [modelName '/Velocity Output']);
    set_param([modelName '/Velocity Output'], 'Position', [300, 160, 360, 190]);
    set_param([modelName '/Velocity Output'], 'VariableName', 'velocity_data');
    
    % Connect blocks
    add_line(modelName, 'Force Input/1', 'Sum/1');
    add_line(modelName, 'Sum/1', '1_over_Mass/1');
    add_line(modelName, '1_over_Mass/1', 'Velocity/1');
    add_line(modelName, 'Velocity/1', 'Position/1');
    add_line(modelName, 'Position/1', 'Position Scope/1');
    add_line(modelName, 'Position/1', 'Position Output/1');
    add_line(modelName, 'Velocity/1', 'Velocity Scope/1');
    add_line(modelName, 'Velocity/1', 'Velocity Output/1');
    
    % Feedback connections
    add_line(modelName, 'Position/1', 'Spring/1');
    add_line(modelName, 'Velocity/1', 'Damper/1');
    add_line(modelName, 'Spring/1', 'Sum/2');
    add_line(modelName, 'Damper/1', 'Sum/3');
    
    % Model workspace parameters
    hws = get_param(modelName, 'modelworkspace');
    hws.assignin('mass', 1);    % 1 kg
    hws.assignin('k', 100);     % Spring constant (N/m)
    hws.assignin('b', 10);      % Damping coefficient (N*s/m)
    
    % Set simulation parameters
    set_param(modelName, 'StopTime', '10');
    set_param(modelName, 'Solver', 'ode45');
    
    % Save the model
    save_system(modelName, fullfile(pwd, 'src', 'simulink', [modelName '.slx']));
    
    fprintf('Simulink model "%s" created successfully.\n', modelName);
    fprintf('Default parameters: mass=1kg, k=100N/m, b=10N*s/m\n');
end