function plan = buildfile
%BUILDFILE Build file for MATLAB project
%   This file defines build tasks for the project

% Create a plan from task functions
plan = buildplan(localfunctions);

% Configure the "build" task
plan("build").Description = "Build the MATLAB project";
plan("build").Dependencies = ["check", "test"];

% Configure the "test" task
plan("test").Description = "Run all unit tests";
plan("test").Dependencies = "check";

% Configure the "check" task  
plan("check").Description = "Check code for issues";

% Make "build" the default task
plan.DefaultTasks = "build";

end

function checkTask(~)
%CHECKTASK Check code for issues using checkcode

fprintf('Checking code quality...\n');

% Get all MATLAB files in src directory
srcFiles = dir(fullfile('src', '**', '*.m'));
hasIssues = false;

for i = 1:length(srcFiles)
    filePath = fullfile(srcFiles(i).folder, srcFiles(i).name);
    issues = checkcode(filePath, '-struct');
    
    if ~isempty(issues)
        hasIssues = true;
        fprintf('\nIssues in %s:\n', filePath);
        for j = 1:length(issues)
            fprintf('  Line %d: %s\n', issues(j).line, issues(j).message);
        end
    end
end

if hasIssues
    error('Code quality issues found. Please fix them before building.');
else
    fprintf('All code quality checks passed!\n');
end

end

function testTask(~)
%TESTTASK Run all unit tests

fprintf('Running unit tests...\n');

% Import test framework
import matlab.unittest.TestRunner;
import matlab.unittest.TestSuite;
import matlab.unittest.plugins.CodeCoveragePlugin;
import matlab.unittest.plugins.codecoverage.CoberturaFormat;

% Create test suite from unit test folder
suite = TestSuite.fromFolder('tests/unit');

% Create test runner with coverage
runner = TestRunner.withTextOutput;

% Add code coverage plugin
sourceFolder = fullfile(pwd, 'src');
reportFolder = fullfile(pwd, 'code-coverage');
if ~exist(reportFolder, 'dir')
    mkdir(reportFolder);
end

coverageFile = fullfile(reportFolder, 'coverage.xml');
plugin = CodeCoveragePlugin.forFolder(sourceFolder, ...
    'Producing', CoberturaFormat(coverageFile));
runner.addPlugin(plugin);

% Run tests
results = runner.run(suite);

% Display summary
fprintf('\n========== Test Summary ==========\n');
fprintf('Total tests: %d\n', numel(results));
fprintf('Passed: %d\n', sum([results.Passed]));
fprintf('Failed: %d\n', sum([results.Failed]));
fprintf('Coverage report: %s\n', coverageFile);
fprintf('==================================\n');

% Throw error if any tests failed
if any([results.Failed])
    error('Some tests failed. Please fix them before building.');
end

end

function buildTask(~)
%BUILDTASK Main build task

fprintf('\n========== Build Complete ==========\n');
fprintf('All checks and tests passed successfully!\n');
fprintf('Project is ready for deployment.\n');
fprintf('====================================\n');

end