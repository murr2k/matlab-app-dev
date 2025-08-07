# Enhancement: Implement Graphical Solution Termination Detection for matlab-computational-engineer

## Summary
Implement intelligent detection of graphical solution delivery in the `matlab-computational-engineer` agent. When MATLAB generates plots, charts, or figures as the solution to a problem, the agent must recognize this graphical output as a valid terminating condition and stop processing without over-analyzing or attempting additional computations.

## Background

### Current Limitation
The `matlab-computational-engineer` agent lacks awareness of when MATLAB has delivered a graphical solution, leading to:
- **Over-solving**: Continuing computation after the visual answer is complete
- **Redundant processing**: Attempting numeric analysis of inherently visual problems  
- **Poor user experience**: Generating unnecessary explanations for self-evident plots
- **Resource waste**: Extended MATLAB sessions for completed graphical solutions

### MATLAB Engine API Graphics Behavior
Based on documentation analysis:
- ✅ **Graphics Creation**: Engine API spawns MATLAB figure windows successfully
- ❌ **Event Communication**: No direct callbacks/events from MATLAB to Python  
- ✅ **Headless Support**: Works in CI environments with invisible figures
- ✅ **Indirect Detection**: Graphics state can be monitored via workspace inspection

## Technical Requirements

### 1. Graphical Problem Classification
```python
GRAPHICAL_PROBLEM_INDICATORS = {
    'explicit_visualization': [
        'plot', 'graph', 'chart', 'visualize', 'show', 'display',
        'histogram', 'scatter', 'bar chart', 'line plot', 'surface'
    ],
    'analysis_with_graphics': [
        'analyze and plot', 'compare graphically', 'visualize the relationship',
        'show the distribution', 'plot the function', 'graph the solution'
    ],
    'inherently_visual': [
        'frequency response', 'time series plot', 'phase diagram',
        'contour plot', 'mesh', 'waterfall', 'spectrogram'
    ]
}

def classify_graphical_intent(problem_statement):
    """Determine if problem expects graphical solution"""
    statement_lower = problem_statement.lower()
    
    for category, keywords in GRAPHICAL_PROBLEM_INDICATORS.items():
        if any(keyword in statement_lower for keyword in keywords):
            return {
                'is_graphical': True,
                'category': category,
                'confidence': 'high' if category == 'explicit_visualization' else 'medium'
            }
    
    # Secondary indicators (functions that commonly produce plots)
    if any(func in statement_lower for func in ['fft', 'bode', 'nyquist', 'step', 'impulse']):
        return {'is_graphical': True, 'category': 'analysis_with_graphics', 'confidence': 'medium'}
        
    return {'is_graphical': False, 'category': None, 'confidence': 'high'}
```

### 2. Graphics State Detection System
```python
class GraphicsStateMonitor:
    def __init__(self, matlab_engine):
        self.engine = matlab_engine
        self.initial_figure_count = 0
        self.monitoring_graphics = False
        
    def start_monitoring(self):
        """Initialize graphics monitoring before computation"""
        self.initial_figure_count = self._get_figure_count()
        self.monitoring_graphics = True
        
    def check_graphics_completion(self):
        """Check if graphics have been generated as solution"""
        if not self.monitoring_graphics:
            return False
            
        current_figure_count = self._get_figure_count()
        graphics_created = current_figure_count > self.initial_figure_count
        
        if graphics_created:
            return {
                'graphics_generated': True,
                'figure_count': current_figure_count,
                'new_figures': current_figure_count - self.initial_figure_count,
                'termination_reason': 'graphical_solution_delivered'
            }
        
        return {'graphics_generated': False}
    
    def _get_figure_count(self):
        """Get current number of open MATLAB figures"""
        try:
            # Get all figure handles
            figures = self.engine.get(0, 'Children')
            if figures is None:
                return 0
            # Handle both single figure and figure array cases
            return len(figures) if hasattr(figures, '__len__') else (1 if figures else 0)
        except:
            return 0
            
    def get_figure_details(self):
        """Extract figure information for termination report"""
        try:
            figures = self.engine.get(0, 'Children')
            if not figures:
                return []
                
            figure_info = []
            for fig in (figures if hasattr(figures, '__len__') else [figures]):
                info = {
                    'number': self.engine.get(fig, 'Number'),
                    'name': self.engine.get(fig, 'Name'),
                    'visible': self.engine.get(fig, 'Visible')
                }
                figure_info.append(info)
            return figure_info
        except:
            return []
```

### 3. Enhanced Problem Execution with Graphics Termination
```python
class MATLABComputationalEngine:
    def __init__(self):
        self.engine = None
        self.graphics_monitor = None
        
    def execute_with_termination_conditions(self, problem_statement):
        """Execute with intelligent termination based on solution type"""
        
        # 1. Classify problem type and graphics expectation
        classification = self.classify_problem(problem_statement)
        graphics_intent = classify_graphical_intent(problem_statement)
        
        # 2. Configure execution strategy
        if graphics_intent['is_graphical']:
            return self._execute_graphical_problem(problem_statement, classification, graphics_intent)
        else:
            return self._execute_non_graphical_problem(problem_statement, classification)
    
    def _execute_graphical_problem(self, problem, classification, graphics_intent):
        """Execute problems expecting graphical solutions"""
        
        # Initialize graphics monitoring
        self.graphics_monitor = GraphicsStateMonitor(self.engine)
        self.graphics_monitor.start_monitoring()
        
        try:
            # Execute MATLAB computation
            if classification['type'] == 'symbolic':
                result = self._handle_symbolic_with_graphics(problem)
            elif classification['type'] == 'signal_processing':
                result = self._handle_signal_processing_with_graphics(problem)
            elif classification['type'] == 'analysis':
                result = self._handle_analysis_with_graphics(problem)
            else:
                result = self._handle_general_with_graphics(problem)
            
            # Check for graphics completion
            graphics_status = self.graphics_monitor.check_graphics_completion()
            
            if graphics_status['graphics_generated']:
                # TERMINATE: Graphics solution delivered
                figure_details = self.graphics_monitor.get_figure_details()
                
                return {
                    'result': result,
                    'solution_type': 'graphical',
                    'termination_reason': 'graphical_solution_delivered',
                    'graphics_info': {
                        'figures_created': graphics_status['new_figures'],
                        'total_figures': graphics_status['figure_count'],
                        'figure_details': figure_details
                    },
                    'message': f"Graphical solution delivered in {graphics_status['new_figures']} figure(s). Computation complete.",
                    'stopped_processing': True
                }
            else:
                # No graphics generated - may need different handling
                return {
                    'result': result,
                    'solution_type': 'non_graphical',
                    'warning': 'Expected graphical solution but no figures were generated',
                    'graphics_info': {'figures_created': 0}
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'solution_type': 'error',
                'graphics_info': {'figures_created': 0}
            }
    
    def _handle_signal_processing_with_graphics(self, problem):
        """Handle signal processing problems that typically generate plots"""
        # Examples: FFT analysis, filter responses, spectrograms
        self.engine.eval("figure('Name', 'Signal Analysis')", nargout=0)
        result = self.engine.eval(problem)
        
        # Common signal processing visualization commands
        if 'fft' in problem.lower():
            self.engine.eval("xlabel('Frequency (Hz)'); ylabel('Magnitude')", nargout=0)
        elif 'filter' in problem.lower():
            self.engine.eval("xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)')", nargout=0)
            
        return result
    
    def _handle_analysis_with_graphics(self, problem):
        """Handle analysis problems expecting visualization"""
        self.engine.eval("figure('Name', 'Analysis Results')", nargout=0)
        result = self.engine.eval(problem)
        
        # Auto-add common visualization enhancements
        self.engine.eval("grid on; title('Analysis Results')", nargout=0)
        return result
```

### 4. Headless Environment Handling
```python
def configure_headless_graphics(self):
    """Configure graphics for headless CI environments"""
    try:
        # Detect if running in headless mode
        has_display = self.engine.usejava('desktop')
        
        if not has_display:
            # Configure for headless operation
            self.engine.eval("set(0, 'DefaultFigureVisible', 'off')", nargout=0)
            self.headless_mode = True
            
            return {
                'headless_configured': True,
                'message': 'Graphics configured for headless environment'
            }
        else:
            self.headless_mode = False
            return {
                'headless_configured': False,
                'message': 'Interactive graphics mode enabled'
            }
    except:
        # Assume headless if detection fails
        self.headless_mode = True
        return {
            'headless_configured': True,
            'message': 'Defaulting to headless mode due to detection failure'
        }

def save_graphics_in_headless_mode(self):
    """Save figures to files when running headless"""
    if self.headless_mode:
        figure_count = self.graphics_monitor._get_figure_count()
        saved_files = []
        
        for i in range(1, figure_count + 1):
            filename = f"matlab_solution_figure_{i}.png"
            self.engine.eval(f"figure({i}); saveas(gcf, '{filename}')", nargout=0)
            saved_files.append(filename)
            
        return {
            'files_saved': saved_files,
            'message': f'Saved {len(saved_files)} figures to files for headless environment'
        }
    
    return {'files_saved': [], 'message': 'Interactive mode - figures displayed in windows'}
```

## Implementation Examples

### Example 1: Signal Processing Problem
```python
# Input: "Generate FFT analysis of a 1kHz sine wave"
problem = "t = 0:1/1000:1; signal = sin(2*pi*1000*t); Y = fft(signal); f = (0:length(Y)-1)*1000/length(Y); plot(f, abs(Y)); title('FFT Analysis')"

result = engine.execute_with_termination_conditions(problem)

# Expected Result:
{
    'result': <matlab_computation_result>,
    'solution_type': 'graphical',
    'termination_reason': 'graphical_solution_delivered',
    'graphics_info': {
        'figures_created': 1,
        'total_figures': 1,
        'figure_details': [{'number': 1, 'name': '', 'visible': 'on'}]
    },
    'message': 'Graphical solution delivered in 1 figure(s). Computation complete.',
    'stopped_processing': True
}
```

### Example 2: Multiple Graphics Solution
```python
# Input: "Compare original signal vs filtered signal"
problem = "t = 0:0.01:1; signal = sin(2*pi*5*t) + 0.5*randn(size(t)); filtered = lowpass(signal, 2, 100); subplot(2,1,1); plot(t,signal); title('Original'); subplot(2,1,2); plot(t,filtered); title('Filtered');"

result = engine.execute_with_termination_conditions(problem)

# Expected Result:
{
    'result': <matlab_computation_result>,
    'solution_type': 'graphical',
    'termination_reason': 'graphical_solution_delivered',
    'graphics_info': {
        'figures_created': 1,
        'total_figures': 1,
        'figure_details': [{'number': 1, 'name': 'Signal Comparison', 'visible': 'on'}]
    },
    'message': 'Graphical solution delivered in 1 figure(s). Computation complete.',
    'stopped_processing': True
}
```

### Example 3: Non-Graphical Problem (No Early Termination)
```python
# Input: "Calculate the square root of 64"
problem = "sqrt(64)"

result = engine.execute_with_termination_conditions(problem)

# Expected Result:
{
    'result': 8.0,
    'solution_type': 'numeric',
    'termination_reason': 'numeric_computation_complete',
    'graphics_info': {'figures_created': 0},
    'message': 'Numeric solution: 8.0',
    'stopped_processing': False
}
```

## Testing Strategy

### Unit Tests
```python
class TestGraphicalTermination:
    def test_graphics_detection_explicit_plot(self):
        engine = MATLABComputationalEngine()
        result = engine.execute_with_termination_conditions("plot([1,2,3,4])")
        assert result['solution_type'] == 'graphical'
        assert result['termination_reason'] == 'graphical_solution_delivered'
        assert result['graphics_info']['figures_created'] > 0
    
    def test_no_graphics_numeric_problem(self):
        engine = MATLABComputationalEngine()
        result = engine.execute_with_termination_conditions("2 + 2")
        assert result['solution_type'] == 'numeric'
        assert result['graphics_info']['figures_created'] == 0
    
    def test_headless_mode_file_saving(self):
        engine = MATLABComputationalEngine()
        engine.headless_mode = True
        result = engine.execute_with_termination_conditions("plot([1,2,3])")
        assert 'files_saved' in result
        assert len(result['files_saved']) > 0
```

### Integration Tests
```python
def test_signal_processing_termination():
    # Test FFT analysis terminates after plot generation
    engine = MATLABComputationalEngine()
    problem = "t = 0:0.1:10; signal = sin(t); Y = fft(signal); plot(abs(Y))"
    result = engine.execute_with_termination_conditions(problem)
    
    assert result['solution_type'] == 'graphical'
    assert result['stopped_processing'] == True
    
def test_multiple_figures_detection():
    # Test detection of multiple figure creation
    engine = MATLABComputationalEngine()
    problem = "figure(1); plot([1,2,3]); figure(2); plot([4,5,6])"
    result = engine.execute_with_termination_conditions(problem)
    
    assert result['graphics_info']['figures_created'] == 2
```

### CI/CD Integration
```yaml
# .github/workflows/test-graphical-termination.yml
- name: Test Graphics Termination in Headless Mode
  run: |
    export DISPLAY=""  # Force headless
    python -m pytest tests/test_graphical_termination.py -v
    
- name: Verify Saved Graphics Files
  run: |
    ls -la matlab_solution_figure_*.png
    test -f matlab_solution_figure_1.png
```

## Benefits

### 1. Intelligent Problem Solving
- **Appropriate Termination**: Stop when graphical solution is delivered
- **Resource Efficiency**: Avoid unnecessary computation after visual answer
- **User Experience**: Provide clear feedback about solution delivery method

### 2. Robust Graphics Handling  
- **Multi-Environment**: Works in both interactive and headless modes
- **File Management**: Automatic figure saving in CI environments
- **Error Recovery**: Graceful handling of graphics failures

### 3. Enhanced Agent Intelligence
- **Problem Classification**: Understand when graphics are expected
- **Solution Recognition**: Identify visual solutions as complete answers
- **Stopping Conditions**: Prevent over-analysis of inherently visual problems

## Success Metrics

### Technical Metrics
- [ ] **Graphics Detection Accuracy**: 95% correct identification of graphical solutions
- [ ] **Termination Efficiency**: 80% reduction in unnecessary post-graphics processing
- [ ] **Headless Compatibility**: 100% functionality in CI/CD environments
- [ ] **Figure Management**: Proper cleanup and file saving in all modes

### Performance Metrics
- [ ] **Response Time**: <2 seconds for graphics detection and termination
- [ ] **Memory Usage**: Efficient figure handle management without leaks
- [ ] **CI/CD Integration**: Zero graphics-related failures in automated testing

### User Experience Metrics
- [ ] **Solution Clarity**: Clear indication when graphics provide the answer
- [ ] **Appropriate Stopping**: No over-explanation of self-evident visual results
- [ ] **Error Communication**: Helpful messages when expected graphics aren't generated

## Implementation Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1 | Week 1 | Graphics state monitoring system |
| 2 | Week 2 | Problem classification and termination logic |
| 3 | Week 3 | Headless mode support and file management |
| 4 | Week 4 | Testing, integration, and documentation |

## Related Issues
- #2 - Enhancement: Upgrade matlab-computational-engineer Agent with MATLAB Engine API for Python
- #1 - Enhancement: Integrate MATLAB Engine API for Python

## Risk Mitigation

### Technical Risks
- **Graphics Detection Failure**: Implement fallback mechanisms for figure counting
- **Headless Mode Issues**: Comprehensive testing across CI environments  
- **Memory Leaks**: Proper figure handle cleanup and session management
- **MATLAB Version Compatibility**: Test across MATLAB releases

### Performance Risks
- **Figure Monitoring Overhead**: Optimize graphics state checking frequency
- **File I/O in Headless Mode**: Efficient file saving without blocking
- **Session Management**: Prevent figure accumulation across multiple problems

---

**Labels**: `enhancement`, `matlab-computational-engineer`, `graphics`, `termination-conditions`, `intelligent-stopping`  
**Milestone**: v2.2 - Intelligent Graphics Termination  
**Assignee**: @murr2k  
**Priority**: High  
**Dependencies**: Issue #2 (MATLAB Engine API Upgrade)