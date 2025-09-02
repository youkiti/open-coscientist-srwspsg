# Coscientist Test Suite Documentation

## Overview

The Coscientist test suite provides comprehensive testing coverage for the multi-agent scientific research framework. The suite is designed to ensure reliability, performance, and correctness across all components.

## Test Structure

### Test Categories

#### 🔧 Unit Tests (`test_*_agent.py`)
- **Purpose**: Test individual agent components in isolation
- **Coverage**: Agent logic, data processing, response parsing
- **Execution**: Fast (~1-5 seconds per test)
- **Dependencies**: Mocked LLMs and external services

**Files:**
- `test_literature_review_agent.py` - Literature review and research decomposition
- `test_generation_agent.py` - Hypothesis generation and parsing  
- `test_ranking_agent.py` - ELO tournament system and debates
- `test_reflection_agent.py` - Causal reasoning and assumption analysis
- `test_proximity_agent.py` - Embedding generation and graph analysis

#### 🔗 Integration Tests (`test_*_integration.py`)
- **Purpose**: Test component interactions and workflows
- **Coverage**: Agent communication, state transitions, data flow
- **Execution**: Medium speed (~5-15 seconds per test)
- **Dependencies**: Mocked LLMs with realistic response patterns

**Files:**
- `test_framework_integration.py` - Full framework pipeline testing
- `test_cli_integration.py` - Command-line interface functionality

#### 🚀 End-to-End Tests (`test_e2e_scenarios.py`)
- **Purpose**: Test complete research workflows from goal to report
- **Coverage**: Full system behavior with realistic scenarios
- **Execution**: Slow (~30-120 seconds per test)
- **Dependencies**: Comprehensive mocking of all external services

**Test Scenarios:**
- Alzheimer's disease research (quick scenario, 4 hypotheses)
- Climate change mitigation (complex scenario, 6+ hypotheses)
- Battery technology development (focused scenario, 3 hypotheses)
- Error recovery and checkpoint resume

#### ⚡ Performance Tests (`test_performance.py`)
- **Purpose**: Validate scalability and resource usage
- **Coverage**: Memory usage, execution time, concurrency
- **Execution**: Variable (~10-300 seconds depending on test)
- **Dependencies**: System monitoring tools (psutil)

**Test Areas:**
- Large hypothesis set handling (up to 50 hypotheses)
- Memory leak detection
- Tournament scaling performance
- State file size optimization

## Test Execution

### Quick Test Run
```bash
# Run all unit tests (fastest)
pytest tests/test_*_agent.py -v

# Run with coverage
pytest tests/test_*_agent.py --cov=coscientist --cov-report=html
```

### Integration Testing
```bash
# Run integration tests
pytest tests/test_*_integration.py -v -m integration

# Run with mocked dependencies
pytest tests/test_*_integration.py -v -m mock
```

### End-to-End Testing
```bash
# Run E2E scenarios (slow)
pytest tests/test_e2e_scenarios.py -v -m e2e

# Run specific scenario
pytest tests/test_e2e_scenarios.py::TestQuickResearchScenario -v
```

### Performance Testing
```bash
# Run performance benchmarks
pytest tests/test_performance.py -v -m performance

# Run stress tests
pytest tests/test_performance.py -v -m slow
```

### Real API Testing (Requires API Keys)
```bash
# Run tests with real APIs (requires API keys)
pytest tests/ -v -m requires_api

# Run specific API tests
pytest tests/ -v -m requires_openai
pytest tests/ -v -m requires_anthropic
pytest tests/ -v -m requires_tavily
```

### Full Test Suite
```bash
# Run everything except real API tests
pytest tests/ -v -m "not requires_api"

# Run complete suite including real APIs
pytest tests/ -v
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)
- Test discovery patterns
- Coverage settings (70% minimum)
- Marker definitions
- Timeout settings (5 minutes default)
- Async test support

### Test Markers
- `unit` - Unit tests
- `integration` - Integration tests  
- `e2e` - End-to-end tests
- `performance` - Performance tests
- `slow` - Tests taking >30 seconds
- `requires_api` - Tests needing real API keys
- `requires_openai` - OpenAI API required
- `requires_anthropic` - Anthropic API required
- `requires_google` - Google API required
- `requires_tavily` - Tavily API required
- `mock` - Tests using mocked dependencies

## Test Fixtures and Data

### Fixtures (`conftest.py`)
- `mock_env` - Mock environment variables
- `temp_dir` - Temporary directory for testing
- `test_state` - Pre-configured CoscientistState
- `mock_llm_pools` - Mocked LLM collections
- `sample_hypotheses` - Sample hypothesis data
- `mock_embeddings` - Mocked embedding generation

### Test Data (`fixtures/`)
- `sample_literature_reviews.json` - Literature review examples
- `sample_hypotheses.json` - Hypothesis examples by domain
- `sample_tournament_results.json` - Tournament outcomes and debates

### Utilities (`utils.py`)
- `MockLLM` - Mock language model for testing
- `MockGPTResearcher` - Mock research agent
- `create_mock_hypothesis()` - Hypothesis factory
- `create_test_state()` - State factory
- Memory management and cleanup utilities

## Writing New Tests

### Unit Test Template
```python
@pytest.mark.unit
@pytest.mark.asyncio
class TestNewAgent:
    async def test_basic_functionality(self, mock_llm):
        # Setup
        agent = NewAgent(llm=mock_llm)
        mock_llm.add_response("Expected response")
        
        # Execute
        result = await agent.some_method(input_data)
        
        # Verify
        assert result is not None
        assert mock_llm.call_count == 1
```

### Integration Test Template
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_component_interaction(mock_llm_pools, test_state):
    # Setup framework
    config = CoscientistConfig()
    framework = CoscientistFramework(config, test_state)
    
    # Mock responses
    mock_llm_pools["gpt-5"].add_response("Response 1")
    
    # Execute workflow
    result = await framework.some_workflow()
    
    # Verify integration
    assert result is not None
    assert test_state.some_field is not None
```

### E2E Test Template
```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_complete_scenario(mock_llm_pools, temp_dir, mock_env):
    # Full scenario setup
    goal = "Research question"
    config = CoscientistConfig()
    state = CoscientistState(goal=goal)
    framework = CoscientistFramework(config, state)
    
    # Mock all required responses
    # ... comprehensive mocking
    
    # Execute complete workflow
    final_report, meta_review = await framework.start(n_hypotheses=4)
    
    # Verify complete results
    assert final_report is not None
    assert len(state.hypotheses) == 4
```

## Testing Best Practices

### 1. Test Isolation
- Use fixtures for setup/teardown
- Clean up temporary files and state
- Reset global variables between tests

### 2. Mocking Strategy
- Mock external services (APIs, file system)
- Use realistic mock responses
- Test both success and failure cases

### 3. Assertion Guidelines
- Test specific behaviors, not implementation
- Use descriptive assertion messages
- Verify both positive and negative cases

### 4. Performance Considerations
- Mark slow tests with `@pytest.mark.slow`
- Use timeouts for long-running tests
- Monitor memory usage in performance tests

### 5. Error Handling
- Test exception scenarios
- Verify graceful degradation
- Test recovery mechanisms

## Continuous Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -e .[dev]
    - name: Run unit tests
      run: |
        pytest tests/test_*_agent.py -v --cov=coscientist
    - name: Run integration tests  
      run: |
        pytest tests/test_*_integration.py -v -m "not requires_api"
    - name: Run E2E tests
      run: |
        pytest tests/test_e2e_scenarios.py -v -m "e2e and mock"
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-unit
        name: pytest-unit
        entry: pytest tests/test_*_agent.py -v --tb=short
        language: system
        pass_filenames: false
        always_run: true
```

## Debugging Failed Tests

### Common Issues

#### 1. Mock Response Exhaustion
```python
# Problem: Not enough mock responses
mock_llm.add_response("Response 1")
# ... test uses 2+ responses

# Solution: Add sufficient responses
for i in range(5):
    mock_llm.add_response(f"Response {i}")
```

#### 2. Async Test Issues
```python
# Problem: Missing @pytest.mark.asyncio
def test_async_method():  # Wrong
    result = await some_async_method()

# Solution: Use proper decorator
@pytest.mark.asyncio
async def test_async_method():  # Correct
    result = await some_async_method()
```

#### 3. State Persistence Issues
```python
# Problem: State pollution between tests
def test_1():
    global_state.value = "test1"

def test_2():
    assert global_state.value is None  # May fail

# Solution: Use fixtures for isolation
@pytest.fixture
def clean_state():
    state = create_fresh_state()
    yield state
    cleanup_state(state)
```

### Debugging Commands
```bash
# Run single test with full output
pytest tests/test_file.py::test_name -v -s

# Drop into debugger on failure
pytest tests/test_file.py::test_name --pdb

# Show local variables on failure
pytest tests/test_file.py::test_name -l

# Run with maximum verbosity
pytest tests/test_file.py::test_name -vv --tb=long
```

## Coverage Analysis

### Generating Coverage Reports
```bash
# HTML coverage report
pytest tests/ --cov=coscientist --cov-report=html
# View: htmlcov/index.html

# Terminal coverage report
pytest tests/ --cov=coscientist --cov-report=term-missing

# XML coverage (for CI)
pytest tests/ --cov=coscientist --cov-report=xml
```

### Coverage Goals
- **Unit Tests**: >80% line coverage
- **Integration Tests**: >70% line coverage  
- **Critical Paths**: 100% coverage
- **Error Handling**: >90% coverage

## Troubleshooting

### Environment Issues
```bash
# Check test environment
python -m pytest --collect-only | grep tests/

# Verify imports
python -c "from coscientist.framework import CoscientistFramework"

# Check API keys (for real API tests)
echo $OPENAI_API_KEY | head -c 10
```

### Test Discovery Issues
```bash
# Force test discovery refresh  
pytest --cache-clear

# Show test collection
pytest --collect-only -q

# Check for import errors
pytest --import-mode=importlib
```

### Performance Issues
```bash
# Profile slow tests
pytest tests/ --durations=10

# Run tests in parallel (if pytest-xdist installed)
pytest tests/ -n auto

# Memory profiling
pytest tests/ --profile
```

## Contributing Test Cases

### Guidelines for New Tests
1. **Naming**: Use descriptive test names that explain the scenario
2. **Documentation**: Include docstrings for complex test logic
3. **Coverage**: Aim to test both success and failure paths
4. **Independence**: Tests should not depend on each other
5. **Determinism**: Avoid random behavior; use fixed seeds if needed

### Test Review Checklist
- [ ] Test name clearly describes scenario
- [ ] Appropriate markers applied (`@pytest.mark.unit`, etc.)
- [ ] Proper async handling if needed
- [ ] Mocks are realistic and sufficient
- [ ] Assertions are specific and meaningful
- [ ] Cleanup is handled properly
- [ ] Performance impact is reasonable

## Support

For test-related questions:
1. Check existing test examples in similar categories
2. Review fixture documentation in `conftest.py`
3. Consult utility functions in `utils.py`
4. Create issue in GitHub repository with test failure details

## Future Enhancements

### Planned Improvements
- [ ] Property-based testing for hypothesis generation
- [ ] Mutation testing for test quality assessment
- [ ] Load testing with realistic API rate limits
- [ ] Integration with external CI services
- [ ] Automated performance regression detection