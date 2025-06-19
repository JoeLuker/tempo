# Contributing to TEMPO

Thank you for your interest in contributing to TEMPO! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- **Be respectful**: Treat all contributors with respect and courtesy
- **Be inclusive**: Welcome contributors of all backgrounds and experience levels
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Remember that everyone is learning and improving
- **Be professional**: Keep discussions focused on the project

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. Python 3.8 or higher
2. Node.js 16 or higher (for frontend contributions)
3. Git installed and configured
4. A GitHub account
5. Familiarity with Git workflows

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/tempo.git
   cd tempo
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/JoeLuker/tempo.git
   ```

4. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-test.txt
   pip install -e .
   ```

6. **Install pre-commit hooks**:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## How to Contribute

### Reporting Issues

Before creating an issue:

1. **Search existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, GPU)
   - Error messages and logs

#### Bug Report Template

```markdown
**Describe the bug**
A clear and concise description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. Set parameters '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots/Logs**
If applicable, add screenshots or log outputs.

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.6]
- CUDA version: [e.g., 11.8]
- GPU: [e.g., NVIDIA RTX 3090]

**Additional context**
Any other context about the problem.
```

### Suggesting Enhancements

For feature requests:

1. **Check existing requests** first
2. **Explain the motivation** for the feature
3. **Describe the solution** you'd like
4. **Consider alternatives** you've thought about
5. **Provide examples** of how it would be used

### Pull Request Process

#### 1. Choose or Create an Issue

- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to claim it
- If creating a new feature, discuss it in an issue first

#### 2. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# Or for bug fixes
git checkout -b fix/issue-description
```

#### 3. Make Your Changes

Follow these guidelines:

- **Write clean, readable code** following our style guide
- **Add tests** for new functionality
- **Update documentation** as needed
- **Keep commits focused** and atomic
- **Write clear commit messages** following conventional commits

##### Commit Message Format

```
type(scope): subject

body

footer
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```bash
git commit -m "feat(pruning): add entropy-based pruning strategy"
git commit -m "fix(api): handle empty prompt gracefully"
git commit -m "docs: update installation instructions for M1 Macs"
```

#### 4. Test Your Changes

```bash
# Run unit tests
python run_tests.py --unit-only

# Run integration tests
python run_tests.py --integration-only

# Run linting
black src/
isort src/
flake8 src/

# Run frontend tests (if applicable)
cd frontend && npm test
```

#### 5. Update Documentation

- Update README.md if needed
- Add/update docstrings
- Update relevant documentation in `docs/`
- Add examples if introducing new features

#### 6. Submit Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub** with:
   - Clear title describing the change
   - Reference to related issue(s)
   - Description of changes made
   - Screenshots/examples if applicable
   - Checklist completion

##### PR Description Template

```markdown
## Description
Brief description of changes made.

## Related Issue
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Changes Made
- Added X functionality
- Fixed Y bug
- Updated Z documentation

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] Commits are atomic and well-described
```

#### 7. Code Review Process

- **Respond promptly** to review feedback
- **Be open** to suggestions and criticism
- **Update your PR** based on feedback
- **Re-request review** after making changes
- **Be patient** - reviews take time

### Code Style Guidelines

#### Python Code

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 88 characters (Black default)
- Use Google-style docstrings

```python
def calculate_threshold(
    base: float,
    step: int,
    max_steps: int,
    curve_type: str = "linear"
) -> float:
    """Calculate dynamic threshold for given step.
    
    Args:
        base: Base threshold value.
        step: Current generation step.
        max_steps: Maximum number of steps.
        curve_type: Type of curve ('linear', 'bezier', 'relu').
        
    Returns:
        Calculated threshold value.
        
    Raises:
        ValueError: If curve_type is not supported.
    """
    # Implementation
    pass
```

#### Frontend Code

- Use TypeScript for new components
- Follow Svelte conventions
- Use consistent naming (camelCase for variables, PascalCase for components)
- Add JSDoc comments for complex functions

```typescript
/**
 * Format generation output for display
 * @param raw - Raw output with bracket notation
 * @returns Formatted text for display
 */
export function formatOutput(raw: string): string {
  // Implementation
}
```

### Testing Guidelines

#### Writing Tests

1. **Test file naming**: `test_<module_name>.py`
2. **Test class naming**: `Test<ClassName>`
3. **Test method naming**: `test_<what_is_being_tested>`
4. **Use descriptive names** that explain what's being tested

```python
class TestTokenSelector:
    def test_selects_tokens_above_threshold(self):
        """Test that tokens above threshold are selected."""
        # Test implementation
        
    def test_handles_empty_input_gracefully(self):
        """Test graceful handling of empty input."""
        # Test implementation
```

#### Test Coverage

- Aim for >80% code coverage
- Focus on critical paths and edge cases
- Test both success and failure scenarios
- Include integration tests for complex features

### Documentation Standards

1. **Code documentation**:
   - All public APIs must have docstrings
   - Include examples in docstrings when helpful
   - Keep documentation up-to-date with code

2. **User documentation**:
   - Update relevant docs in `docs/`
   - Include examples for new features
   - Keep language clear and concise
   - Test all examples before submitting

3. **API documentation**:
   - Update OpenAPI schemas
   - Include request/response examples
   - Document error responses

## Development Workflow

### Daily Development

1. **Sync with upstream**:
   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```

2. **Create/update feature branch**:
   ```bash
   git checkout -b feature/new-feature
   # Or update existing
   git rebase main
   ```

3. **Make changes and test**
4. **Commit with clear messages**
5. **Push and create/update PR**

### Long-Running Features

For features taking multiple days:

1. **Regularly sync with main**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Keep PR updated** with progress
3. **Mark as draft** if not ready for review
4. **Request early feedback** if needed

## Community

### Getting Help

- **GitHub Discussions**: Ask questions and share ideas
- **Issue Tracker**: Report bugs and request features
- **Pull Requests**: Submit code contributions

### Recognition

Contributors are recognized in:
- The project README
- Release notes
- GitHub contributors page

## Release Process

### Version Numbering

We follow Semantic Versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist

1. [ ] All tests passing
2. [ ] Documentation updated
3. [ ] CHANGELOG.md updated
4. [ ] Version bumped
5. [ ] Release notes prepared

## Additional Resources

- [Development Guide](docs/development.md)
- [Architecture Documentation](docs/architecture.md)
- [API Reference](docs/api-reference.md)
- [Python Style Guide (PEP 8)](https://www.python.org/dev/peps/pep-0008/)
- [Conventional Commits](https://www.conventionalcommits.org/)

## Questions?

If you have questions not covered here:

1. Check existing documentation
2. Search closed issues
3. Ask in GitHub Discussions
4. Create a new issue with the question label

Thank you for contributing to TEMPO! Your efforts help make this project better for everyone.