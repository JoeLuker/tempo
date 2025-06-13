# TEMPO Frontend E2E Tests

This directory contains end-to-end tests for the TEMPO visualizer frontend using Playwright.

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Install Playwright browsers:
   ```bash
   npm run test:install
   ```

## Running Tests

### Run all tests
```bash
npm test
```

### Run tests with UI mode (recommended for development)
```bash
npm run test:ui
```

### Run tests in headed mode (see browser)
```bash
npm run test:headed
```

### Debug tests
```bash
npm run test:debug
```

### Generate new tests using recorder
```bash
npm run test:codegen
```

### View test report
```bash
npm run test:report
```

## Test Structure

```
tests/
├── e2e/                      # End-to-end tests
│   ├── tempo-generation.spec.ts    # Main generation flow tests
│   ├── components.spec.ts          # UI component tests
│   ├── simple.spec.ts              # Basic functionality tests
│   └── visualization.spec.ts       # Chart visualization tests
├── unit/                     # Unit/Integration tests (NEW)
│   ├── main-page.spec.ts           # Main page interactions
│   ├── generation-history.spec.ts  # History panel functionality
│   ├── export-dialog.spec.ts       # Export features
│   ├── keyboard-shortcuts.spec.ts  # Keyboard navigation
│   ├── token-stats-dashboard.spec.ts # Statistics visualization
│   └── interactive-output.spec.ts  # Interactive token display
├── fixtures/                 # Test data and fixtures
│   └── test-data.ts         # Reusable test data
└── helpers/                  # Test utilities
    └── api-mock.ts          # API mocking helpers
```

## Writing Tests

### Basic Test Structure
```typescript
import { test, expect } from '@playwright/test';

test.describe('Feature Name', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should do something', async ({ page }) => {
    // Arrange
    await page.fill('input', 'value');
    
    // Act
    await page.click('button');
    
    // Assert
    await expect(page.locator('.result')).toBeVisible();
  });
});
```

### Using API Mocks
```typescript
import { mockAPIResponse, mockSuccessResponse } from '../helpers/api-mock';

test('should handle API response', async ({ page }) => {
  await mockAPIResponse(page, mockSuccessResponse);
  // ... rest of test
});
```

## New Unit Tests

The unit test suite provides comprehensive coverage of individual components and features:

### Main Page (`main-page.spec.ts`)
- Header elements and navigation
- Prompt input validation
- Settings panel interactions
- Preset applications
- Output tab management

### Generation History (`generation-history.spec.ts`)
- History panel display and search
- Loading previous generations
- History management (delete, clear)
- Local storage persistence
- Relative time display

### Export Dialog (`export-dialog.spec.ts`)
- Export dialog functionality
- Copy to clipboard feature
- Multiple export formats (JSON, Markdown, CSV)
- Export preview statistics
- File download handling

### Keyboard Shortcuts (`keyboard-shortcuts.spec.ts`)
- All keyboard navigation features
- Context-aware shortcut handling
- Input field exception handling
- Theme toggling and settings reset

### Token Stats Dashboard (`token-stats-dashboard.spec.ts`)
- Statistics visualization
- Chart rendering (D3.js)
- Position heatmap display
- Token frequency analysis
- Empty state handling

### Interactive Output (`interactive-output.spec.ts`)
- Token highlighting and hover effects
- Alternative token popups
- Retroactive removal indicators
- Position badges

## Best Practices

1. **Use data-testid attributes** for reliable element selection
2. **Mock API responses** to avoid dependencies on backend
3. **Keep tests independent** - each test should be able to run in isolation
4. **Use descriptive test names** that explain what is being tested
5. **Follow AAA pattern** - Arrange, Act, Assert
6. **Wait for elements** before interacting with them
7. **Use page objects** for complex pages (if needed)
8. **Test user behavior** not implementation details
9. **Handle async operations** properly with appropriate waits
10. **Clean up after tests** (close dialogs, reset state)

## CI/CD Integration

Tests run automatically on:
- Push to main/develop branches
- Pull requests affecting frontend code

The GitHub Actions workflow:
- Runs tests on multiple OS (Ubuntu, macOS, Windows)
- Tests with multiple Node.js versions (18, 20)
- Uploads test reports and screenshots on failure

## Debugging Failed Tests

1. **Local debugging:**
   - Run `npm run test:debug` to step through tests
   - Use `await page.pause()` to pause execution

2. **CI debugging:**
   - Check uploaded artifacts for screenshots
   - View test report in GitHub Actions

3. **Common issues:**
   - Timing issues: Use proper waits (`waitForSelector`, `waitForLoadState`)
   - Flaky tests: Ensure proper test isolation
   - Environment differences: Check CI vs local settings