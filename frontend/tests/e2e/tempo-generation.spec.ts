import { test, expect } from '@playwright/test';
import { mockAPIResponse, mockAPIError, mockSuccessResponse } from '../helpers/api-mock';

test.describe('TEMPO Generation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load the main page with correct title', async ({ page }) => {
    await expect(page).toHaveTitle(/TEMPO/);
    await expect(page.locator('h1')).toContainText('TEMPO Visualizer');
  });

  test('should display error when submitting empty prompt', async ({ page }) => {
    // Click generate without entering prompt
    await page.click('button:has-text("Generate")');
    
    // Check for error message
    await expect(page.locator('.text-red-500')).toContainText('Please enter a prompt');
  });

  test('should generate text with valid prompt', async ({ page }) => {
    // Mock successful API response
    await mockAPIResponse(page, mockSuccessResponse);

    // Enter prompt
    await page.fill('textarea[placeholder*="Enter your prompt"]', 'Once upon a time');
    
    // Click generate
    await page.click('button:has-text("Generate")');
    
    // Wait for generation to complete
    await expect(page.locator('[data-testid="generated-text"]')).toBeVisible();
    
    // Check generated text appears
    await expect(page.locator('[data-testid="generated-text"]')).toContainText('Once upon a time in a land far away');
    
    // Check visualization appears
    await expect(page.locator('.chart-container svg')).toBeVisible();
  });

  test('should handle API errors gracefully', async ({ page }) => {
    // Mock API error
    await mockAPIError(page, 500, 'Model loading failed');

    // Enter prompt and generate
    await page.fill('textarea[placeholder*="Enter your prompt"]', 'Test prompt');
    await page.click('button:has-text("Generate")');
    
    // Check error is displayed
    await expect(page.locator('.text-red-500')).toContainText('Model loading failed');
  });

  test('should update settings when sliders are changed', async ({ page }) => {
    // Open settings tab
    await page.click('button[role="tab"]:has-text("Settings")');
    
    // Find max tokens slider and change it
    const maxTokensSlider = page.locator('input[type="range"]').first();
    await maxTokensSlider.fill('100');
    
    // Verify the value display updates
    await expect(page.locator('text=Max Tokens: 100')).toBeVisible();
  });

  test('should toggle theme between light and dark', async ({ page }) => {
    // Check initial theme
    const htmlElement = page.locator('html');
    const initialTheme = await htmlElement.getAttribute('class');
    
    // Click theme toggle
    await page.click('[data-testid="theme-toggle"]');
    
    // Check theme changed
    const newTheme = await htmlElement.getAttribute('class');
    expect(newTheme).not.toBe(initialTheme);
  });

  test('should show keyboard shortcuts dialog', async ({ page }) => {
    // Press keyboard shortcut for help
    await page.keyboard.press('Control+/');
    
    // Check dialog appears
    await expect(page.locator('[role="dialog"]')).toBeVisible();
    await expect(page.locator('[role="dialog"]')).toContainText('Keyboard Shortcuts');
    
    // Close dialog
    await page.keyboard.press('Escape');
    await expect(page.locator('[role="dialog"]')).not.toBeVisible();
  });

  test('should switch between API versions', async ({ page }) => {
    // Open settings
    await page.click('button[role="tab"]:has-text("Settings")');
    
    // Find API version toggle
    const apiVersionToggle = page.locator('[data-testid="api-version-toggle"]');
    await apiVersionToggle.click();
    
    // Verify UI updates for v1 API
    await expect(page.locator('text=MCTS Settings')).toBeVisible();
  });

  test('should copy generated text to clipboard', async ({ page, context }) => {
    // Grant clipboard permissions
    await context.grantPermissions(['clipboard-write']);
    
    // Mock API response
    await mockAPIResponse(page, mockSuccessResponse);

    // Generate text
    await page.fill('textarea[placeholder*="Enter your prompt"]', 'Once upon a time');
    await page.click('button:has-text("Generate")');
    
    // Wait for generation
    await expect(page.locator('[data-testid="generated-text"]')).toBeVisible();
    
    // Click copy button
    await page.click('button[data-testid="copy-button"]');
    
    // Verify copy notification appears
    await expect(page.locator('text=Copied to clipboard')).toBeVisible();
  });

  test('should apply preset configurations', async ({ page }) => {
    // Open settings
    await page.click('button[role="tab"]:has-text("Settings")');
    
    // Select a preset
    await page.selectOption('select[data-testid="preset-selector"]', 'creative');
    
    // Verify settings updated
    await expect(page.locator('text=Selection Threshold: 0.15')).toBeVisible();
  });
});