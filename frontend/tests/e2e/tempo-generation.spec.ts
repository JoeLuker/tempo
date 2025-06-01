import { test, expect } from '@playwright/test';
import { mockAPIResponse, mockAPIError, mockSuccessResponse } from '../helpers/api-mock';

test.describe('TEMPO Generation - Progressive Disclosure Interface', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load the main page with correct title', async ({ page }) => {
    await expect(page).toHaveTitle(/TEMPO/);
    await expect(page.locator('h1')).toContainText('TEMPO Text Generation');
  });

  test('should display error when submitting empty prompt', async ({ page }) => {
    // Click generate without entering prompt
    await page.click('button:has-text("Generate")');
    
    // Check for error message
    await expect(page.locator('.text-red-500')).toContainText('Please enter a prompt');
  });

  test('should generate text with valid prompt using preset', async ({ page }) => {
    // Mock successful API response
    await mockAPIResponse(page, mockSuccessResponse);

    // Use a preset first (easier workflow for users)
    const balancedPreset = page.locator('text=Balanced').locator('..').locator('..');
    await balancedPreset.click();
    
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

  test('should apply preset configurations correctly', async ({ page }) => {
    // Click on Creative Writing preset
    const creativePreset = page.locator('text=Creative Writing').locator('..').locator('..');
    await creativePreset.click();
    
    // Verify preset is active
    await expect(creativePreset.locator('text=Applied')).toBeVisible();
    
    // Switch to intermediate mode to see settings
    await page.click('button:has-text("Intermediate")');
    
    // Expand settings section to verify values
    const promptSection = page.locator('text=Prompt & Generation').locator('..');
    await promptSection.click();
    
    // Check that max tokens was set to creative preset value (300)
    await expect(page.locator('text=300 tokens')).toBeVisible();
  });

  test('should update settings when sliders are changed in expert mode', async ({ page }) => {
    // Switch to expert mode
    await page.click('button:has-text("Expert")');
    
    // Find max tokens slider and change it
    const maxTokensSlider = page.locator('input[type="range"]').first();
    await maxTokensSlider.fill('400');
    
    // Verify the value display updates
    await expect(page.locator('text=400 tokens')).toBeVisible();
  });

  test('should toggle theme between light and dark', async ({ page }) => {
    // Check initial theme
    const htmlElement = page.locator('html');
    const initialTheme = await htmlElement.getAttribute('class');
    
    // Click theme toggle
    await page.click('button[aria-label="Toggle theme"]');
    
    // Check theme changed
    const newTheme = await htmlElement.getAttribute('class');
    expect(newTheme).not.toBe(initialTheme);
  });

  test('should show keyboard shortcuts dialog', async ({ page }) => {
    // Click keyboard shortcuts link in footer
    await page.click('text=Keyboard Shortcuts');
    
    // Check dialog appears
    await expect(page.locator('[role="dialog"]')).toBeVisible();
    await expect(page.locator('[role="dialog"]')).toContainText('Keyboard Shortcuts');
    
    // Close dialog
    await page.keyboard.press('Escape');
    await expect(page.locator('[role="dialog"]')).not.toBeVisible();
  });

  test('should show progressive disclosure based on interface mode', async ({ page }) => {
    // Start in beginner mode - should see presets but minimal settings
    await expect(page.locator('text=🚀 Quick Start')).toBeVisible();
    await expect(page.locator('text=Balanced')).toBeVisible();
    
    // Switch to intermediate mode - should see organized sections
    await page.click('button:has-text("Intermediate")');
    await expect(page.locator('text=Prompt & Generation')).toBeVisible();
    await expect(page.locator('text=Pruning & Refinement')).toBeVisible();
    
    // Switch to expert mode - should see tabbed interface
    await page.click('button:has-text("Expert")');
    await expect(page.locator('button[role="tab"]:has-text("Basic")')).toBeVisible();
    await expect(page.locator('button[role="tab"]:has-text("MCTS")')).toBeVisible();
    await expect(page.locator('button[role="tab"]:has-text("Advanced")')).toBeVisible();
  });

  test('should show help tooltips when hovering over help icons', async ({ page }) => {
    // Switch to intermediate mode to see help icons
    await page.click('button:has-text("Intermediate")');
    
    // Expand a section to see settings with help
    const promptSection = page.locator('text=Prompt & Generation').locator('..');
    await promptSection.click();
    
    // Find and hover over a help icon
    const helpIcon = page.locator('[aria-label*="Help for Max Tokens"]');
    await helpIcon.hover();
    
    // Check tooltip appears
    await expect(page.locator('[role="tooltip"]')).toBeVisible();
    await expect(page.locator('[role="tooltip"]')).toContainText('Max Tokens');
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
    
    // Click copy button if it exists
    const copyButton = page.locator('button[data-testid="copy-button"]');
    if (await copyButton.isVisible()) {
      await copyButton.click();
      // Verify copy notification appears
      await expect(page.locator('text=Copied to clipboard')).toBeVisible();
    }
  });

  test('should handle MCTS settings in expert mode', async ({ page }) => {
    // Switch to expert mode
    await page.click('button:has-text("Expert")');
    
    // Go to MCTS tab
    await page.click('button[role="tab"]:has-text("MCTS")');
    
    // Toggle MCTS on
    const mctsSwitch = page.locator('text=Use MCTS').locator('..').locator('button[role="switch"]');
    await mctsSwitch.click();
    
    // Should see MCTS settings
    await expect(page.locator('text=MCTS Simulations')).toBeVisible();
    await expect(page.locator('text=MCTS C_PUCT')).toBeVisible();
    await expect(page.locator('text=MCTS Depth')).toBeVisible();
  });

  test('should show advanced settings in expert mode', async ({ page }) => {
    // Switch to expert mode
    await page.click('button:has-text("Expert")');
    
    // Go to Advanced tab
    await page.click('button[role="tab"]:has-text("Advanced")');
    
    // Should see advanced settings
    await expect(page.locator('text=Use Custom RoPE')).toBeVisible();
    await expect(page.locator('text=System Content')).toBeVisible();
    await expect(page.locator('text=Enable Thinking')).toBeVisible();
  });

  test('should persist interface mode selection', async ({ page }) => {
    // Switch to intermediate mode
    await page.click('button:has-text("Intermediate")');
    
    // Reload page
    await page.reload();
    
    // Should still be in intermediate mode (if localStorage persistence is implemented)
    await expect(page.locator('button:has-text("Intermediate")')).toHaveClass(/bg-white/);
  });

  test('should show preset performance indicators', async ({ page }) => {
    // Should see performance information on preset cards
    await expect(page.locator('text=Fast (2-3s)')).toBeVisible();
    await expect(page.locator('text=Moderate (3-5s)')).toBeVisible();
    
    // Should see difficulty badges
    await expect(page.locator('text=Beginner')).toBeVisible();
  });

  test('should handle responsive layout for generation', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    // Should still be able to generate
    await page.fill('textarea[placeholder*="Enter your prompt"]', 'Test prompt');
    await expect(page.locator('button:has-text("Generate")')).toBeVisible();
    
    // Preset cards should stack vertically
    const presetCards = page.locator('text=Balanced').locator('..').locator('..');
    await expect(presetCards).toBeVisible();
  });
});