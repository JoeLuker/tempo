import { test, expect } from '@playwright/test';

test.describe('UI Components - Progressive Disclosure Interface', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('Interface mode toggle should work correctly', async ({ page }) => {
    // Should start in beginner mode by default
    await expect(page.locator('button:has-text("Beginner")')).toHaveClass(/bg-white/);
    
    // Switch to intermediate mode
    await page.click('button:has-text("Intermediate")');
    await expect(page.locator('button:has-text("Intermediate")')).toHaveClass(/bg-white/);
    
    // Switch to expert mode
    await page.click('button:has-text("Expert")');
    await expect(page.locator('button:has-text("Expert")')).toHaveClass(/bg-white/);
    
    // Expert mode should show tabbed interface
    await expect(page.locator('button[role="tab"]:has-text("Basic")')).toBeVisible();
  });

  test('Enhanced preset cards should be visible and interactive in beginner mode', async ({ page }) => {
    // Should see preset cards in beginner mode
    await expect(page.locator('text=Quick Start')).toBeVisible();
    
    // Should see enhanced preset cards
    const presetCards = page.locator('[data-testid="preset-card"]');
    await expect(presetCards).toHaveCount(3); // Beginner presets only
    
    // Test clicking a preset card
    const balancedPreset = page.locator('text=Balanced').locator('..').locator('..');
    await balancedPreset.click();
    
    // Should show as active
    await expect(balancedPreset.locator('text=Applied')).toBeVisible();
  });

  test('Setting sections should be collapsible', async ({ page }) => {
    // Switch to intermediate mode to see sections
    await page.click('button:has-text("Intermediate")');
    
    // Should see collapsible sections
    await expect(page.locator('text=Prompt & Generation')).toBeVisible();
    await expect(page.locator('text=Pruning & Refinement')).toBeVisible();
    
    // Test expanding/collapsing a section
    const pruningSection = page.locator('text=Pruning & Refinement').locator('..');
    await pruningSection.click();
    
    // Section content should be visible after clicking
    await expect(page.locator('text=Use Retroactive Pruning')).toBeVisible();
  });

  test('Rich tooltips should appear on hover', async ({ page }) => {
    // Switch to intermediate mode
    await page.click('button:has-text("Intermediate")');
    
    // Find a help icon next to a setting
    const helpIcon = page.locator('[aria-label*="Help for"]').first();
    
    // Hover over help icon
    await helpIcon.hover();
    
    // Tooltip should appear
    await expect(page.locator('[role="tooltip"]')).toBeVisible();
    
    // Tooltip should contain helpful information
    await expect(page.locator('[role="tooltip"]')).toContainText(/description|example|threshold/i);
  });

  test('Generate button should be interactive', async ({ page }) => {
    // Test generate button
    const generateButton = page.locator('button:has-text("Generate")');
    
    // Check button is visible and enabled initially
    await expect(generateButton).toBeVisible();
    await expect(generateButton).toBeEnabled();
    
    // Check hover state
    await generateButton.hover();
    const buttonStyles = await generateButton.evaluate((el) => {
      return window.getComputedStyle(el);
    });
    expect(buttonStyles).toBeTruthy();
  });

  test('Textarea should accept input', async ({ page }) => {
    const textarea = page.locator('textarea[placeholder*="Enter your prompt"]');
    
    // Type in textarea
    const testText = 'This is a test prompt for TEMPO generation';
    await textarea.fill(testText);
    
    // Verify text was entered
    await expect(textarea).toHaveValue(testText);
  });

  test('Expert mode should preserve original tabbed interface', async ({ page }) => {
    // Switch to expert mode
    await page.click('button:has-text("Expert")');
    
    // Should see tabs
    await expect(page.locator('button[role="tab"]:has-text("Basic")')).toBeVisible();
    await expect(page.locator('button[role="tab"]:has-text("MCTS")')).toBeVisible();
    await expect(page.locator('button[role="tab"]:has-text("Advanced")')).toBeVisible();
    
    // Test tab switching
    await page.click('button[role="tab"]:has-text("MCTS")');
    await expect(page.locator('text=Use MCTS')).toBeVisible();
    
    // Go back to basic tab
    await page.click('button[role="tab"]:has-text("Basic")');
    await expect(page.locator('textarea[placeholder*="Enter your prompt"]')).toBeVisible();
  });

  test('Sliders should update values correctly', async ({ page }) => {
    // Switch to expert mode for easier access
    await page.click('button:has-text("Expert")');
    
    // Test max tokens slider
    const maxTokensSlider = page.locator('input[type="range"]').first();
    
    // Get initial value
    const initialValue = await maxTokensSlider.inputValue();
    
    // Change slider value
    await maxTokensSlider.fill('300');
    
    // Verify value changed
    const newValue = await maxTokensSlider.inputValue();
    expect(newValue).not.toBe(initialValue);
    expect(parseFloat(newValue)).toBe(300);
    
    // Verify display updated
    await expect(page.locator('text=300 tokens')).toBeVisible();
  });

  test('Switches should toggle correctly', async ({ page }) => {
    // Switch to expert mode
    await page.click('button:has-text("Expert")');
    
    // Find retroactive pruning switch
    const pruningSwitch = page.locator('text=Use Retroactive Pruning').locator('..').locator('button[role="switch"]');
    
    // Get initial state
    const initialState = await pruningSwitch.getAttribute('data-state');
    
    // Click to toggle
    await pruningSwitch.click();
    
    // Verify state changed
    const newState = await pruningSwitch.getAttribute('data-state');
    expect(newState).not.toBe(initialState);
  });

  test('Setting importance badges should be displayed', async ({ page }) => {
    // Switch to intermediate mode to see section headers
    await page.click('button:has-text("Intermediate")');
    
    // Should see importance badges in section headers
    await expect(page.locator('text=Essential')).toBeVisible();
    await expect(page.locator('text=Important')).toBeVisible();
  });

  test('Category icons should be displayed in sections', async ({ page }) => {
    // Switch to intermediate mode
    await page.click('button:has-text("Intermediate")');
    
    // Should see emoji icons in section headers
    await expect(page.locator('text=💬')).toBeVisible(); // Prompt & Generation
    await expect(page.locator('text=✂️')).toBeVisible(); // Pruning & Refinement
  });

  test('Interface should persist mode selection', async ({ page }) => {
    // Switch to intermediate mode
    await page.click('button:has-text("Intermediate")');
    
    // Reload page
    await page.reload();
    
    // Should still be in intermediate mode (if persistence is implemented)
    // Note: This depends on localStorage implementation
    await expect(page.locator('button:has-text("Intermediate")')).toHaveClass(/bg-white/);
  });

  test('Preset cards should show performance and difficulty information', async ({ page }) => {
    // Should see enhanced information on preset cards
    await expect(page.locator('text=Beginner')).toBeVisible(); // Difficulty badge
    await expect(page.locator('text=Fast')).toBeVisible(); // Performance info
    await expect(page.locator('text=Best for:')).toBeVisible(); // Use case info
  });

  test('Help content should be comprehensive', async ({ page }) => {
    // Switch to intermediate mode
    await page.click('button:has-text("Intermediate")');
    
    // Hover over a help icon
    const helpIcon = page.locator('[aria-label*="Help for Max Tokens"]');
    await helpIcon.hover();
    
    // Tooltip should contain rich content
    const tooltip = page.locator('[role="tooltip"]');
    await expect(tooltip).toBeVisible();
    await expect(tooltip).toContainText('Max Tokens'); // Title
    await expect(tooltip).toContainText('essential'); // Importance
    await expect(tooltip).toContainText('core'); // Category
    await expect(tooltip).toContainText(/\d+.*words/); // Example with word count
  });
});