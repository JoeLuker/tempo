import { test, expect } from '@playwright/test';

test.describe('TEMPO Main Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display the header with title and subtitle', async ({ page }) => {
    await expect(page.locator('h1.title')).toHaveText('TEMPO');
    await expect(page.locator('p.subtitle')).toHaveText('Threshold-Enabled Multipath Parallel Output');
  });

  test('should have history button in header', async ({ page }) => {
    const historyButton = page.locator('button:has-text("History")');
    await expect(historyButton).toBeVisible();
  });

  test('should have prompt textarea', async ({ page }) => {
    const promptTextarea = page.locator('textarea#prompt');
    await expect(promptTextarea).toBeVisible();
    await expect(promptTextarea).toHaveAttribute('placeholder', 'Enter your prompt here...');
  });

  test('should have generate button', async ({ page }) => {
    const generateButton = page.locator('button:has-text("Generate")');
    await expect(generateButton).toBeVisible();
  });

  test('should have preset buttons', async ({ page }) => {
    await expect(page.locator('button:has-text("Standard")')).toBeVisible();
    await expect(page.locator('button:has-text("Creative")')).toBeVisible();
    await expect(page.locator('button:has-text("TEMPO Explorer")')).toBeVisible();
  });

  test('should toggle TEMPO generation', async ({ page }) => {
    const tempoSwitch = page.locator('.tempo-toggle switch');
    const tempoDescription = page.locator('.toggle-description');
    
    // Check initial state
    await expect(tempoDescription).toContainText('Enable TEMPO');
    
    // Toggle TEMPO on
    await tempoSwitch.click();
    await expect(tempoDescription).toContainText('TEMPO is active');
  });

  test('should show/hide history panel', async ({ page }) => {
    const historyButton = page.locator('button:has-text("History")');
    const historyPanel = page.locator('.history-column');
    
    // Initially hidden
    await expect(historyPanel).not.toBeVisible();
    
    // Show history
    await historyButton.click();
    await expect(historyPanel).toBeVisible();
    
    // Hide history
    await historyButton.click();
    await expect(historyPanel).not.toBeVisible();
  });

  test('should validate empty prompt', async ({ page }) => {
    const generateButton = page.locator('button:has-text("Generate")');
    
    // Try to generate with empty prompt
    await generateButton.click();
    
    // Should show error
    await expect(page.locator('text="Please enter a prompt"')).toBeVisible();
  });

  test('should apply presets', async ({ page }) => {
    const creativeButton = page.locator('button:has-text("Creative")');
    
    // Click creative preset
    await creativeButton.click();
    
    // Check if preset is applied (button becomes active)
    await expect(creativeButton).toHaveClass(/active/);
  });

  test('should have output tabs', async ({ page }) => {
    // Check for tab triggers
    await expect(page.locator('[role="tab"]:has-text("Interactive")')).toBeVisible();
    await expect(page.locator('[role="tab"]:has-text("Plain Text")')).toBeVisible();
    await expect(page.locator('[role="tab"]:has-text("Stats")')).toBeVisible();
    await expect(page.locator('[role="tab"]:has-text("Threshold")')).toBeVisible();
    await expect(page.locator('[role="tab"]:has-text("Analysis")')).toBeVisible();
    await expect(page.locator('[role="tab"]:has-text("Token Flow")')).toBeVisible();
  });

  test('should handle keyboard shortcuts', async ({ page }) => {
    const promptTextarea = page.locator('textarea#prompt');
    
    // Focus prompt with '/'
    await page.keyboard.press('/');
    await expect(promptTextarea).toBeFocused();
    
    // Clear prompt with Escape
    await promptTextarea.fill('Test prompt');
    await page.keyboard.press('Escape');
    await expect(promptTextarea).toHaveValue('');
    
    // Toggle history with Ctrl+H
    const historyPanel = page.locator('.history-column');
    await page.keyboard.press('Control+h');
    await expect(historyPanel).toBeVisible();
    await page.keyboard.press('Control+h');
    await expect(historyPanel).not.toBeVisible();
  });

  test('should show collapsible settings sections', async ({ page }) => {
    // Enable TEMPO first
    const tempoSwitch = page.locator('.tempo-toggle switch');
    await tempoSwitch.click();
    
    // Check settings sections
    await expect(page.locator('text="Core Settings"')).toBeVisible();
    await expect(page.locator('text="Retroactive Removal"')).toBeVisible();
    await expect(page.locator('text="Dynamic Threshold"')).toBeVisible();
    await expect(page.locator('text="MCTS Search"')).toBeVisible();
    await expect(page.locator('text="Advanced Settings"')).toBeVisible();
  });
});

test.describe('TEMPO Settings', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Enable TEMPO
    const tempoSwitch = page.locator('.tempo-toggle switch');
    await tempoSwitch.click();
  });

  test('should adjust selection threshold slider', async ({ page }) => {
    const slider = page.locator('slider[id="selection-threshold"]');
    const value = page.locator('.setting-value').first();
    
    // Check initial value
    await expect(value).toContainText('0.10');
    
    // Move slider (this is simplified, actual implementation may vary)
    await slider.click({ position: { x: 100, y: 10 } });
    
    // Value should change
    await expect(value).not.toContainText('0.10');
  });

  test('should toggle retroactive removal', async ({ page }) => {
    const retroactiveSwitch = page.locator('switch[id="use-retroactive-removal"]');
    const attentionThresholdSection = page.locator('label:has-text("Attention Threshold")');
    
    // Initially off
    await expect(attentionThresholdSection).not.toBeVisible();
    
    // Turn on
    await retroactiveSwitch.click();
    await expect(attentionThresholdSection).toBeVisible();
  });

  test('should show dynamic threshold preview', async ({ page }) => {
    const dynamicSwitch = page.locator('switch[id="dynamic-threshold"]');
    
    // Turn on dynamic threshold
    await dynamicSwitch.click();
    
    // Should show threshold preview
    await expect(page.locator('.threshold-preview')).toBeVisible();
  });
});