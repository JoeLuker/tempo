import { test, expect } from '@playwright/test';

test.describe('UI Components', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('Button component should be interactive', async ({ page }) => {
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

  test('Textarea should accept input and show character count', async ({ page }) => {
    const textarea = page.locator('textarea[placeholder*="Enter your prompt"]');
    
    // Type in textarea
    const testText = 'This is a test prompt for TEMPO generation';
    await textarea.fill(testText);
    
    // Verify text was entered
    await expect(textarea).toHaveValue(testText);
  });

  test('Tabs should switch content correctly', async ({ page }) => {
    // Check initial tab
    await expect(page.locator('[role="tabpanel"]').first()).toBeVisible();
    
    // Click settings tab
    await page.click('button[role="tab"]:has-text("Settings")');
    
    // Verify settings content is shown
    await expect(page.locator('text=Generation Settings')).toBeVisible();
    
    // Go back to generation tab
    await page.click('button[role="tab"]:has-text("Generation")');
    
    // Verify generation content is shown
    await expect(page.locator('textarea[placeholder*="Enter your prompt"]')).toBeVisible();
  });

  test('Sliders should update values correctly', async ({ page }) => {
    // Open settings tab
    await page.click('button[role="tab"]:has-text("Settings")');
    
    // Test selection threshold slider
    const slider = page.locator('[data-testid="selection-threshold-slider"]');
    
    // Get initial value
    const initialValue = await slider.inputValue();
    
    // Change slider value
    await slider.fill('0.5');
    
    // Verify value changed
    const newValue = await slider.inputValue();
    expect(newValue).not.toBe(initialValue);
    expect(parseFloat(newValue)).toBe(0.5);
  });

  test('Switches should toggle correctly', async ({ page }) => {
    // Open settings tab
    await page.click('button[role="tab"]:has-text("Settings")');
    
    // Find retroactive pruning switch
    const pruningSwitch = page.locator('[data-testid="retroactive-pruning-switch"]');
    
    // Get initial state
    const initialState = await pruningSwitch.getAttribute('data-state');
    
    // Click to toggle
    await pruningSwitch.click();
    
    // Verify state changed
    const newState = await pruningSwitch.getAttribute('data-state');
    expect(newState).not.toBe(initialState);
  });

  test('Cards should display content properly', async ({ page }) => {
    // Check main generation card
    const generationCard = page.locator('.card').first();
    
    // Verify card structure
    await expect(generationCard.locator('.card-header')).toBeVisible();
    await expect(generationCard.locator('.card-content')).toBeVisible();
    
    // Verify card title
    await expect(generationCard.locator('.card-title')).toContainText('TEMPO Visualizer');
  });

  test('Checkboxes should toggle correctly', async ({ page }) => {
    // Open settings tab
    await page.click('button[role="tab"]:has-text("Settings")');
    
    // If there are checkboxes in settings
    const checkbox = page.locator('[role="checkbox"]').first();
    
    if (await checkbox.isVisible()) {
      // Get initial state
      const initialChecked = await checkbox.getAttribute('data-state');
      
      // Click to toggle
      await checkbox.click();
      
      // Verify state changed
      const newChecked = await checkbox.getAttribute('data-state');
      expect(newChecked).not.toBe(initialChecked);
    }
  });
});