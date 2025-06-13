import { test, expect } from '@playwright/test';

test.describe('Keyboard Shortcuts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should generate with Ctrl+Enter', async ({ page }) => {
    const promptTextarea = page.locator('textarea#prompt');
    await promptTextarea.fill('Test keyboard generation');
    
    // Mock API response
    await page.route('/api/v2/generate', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          clean_text: 'Generated text',
          generated_text: 'Generated text',
          steps: []
        })
      });
    });
    
    // Press Ctrl+Enter
    await page.keyboard.press('Control+Enter');
    
    // Should trigger generation
    await expect(page.locator('text="Generating with TEMPO..."')).toBeVisible();
  });

  test('should clear prompt with Escape', async ({ page }) => {
    const promptTextarea = page.locator('textarea#prompt');
    
    await promptTextarea.fill('Test prompt to clear');
    await promptTextarea.press('Escape');
    
    await expect(promptTextarea).toHaveValue('');
    await expect(promptTextarea).toBeFocused();
  });

  test('should toggle history with Ctrl+H', async ({ page }) => {
    const historyPanel = page.locator('.history-column');
    
    // Initially hidden
    await expect(historyPanel).not.toBeVisible();
    
    // Show with Ctrl+H
    await page.keyboard.press('Control+h');
    await expect(historyPanel).toBeVisible();
    
    // Hide with Ctrl+H
    await page.keyboard.press('Control+h');
    await expect(historyPanel).not.toBeVisible();
  });

  test('should focus prompt with /', async ({ page }) => {
    const promptTextarea = page.locator('textarea#prompt');
    
    // Click somewhere else first
    await page.locator('h1').click();
    await expect(promptTextarea).not.toBeFocused();
    
    // Press /
    await page.keyboard.press('/');
    
    // Should focus prompt
    await expect(promptTextarea).toBeFocused();
    
    // Should not type the / character
    await page.waitForTimeout(100);
    await expect(promptTextarea).toHaveValue('');
  });

  test('should open export with Ctrl+E', async ({ page }) => {
    // Generate some data first
    await page.locator('textarea#prompt').fill('Test');
    
    // Mock API response
    await page.route('/api/v2/generate', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          clean_text: 'Generated text',
          generated_text: 'Generated text',
          steps: []
        })
      });
    });
    
    await page.locator('button:has-text("Generate")').click();
    await page.waitForTimeout(500);
    
    // Press Ctrl+E
    await page.keyboard.press('Control+e');
    
    // Should open export dialog
    await expect(page.locator('[role="dialog"]')).toBeVisible();
    await expect(page.locator('text="Export Generation Results"')).toBeVisible();
  });

  test('should reset settings with Alt+R', async ({ page }) => {
    // Change a setting first
    const tempoSwitch = page.locator('.tempo-toggle switch');
    await tempoSwitch.click();
    await expect(page.locator('.toggle-description')).toContainText('TEMPO is active');
    
    // Press Alt+R
    await page.keyboard.press('Alt+r');
    
    // Should reset to default
    await expect(page.locator('.toggle-description')).toContainText('Enable TEMPO');
  });

  test('should toggle theme with Alt+T', async ({ page }) => {
    const html = page.locator('html');
    
    // Check initial state
    const initialHasDark = await html.evaluate(el => el.classList.contains('dark'));
    
    // Press Alt+T
    await page.keyboard.press('Alt+t');
    
    // Should toggle theme
    const afterHasDark = await html.evaluate(el => el.classList.contains('dark'));
    expect(afterHasDark).toBe(!initialHasDark);
  });

  test('should show shortcuts help with Alt+?', async ({ page }) => {
    // Set up dialog handler
    let dialogShown = false;
    page.on('dialog', async dialog => {
      expect(dialog.message()).toContain('Keyboard Shortcuts');
      expect(dialog.message()).toContain('Ctrl+Enter - Generate text');
      dialogShown = true;
      await dialog.accept();
    });
    
    // Press Alt+?
    await page.keyboard.press('Alt+?');
    
    // Wait for dialog
    await page.waitForTimeout(500);
    expect(dialogShown).toBe(true);
  });

  test('should not trigger shortcuts in input fields', async ({ page }) => {
    const promptTextarea = page.locator('textarea#prompt');
    await promptTextarea.focus();
    
    // Type 'h' in textarea - should not toggle history
    await promptTextarea.press('h');
    await expect(promptTextarea).toHaveValue('h');
    
    const historyPanel = page.locator('.history-column');
    await expect(historyPanel).not.toBeVisible();
  });
});