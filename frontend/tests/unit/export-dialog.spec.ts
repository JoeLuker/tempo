import { test, expect } from '@playwright/test';

test.describe('Export Dialog', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    
    // Generate some test data
    await page.locator('textarea#prompt').fill('Test export generation');
    await page.locator('button:has-text("Generate")').click();
    await page.waitForTimeout(2000); // Wait for generation
    
    // Open export dialog
    await page.locator('button:has-text("Export")').click();
  });

  test('should open export dialog', async ({ page }) => {
    const dialog = page.locator('[role="dialog"]');
    await expect(dialog).toBeVisible();
    await expect(dialog).toContainText('Export Generation Results');
  });

  test('should have export options', async ({ page }) => {
    // Quick actions
    await expect(page.locator('button:has-text("Copy Text")')).toBeVisible();
    
    // Export formats
    await expect(page.locator('button:has-text("JSON (Complete Data)")')).toBeVisible();
    await expect(page.locator('button:has-text("Markdown Report")')).toBeVisible();
    await expect(page.locator('button:has-text("CSV (Token Data)")')).toBeVisible();
  });

  test('should copy text to clipboard', async ({ page, context }) => {
    // Grant clipboard permissions
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    
    const copyButton = page.locator('button:has-text("Copy Text")');
    await copyButton.click();
    
    // Button should show feedback
    await expect(copyButton).toContainText('Copied!');
    
    // Should revert after timeout
    await page.waitForTimeout(2500);
    await expect(copyButton).toContainText('Copy Text');
  });

  test('should show export preview', async ({ page }) => {
    const preview = page.locator('.export-preview');
    await expect(preview).toBeVisible();
    
    // Should show stats
    await expect(preview).toContainText('Text Length:');
    await expect(preview).toContainText('Total Steps:');
    await expect(preview).toContainText('Generation Time:');
  });

  test('should download JSON export', async ({ page }) => {
    // Set up download promise before clicking
    const downloadPromise = page.waitForEvent('download');
    
    await page.locator('button:has-text("JSON (Complete Data)")').click();
    
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/tempo-generation-\d+\.json/);
  });

  test('should download Markdown export', async ({ page }) => {
    const downloadPromise = page.waitForEvent('download');
    
    await page.locator('button:has-text("Markdown Report")').click();
    
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/tempo-report-\d+\.md/);
  });

  test('should download CSV export', async ({ page }) => {
    const downloadPromise = page.waitForEvent('download');
    
    await page.locator('button:has-text("CSV (Token Data)")').click();
    
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/tempo-tokens-\d+\.csv/);
  });

  test('should close dialog after export', async ({ page }) => {
    const dialog = page.locator('[role="dialog"]');
    
    // Export and check dialog closes
    const downloadPromise = page.waitForEvent('download');
    await page.locator('button:has-text("JSON (Complete Data)")').click();
    await downloadPromise;
    
    await expect(dialog).not.toBeVisible();
  });
});