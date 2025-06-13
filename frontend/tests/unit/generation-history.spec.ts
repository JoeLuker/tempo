import { test, expect } from '@playwright/test';

test.describe('Generation History', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    
    // Generate some test data first
    await page.locator('textarea#prompt').fill('Test generation 1');
    await page.locator('button:has-text("Generate")').click();
    await page.waitForTimeout(1000); // Wait for generation
    
    await page.locator('textarea#prompt').fill('Test generation 2');
    await page.locator('button:has-text("Generate")').click();
    await page.waitForTimeout(1000);
    
    // Open history panel
    await page.locator('button:has-text("History")').click();
  });

  test('should display generation history', async ({ page }) => {
    const historyPanel = page.locator('.generation-history');
    await expect(historyPanel).toBeVisible();
    
    // Should show item count
    await expect(page.locator('.item-count')).toContainText('2 items');
    
    // Should show history items
    const historyItems = page.locator('.history-item');
    await expect(historyItems).toHaveCount(2);
  });

  test('should search history', async ({ page }) => {
    const searchInput = page.locator('.search-input');
    await searchInput.fill('generation 1');
    
    // Should filter results
    const historyItems = page.locator('.history-item');
    await expect(historyItems).toHaveCount(1);
    await expect(historyItems.first()).toContainText('Test generation 1');
  });

  test('should load generation from history', async ({ page }) => {
    const firstItem = page.locator('.history-item').first();
    await firstItem.click();
    
    // Should show confirmation dialog
    await page.on('dialog', dialog => dialog.accept());
    
    // History should close
    const historyPanel = page.locator('.generation-history');
    await expect(historyPanel).not.toBeVisible();
    
    // Prompt should be restored
    const promptTextarea = page.locator('textarea#prompt');
    await expect(promptTextarea).toHaveValue(/Test generation/);
  });

  test('should delete history item', async ({ page }) => {
    const deleteButton = page.locator('.delete-button').first();
    
    // Hover to show delete button
    await page.locator('.history-item').first().hover();
    await deleteButton.click();
    
    // Confirm deletion
    await page.on('dialog', dialog => dialog.accept());
    
    // Should have one less item
    await expect(page.locator('.item-count')).toContainText('1 items');
  });

  test('should clear all history', async ({ page }) => {
    const clearButton = page.locator('button:has-text("Clear All")');
    await clearButton.click();
    
    // Confirm clear
    await page.on('dialog', dialog => dialog.accept());
    
    // Should show empty state
    await expect(page.locator('.empty-state')).toBeVisible();
    await expect(page.locator('text="No generation history yet"')).toBeVisible();
  });

  test('should show generation stats', async ({ page }) => {
    const firstItem = page.locator('.history-item').first();
    
    // Should show various stats
    await expect(firstItem.locator('.stat').first()).toBeVisible();
    await expect(firstItem).toContainText('tokens');
    await expect(firstItem).toContainText('s'); // generation time
  });

  test('should show relative time', async ({ page }) => {
    const timeSpan = page.locator('.item-time').first();
    await expect(timeSpan).toContainText(/ago|just now/);
  });
});