import { test, expect } from '@playwright/test';

test.describe('Interactive Output', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    
    // Mock API response with parallel tokens
    await page.route('/api/v2/generate', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          clean_text: 'The cat sat on the mat.',
          generated_text: '[The/A] [cat/dog] sat on [the/a] mat.',
          steps: [
            {
              position: 0,
              parallel_tokens: [
                { token_text: 'The', token_id: 1, probability: 0.7 },
                { token_text: 'A', token_id: 2, probability: 0.3 }
              ]
            },
            {
              position: 1,
              parallel_tokens: [
                { token_text: 'cat', token_id: 3, probability: 0.6 },
                { token_text: 'dog', token_id: 4, probability: 0.4 }
              ],
              removed_tokens: [
                { token_text: 'dog', token_id: 4, probability: 0.4 }
              ]
            }
          ],
          retroactive_removal: {
            attention_threshold: 0.01
          }
        })
      });
    });
    
    // Generate text
    await page.locator('textarea#prompt').fill('Test interactive');
    await page.locator('button:has-text("Generate")').click();
    await page.waitForTimeout(500);
    
    // Make sure Interactive tab is selected
    await page.locator('[role="tab"]:has-text("Interactive")').click();
  });

  test('should display interactive output', async ({ page }) => {
    const interactiveOutput = page.locator('.interactive-output');
    await expect(interactiveOutput).toBeVisible();
    
    // Should have help text
    await expect(interactiveOutput).toContainText('Hover over highlighted tokens');
  });

  test('should show retroactive removal info when enabled', async ({ page }) => {
    const removalInfo = page.locator('.removal-info');
    await expect(removalInfo).toBeVisible();
    await expect(removalInfo).toContainText('Retroactive removal was applied');
  });

  test('should highlight parallel tokens', async ({ page }) => {
    const parallelTokens = page.locator('.selected-token');
    await expect(parallelTokens).toHaveCount(4); // Based on our mock data
    
    // Should have special styling
    const firstToken = parallelTokens.first();
    await expect(firstToken).toHaveCSS('background', /rgba?\(.*\)/);
    await expect(firstToken).toHaveCSS('border-bottom', /2px solid/);
  });

  test('should show alternatives on hover', async ({ page }) => {
    const firstParallelToken = page.locator('.selected-token').first();
    
    // Hover over token
    await firstParallelToken.hover();
    
    // Should show alternatives popup
    const popup = page.locator('.alternatives-popup');
    await expect(popup).toBeVisible();
    await expect(popup).toContainText('TEMPO considered');
    await expect(popup).toContainText('tokens');
  });

  test('should display alternative tokens in popup', async ({ page }) => {
    const firstParallelToken = page.locator('.selected-token').first();
    await firstParallelToken.hover();
    
    const alternatives = page.locator('.alternative-token');
    await expect(alternatives).toHaveCount(2); // The and A
    
    // Should show rank, text, and probability
    const firstAlt = alternatives.first();
    await expect(firstAlt).toContainText('#1');
    await expect(firstAlt).toContainText('The');
    await expect(firstAlt).toContainText('%');
  });

  test('should mark selected token in alternatives', async ({ page }) => {
    const firstParallelToken = page.locator('.selected-token').first();
    await firstParallelToken.hover();
    
    const selectedAlt = page.locator('.alternative-token.selected');
    await expect(selectedAlt).toBeVisible();
    await expect(selectedAlt).toHaveCSS('background', /rgba?\(.*\)/);
  });

  test('should show removed tokens with strikethrough', async ({ page }) => {
    // Hover over second token (cat/dog)
    const secondToken = page.locator('.selected-token').nth(1);
    await secondToken.hover();
    
    const removedToken = page.locator('.alternative-token.removed');
    await expect(removedToken).toBeVisible();
    await expect(removedToken.locator('.token-text')).toHaveCSS('text-decoration', /line-through/);
  });

  test('should hide popup on mouse leave', async ({ page }) => {
    const firstParallelToken = page.locator('.selected-token').first();
    
    // Hover and verify popup appears
    await firstParallelToken.hover();
    const popup = page.locator('.alternatives-popup');
    await expect(popup).toBeVisible();
    
    // Move mouse away
    await page.mouse.move(0, 0);
    await page.waitForTimeout(100);
    
    // Popup should be hidden
    await expect(popup).not.toBeVisible();
  });

  test('should show position badge in popup', async ({ page }) => {
    const firstParallelToken = page.locator('.selected-token').first();
    await firstParallelToken.hover();
    
    const positionBadge = page.locator('.position-badge');
    await expect(positionBadge).toBeVisible();
    await expect(positionBadge).toContainText('Pos');
  });

  test('should handle single tokens without alternatives', async ({ page }) => {
    const singleTokens = page.locator('.single-token');
    const count = await singleTokens.count();
    
    if (count > 0) {
      // Single tokens should not have hover effects
      const firstSingle = singleTokens.first();
      await firstSingle.hover();
      
      // Should not show popup
      const popup = page.locator('.alternatives-popup');
      await expect(popup).not.toBeVisible();
    }
  });
});