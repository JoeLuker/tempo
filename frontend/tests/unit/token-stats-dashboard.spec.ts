import { test, expect } from '@playwright/test';

test.describe('Token Stats Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    
    // Mock API response with rich token data
    await page.route('/api/v2/generate', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          clean_text: 'The quick brown fox jumps over the lazy dog.',
          generated_text: '[The/A] [quick/fast] brown fox jumps over the [lazy/sleepy] dog.',
          steps: [
            {
              position: 0,
              parallel_tokens: [
                { token_text: 'The', token_id: 1, probability: 0.7 },
                { token_text: 'A', token_id: 2, probability: 0.3 }
              ],
              removed_tokens: []
            },
            {
              position: 1,
              parallel_tokens: [
                { token_text: 'quick', token_id: 3, probability: 0.6 },
                { token_text: 'fast', token_id: 4, probability: 0.4 }
              ],
              removed_tokens: [
                { token_text: 'fast', token_id: 4, probability: 0.4 }
              ]
            }
          ],
          original_parallel_positions: [0, 1, 7],
          timing: { elapsed_time: 2.5 },
          retroactive_removal: {
            attention_threshold: 0.01
          }
        })
      });
    });
    
    // Generate and navigate to stats
    await page.locator('textarea#prompt').fill('Test stats');
    await page.locator('button:has-text("Generate")').click();
    await page.waitForTimeout(500);
    
    // Click on Stats tab
    await page.locator('[role="tab"]:has-text("Stats")').click();
  });

  test('should display overall statistics', async ({ page }) => {
    const dashboard = page.locator('.token-stats-dashboard');
    await expect(dashboard).toBeVisible();
    
    // Check stat cards
    await expect(page.locator('.stat-card').first()).toContainText('Total Tokens');
    await expect(page.locator('.stat-card')).toContainText('Positions');
    await expect(page.locator('.stat-card')).toContainText('Avg Tokens/Pos');
    await expect(page.locator('.stat-card')).toContainText('Removal Rate');
    await expect(page.locator('.stat-card')).toContainText('Avg Probability');
    await expect(page.locator('.stat-card')).toContainText('Token Diversity');
  });

  test('should show probability distribution chart', async ({ page }) => {
    const vizSection = page.locator('.viz-section:has-text("Probability Distribution")');
    await expect(vizSection).toBeVisible();
    
    // Should have SVG chart
    const svg = vizSection.locator('svg');
    await expect(svg).toBeVisible();
    
    // Should have bars
    await expect(svg.locator('.bar')).toHaveCount(5); // 5 probability ranges
  });

  test('should show token frequency chart', async ({ page }) => {
    const vizSection = page.locator('.viz-section:has-text("Most Frequent Tokens")');
    await expect(vizSection).toBeVisible();
    
    // Should have SVG chart
    const svg = vizSection.locator('svg');
    await expect(svg).toBeVisible();
    
    // Should have frequency bars
    await expect(svg.locator('.freq-bar').first()).toBeVisible();
  });

  test('should show position heatmap', async ({ page }) => {
    const heatmapSection = page.locator('.position-heatmap-section');
    await expect(heatmapSection).toBeVisible();
    await expect(heatmapSection).toContainText('Token Count by Position');
    
    // Should have position cells
    const svg = heatmapSection.locator('svg');
    await expect(svg.locator('.position-cell').first()).toBeVisible();
  });

  test('should show position details table', async ({ page }) => {
    const detailsSection = page.locator('.position-details');
    await expect(detailsSection).toBeVisible();
    
    // Should have table headers
    const table = detailsSection.locator('table');
    await expect(table.locator('th')).toContainText(['Pos', 'Tokens', 'Removed', 'Avg Prob', 'Top Tokens']);
    
    // Should have data rows
    await expect(table.locator('tbody tr').first()).toBeVisible();
  });

  test('should highlight removed tokens in table', async ({ page }) => {
    const table = page.locator('.position-table');
    
    // Find cells with removals
    const removalCells = table.locator('td.has-removals');
    const count = await removalCells.count();
    
    if (count > 0) {
      // Should have special styling for cells with removals
      await expect(removalCells.first()).toHaveCSS('color', /rgb\(.*\)/);
    }
  });

  test('should show top tokens with probabilities', async ({ page }) => {
    const topTokensCell = page.locator('.top-tokens').first();
    await expect(topTokensCell).toBeVisible();
    
    // Should show token chips
    const tokenChips = topTokensCell.locator('.token-chip');
    await expect(tokenChips.first()).toBeVisible();
    await expect(tokenChips.first()).toContainText(/%/); // Contains percentage
  });

  test('should be scrollable for long position lists', async ({ page }) => {
    const scrollContainer = page.locator('.position-scroll');
    
    // Check if container has scroll capability
    const hasScroll = await scrollContainer.evaluate(el => {
      return el.scrollHeight > el.clientHeight;
    });
    
    // If there's enough content, it should be scrollable
    if (hasScroll) {
      await expect(scrollContainer).toHaveCSS('overflow-y', 'auto');
    }
  });

  test('should show empty state when no data', async ({ page }) => {
    // Navigate to a fresh page
    await page.goto('/');
    await page.locator('[role="tab"]:has-text("Stats")').click();
    
    // Should show no data message
    await expect(page.locator('.no-data')).toBeVisible();
    await expect(page.locator('text="No generation data available"')).toBeVisible();
  });
});