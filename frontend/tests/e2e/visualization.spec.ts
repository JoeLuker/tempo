import { test, expect } from '@playwright/test';
import { mockAPIResponse, mockSuccessResponse } from '../helpers/api-mock';

test.describe('Visualization Features', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    
    // Mock API and generate text to show visualization
    await mockAPIResponse(page, mockSuccessResponse);
    await page.fill('textarea[placeholder*="Enter your prompt"]', 'Once upon a time');
    await page.click('button:has-text("Generate")');
    
    // Wait for generation to complete
    await expect(page.locator('[data-testid="generated-text"]')).toBeVisible();
  });

  test('should display bar chart visualization', async ({ page }) => {
    // Check SVG chart is rendered
    const chart = page.locator('.chart-container svg');
    await expect(chart).toBeVisible();
    
    // Check chart has bars
    const bars = chart.locator('rect.bar');
    await expect(bars).toHaveCount(6); // 3 tokens per step, 2 steps
  });

  test('should show tooltips on hover', async ({ page }) => {
    // Find a bar in the chart
    const firstBar = page.locator('.chart-container svg rect.bar').first();
    
    // Hover over the bar
    await firstBar.hover();
    
    // Check tooltip appears
    await expect(page.locator('.tooltip')).toBeVisible();
    
    // Verify tooltip content
    const tooltipText = await page.locator('.tooltip').textContent();
    expect(tooltipText).toContain('Probability:');
  });

  test('should highlight pruned tokens differently', async ({ page }) => {
    // Get all bars
    const bars = page.locator('.chart-container svg rect.bar');
    
    // Check that some bars have pruned styling
    const prunedBars = bars.filter({ hasClass: 'pruned' });
    const keptBars = bars.filter({ hasClass: 'kept' });
    
    // Verify both types exist
    await expect(prunedBars).toHaveCount(4); // 2 pruned per step
    await expect(keptBars).toHaveCount(2); // 1 kept per step
  });

  test('should update chart on theme change', async ({ page }) => {
    // Get initial bar color
    const firstBar = page.locator('.chart-container svg rect.bar').first();
    const initialColor = await firstBar.evaluate(el => window.getComputedStyle(el).fill);
    
    // Toggle theme
    await page.click('[data-testid="theme-toggle"]');
    
    // Wait for theme transition
    await page.waitForTimeout(300);
    
    // Check bar color changed
    const newColor = await firstBar.evaluate(el => window.getComputedStyle(el).fill);
    expect(newColor).not.toBe(initialColor);
  });

  test('should show step positions on x-axis', async ({ page }) => {
    // Check x-axis labels
    const xAxisLabels = page.locator('.chart-container svg .x-axis text');
    
    // Verify step labels exist
    await expect(xAxisLabels).toHaveCount(2); // 2 steps
    await expect(xAxisLabels.first()).toContainText('Step 0');
    await expect(xAxisLabels.nth(1)).toContainText('Step 1');
  });

  test('should show probability values on y-axis', async ({ page }) => {
    // Check y-axis exists
    const yAxis = page.locator('.chart-container svg .y-axis');
    await expect(yAxis).toBeVisible();
    
    // Check y-axis has tick marks
    const yAxisTicks = yAxis.locator('.tick');
    expect(await yAxisTicks.count()).toBeGreaterThan(0);
  });

  test('should resize chart on window resize', async ({ page }) => {
    // Get initial chart dimensions
    const chart = page.locator('.chart-container svg');
    const initialWidth = await chart.evaluate(el => el.getBoundingClientRect().width);
    
    // Resize viewport
    await page.setViewportSize({ width: 800, height: 600 });
    
    // Wait for resize handler
    await page.waitForTimeout(500);
    
    // Check chart resized
    const newWidth = await chart.evaluate(el => el.getBoundingClientRect().width);
    expect(newWidth).not.toBe(initialWidth);
  });

  test('should display legend for token types', async ({ page }) => {
    // Check legend exists
    const legend = page.locator('.chart-legend');
    await expect(legend).toBeVisible();
    
    // Verify legend items
    await expect(legend.locator('text=Kept Tokens')).toBeVisible();
    await expect(legend.locator('text=Pruned Tokens')).toBeVisible();
  });
});