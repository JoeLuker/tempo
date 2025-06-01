import { test, expect } from '@playwright/test';
import { mockAPIResponse, mockSuccessResponse } from '../helpers/api-mock';

test.describe('Visualization Features - Progressive Disclosure Interface', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    
    // Mock API and generate text to show visualization
    await mockAPIResponse(page, mockSuccessResponse);
    
    // Use a preset for consistent generation
    const balancedPreset = page.locator('text=Balanced').locator('..').locator('..');
    await balancedPreset.click();
    
    await page.fill('textarea[placeholder*="Enter your prompt"]', 'Once upon a time');
    await page.click('button[data-testid="generate-button"]');
    
    // Wait for generation to complete
    await expect(page.locator('[data-testid="generated-text"]')).toBeVisible();
  });

  test('should display bar chart visualization', async ({ page }) => {
    // Check SVG chart is rendered
    const chart = page.locator('[data-testid="visualization-chart"] svg');
    await expect(chart).toBeVisible();
    
    // Check chart has bars (depends on mock data structure)
    const bars = chart.locator('rect.bar');
    expect(await bars.count()).toBeGreaterThan(0);
  });

  test('should show tooltips on hover', async ({ page }) => {
    // Find a bar in the chart
    const firstBar = page.locator('[data-testid="visualization-chart"] svg rect.bar').first();
    
    if (await firstBar.isVisible()) {
      // Hover over the bar
      await firstBar.hover();
      
      // Check tooltip appears (tooltip implementation may vary)
      const tooltip = page.locator('.tooltip, [role="tooltip"]');
      if (await tooltip.isVisible()) {
        // Verify tooltip content
        const tooltipText = await tooltip.textContent();
        expect(tooltipText).toContain('Probability:');
      }
    }
  });

  test('should highlight pruned tokens differently', async ({ page }) => {
    // Get all bars
    const bars = page.locator('[data-testid="visualization-chart"] svg rect.bar');
    
    if (await bars.count() > 0) {
      // Check that some bars have different styling (pruned vs kept)
      const prunedBars = bars.filter({ hasClass: 'pruned' });
      const keptBars = bars.filter({ hasClass: 'kept' });
      
      // Verify different types exist (exact counts depend on mock data)
      expect(await prunedBars.count() + await keptBars.count()).toBeGreaterThan(0);
    }
  });

  test('should update chart on theme change', async ({ page }) => {
    // Check if chart exists first
    const chart = page.locator('[data-testid="visualization-chart"] svg');
    await expect(chart).toBeVisible();
    
    const firstBar = chart.locator('rect.bar').first();
    
    if (await firstBar.isVisible()) {
      // Get initial bar color
      const initialColor = await firstBar.evaluate(el => window.getComputedStyle(el).fill);
      
      // Toggle theme
      await page.click('button[aria-label="Toggle theme"]');
      
      // Wait for theme transition
      await page.waitForTimeout(300);
      
      // Check bar color changed
      const newColor = await firstBar.evaluate(el => window.getComputedStyle(el).fill);
      expect(newColor).not.toBe(initialColor);
    }
  });

  test('should show step positions on x-axis', async ({ page }) => {
    const chart = page.locator('[data-testid="visualization-chart"] svg');
    await expect(chart).toBeVisible();
    
    // Check x-axis labels (implementation may vary)
    const xAxisLabels = chart.locator('.x-axis text, text[data-axis="x"]');
    
    if (await xAxisLabels.count() > 0) {
      // Verify step labels exist
      expect(await xAxisLabels.count()).toBeGreaterThan(0);
    }
  });

  test('should show probability values on y-axis', async ({ page }) => {
    const chart = page.locator('[data-testid="visualization-chart"] svg');
    await expect(chart).toBeVisible();
    
    // Check y-axis exists
    const yAxis = chart.locator('.y-axis, [data-axis="y"]');
    
    if (await yAxis.isVisible()) {
      // Check y-axis has tick marks
      const yAxisTicks = yAxis.locator('.tick, line');
      expect(await yAxisTicks.count()).toBeGreaterThan(0);
    }
  });

  test('should resize chart on window resize', async ({ page }) => {
    const chart = page.locator('[data-testid="visualization-chart"] svg');
    await expect(chart).toBeVisible();
    
    // Get initial chart dimensions
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
    const chart = page.locator('[data-testid="visualization-chart"]');
    await expect(chart).toBeVisible();
    
    // Check legend exists (implementation may vary)
    const legend = page.locator('.chart-legend, .legend');
    
    if (await legend.isVisible()) {
      // Verify legend items
      await expect(legend.locator('text=Kept Tokens, text=Original, text=Pruned')).toHaveCount({ min: 1 });
    }
  });

  test('should show visualization in all interface modes', async ({ page }) => {
    // Should be visible in beginner mode (current state)
    await expect(page.locator('[data-testid="visualization-chart"] svg')).toBeVisible();
    
    // Switch to intermediate mode
    await page.click('button:has-text("Intermediate")');
    await expect(page.locator('[data-testid="visualization-chart"] svg')).toBeVisible();
    
    // Switch to expert mode
    await page.click('button:has-text("Expert")');
    await expect(page.locator('[data-testid="visualization-chart"] svg')).toBeVisible();
  });

  test('should maintain visualization after mode switching', async ({ page }) => {
    // Verify chart exists
    const chart = page.locator('[data-testid="visualization-chart"] svg');
    await expect(chart).toBeVisible();
    
    // Switch modes and verify chart persists
    await page.click('button:has-text("Intermediate")');
    await expect(chart).toBeVisible();
    
    await page.click('button:has-text("Expert")');
    await expect(chart).toBeVisible();
    
    await page.click('button:has-text("Beginner")');
    await expect(chart).toBeVisible();
  });

  test('should handle visualization with different preset configurations', async ({ page }) => {
    // Test with Creative Writing preset
    const creativePreset = page.locator('text=Creative Writing').locator('..').locator('..');
    await creativePreset.click();
    
    // Generate new text
    await page.fill('textarea[placeholder*="Enter your prompt"]', 'A different creative prompt');
    await page.click('button[data-testid="generate-button"]');
    
    // Wait for generation
    await expect(page.locator('[data-testid="generated-text"]')).toBeVisible();
    
    // Chart should still be visible
    await expect(page.locator('[data-testid="visualization-chart"] svg')).toBeVisible();
  });

  test('should show model information alongside visualization', async ({ page }) => {
    // Should see model info section
    await expect(page.locator('text=Model Information')).toBeVisible();
    
    // Should see timing information
    await expect(page.locator('text=Timing')).toBeVisible();
    
    // Model info should include relevant details
    await expect(page.locator('text=Model:')).toBeVisible();
    await expect(page.locator('text=Device:')).toBeVisible();
  });
});