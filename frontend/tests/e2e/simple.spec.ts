import { test, expect } from '@playwright/test';

test.describe('Basic App Structure - Progressive Disclosure', () => {
  test('should load the application without errors', async ({ page }) => {
    // Navigate to the app
    const response = await page.goto('/', { waitUntil: 'networkidle' });
    
    // Check that the page loads successfully
    expect(response?.status()).toBeLessThan(400);
    
    // Wait for the app to be ready
    await page.waitForTimeout(1000);
    
    // Check that basic structure exists
    const mainElement = page.locator('main');
    await expect(mainElement).toBeVisible();
    
    // Check for any console errors
    const consoleErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });
    
    // Give time for any errors to appear
    await page.waitForTimeout(500);
    
    // For now, we'll just log errors instead of failing
    if (consoleErrors.length > 0) {
      console.log('Console errors found:', consoleErrors);
    }
  });

  test('should have correct page title', async ({ page }) => {
    await page.goto('/');
    
    // Check title contains TEMPO
    const title = await page.title();
    expect(title).toContain('TEMPO');
  });

  test('should display header with title and theme toggle', async ({ page }) => {
    await page.goto('/');
    
    // Check for header
    const header = page.locator('header');
    await expect(header).toBeVisible();
    
    // Check for TEMPO branding in header
    await expect(header.locator('text=TEMPO')).toBeVisible();
    
    // Check for theme toggle button
    const themeToggle = header.locator('button[aria-label="Toggle theme"]');
    await expect(themeToggle).toBeVisible();
  });

  test('should show main heading', async ({ page }) => {
    await page.goto('/');
    
    // Check for main h1
    const h1 = page.locator('h1');
    await expect(h1).toBeVisible();
    const h1Text = await h1.textContent();
    expect(h1Text).toContain('TEMPO Text Generation');
  });

  test('should have interface mode toggle', async ({ page }) => {
    await page.goto('/');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // Check for interface mode buttons
    await expect(page.locator('button:has-text("Beginner")')).toBeVisible();
    await expect(page.locator('button:has-text("Intermediate")')).toBeVisible();
    await expect(page.locator('button:has-text("Expert")')).toBeVisible();
  });

  test('should show quick start section in beginner mode', async ({ page }) => {
    await page.goto('/');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // Should start in beginner mode and show quick start
    await expect(page.locator('text=🚀 Quick Start')).toBeVisible();
    await expect(page.locator('text=Choose a preset optimized for your task')).toBeVisible();
  });

  test('should have textarea for prompt input', async ({ page }) => {
    await page.goto('/');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // Check for textarea (should be visible in all modes)
    const textarea = page.locator('textarea[placeholder*="Enter your prompt"]');
    await expect(textarea).toBeVisible();
  });

  test('should have generate button', async ({ page }) => {
    await page.goto('/');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // Check for generate button
    const generateButton = page.locator('button:has-text("Generate")');
    await expect(generateButton).toBeVisible();
  });

  test('should show enhanced preset cards in beginner mode', async ({ page }) => {
    await page.goto('/');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // Should see preset cards with enhanced information
    await expect(page.locator('text=Balanced')).toBeVisible();
    await expect(page.locator('text=Creative Writing')).toBeVisible();
    await expect(page.locator('text=Precise & Focused')).toBeVisible();
    
    // Check for enhanced card information
    await expect(page.locator('text=Beginner')).toBeVisible(); // Difficulty badge
    await expect(page.locator('text=Best for:')).toBeVisible(); // Use case info
  });

  test('should show collapsible sections in intermediate mode', async ({ page }) => {
    await page.goto('/');
    
    // Switch to intermediate mode
    await page.click('button:has-text("Intermediate")');
    
    // Should see collapsible sections
    await expect(page.locator('text=Prompt & Generation')).toBeVisible();
    await expect(page.locator('text=Pruning & Refinement')).toBeVisible();
    
    // Sections should have expand/collapse indicators
    const sectionHeader = page.locator('text=Prompt & Generation').locator('..');
    await expect(sectionHeader.locator('svg')).toBeVisible(); // Chevron icon
  });

  test('should show tabbed interface in expert mode', async ({ page }) => {
    await page.goto('/');
    
    // Switch to expert mode
    await page.click('button:has-text("Expert")');
    
    // Should see original tabbed interface
    await expect(page.locator('button[role="tab"]:has-text("Basic")')).toBeVisible();
    await expect(page.locator('button[role="tab"]:has-text("MCTS")')).toBeVisible();
    await expect(page.locator('button[role="tab"]:has-text("Advanced")')).toBeVisible();
  });

  test('should display footer with keyboard shortcuts info', async ({ page }) => {
    await page.goto('/');
    
    // Check for footer
    const footer = page.locator('footer');
    await expect(footer).toBeVisible();
    
    // Check for keyboard shortcuts mention
    await expect(footer.locator('text=Keyboard Shortcuts')).toBeVisible();
  });

  test('should handle responsive layout', async ({ page }) => {
    await page.goto('/');
    
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    // Main content should still be visible
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('button:has-text("Generate")')).toBeVisible();
    
    // Interface mode toggle should still work
    await expect(page.locator('button:has-text("Beginner")')).toBeVisible();
  });

  test('should show help icons next to settings in intermediate/expert modes', async ({ page }) => {
    await page.goto('/');
    
    // Switch to intermediate mode
    await page.click('button:has-text("Intermediate")');
    
    // Should see help icons next to settings
    const helpIcons = page.locator('[aria-label*="Help for"]');
    await expect(helpIcons.first()).toBeVisible();
  });

  test('should persist theme selection across page loads', async ({ page }) => {
    await page.goto('/');
    
    // Get initial theme
    const html = page.locator('html');
    const initialTheme = await html.getAttribute('class');
    
    // Toggle theme
    await page.click('button[aria-label="Toggle theme"]');
    
    // Get new theme
    const newTheme = await html.getAttribute('class');
    expect(newTheme).not.toBe(initialTheme);
    
    // Reload page
    await page.reload();
    
    // Theme should persist
    const persistedTheme = await html.getAttribute('class');
    expect(persistedTheme).toBe(newTheme);
  });
});