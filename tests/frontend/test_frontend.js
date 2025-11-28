/**
 * Comprehensive Frontend Testing Script for TEMPO
 * Tests UI functionality, interactions, and visual elements
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

// Configuration
const BASE_URL = 'http://localhost:5174';
const SCREENSHOTS_DIR = path.join(__dirname, 'test_screenshots');
const TEST_RESULTS = [];

// Ensure screenshots directory exists
if (!fs.existsSync(SCREENSHOTS_DIR)) {
    fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });
}

/**
 * Log test result
 */
function logResult(testName, status, details = '') {
    const result = {
        test: testName,
        status: status,
        details: details,
        timestamp: new Date().toISOString()
    };
    TEST_RESULTS.push(result);
    console.log(`[${status}] ${testName}: ${details}`);
}

/**
 * Wait for network idle with timeout
 */
async function waitForNetworkIdle(page, timeout = 5000) {
    try {
        await page.waitForNetworkIdle({ timeout });
    } catch (e) {
        console.log('Network idle timeout - continuing anyway');
    }
}

/**
 * Simple timeout helper
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Test 1: Initial Page Load
 */
async function testInitialLoad(page) {
    console.log('\n=== TEST 1: Initial Page Load ===');

    try {
        const response = await page.goto(BASE_URL, {
            waitUntil: 'networkidle2',
            timeout: 30000
        });

        if (response.ok()) {
            logResult('Initial Page Load', 'PASS', `Status: ${response.status()}`);
        } else {
            logResult('Initial Page Load', 'FAIL', `Status: ${response.status()}`);
        }

        // Wait a moment for any dynamic content
        await sleep(2000);

        // Capture screenshot
        await page.screenshot({
            path: path.join(SCREENSHOTS_DIR, '01_initial_load.png'),
            fullPage: true
        });
        logResult('Initial Screenshot', 'PASS', 'Captured full page screenshot');

        // Check for basic UI elements
        const elements = {
            'Header/Title': 'h1, h2, .title, header',
            'Input Field': 'input[type="text"], textarea',
            'Generate Button': 'button',
            'Visualization Area': '.visualization, #tree, svg, canvas'
        };

        for (const [name, selector] of Object.entries(elements)) {
            const element = await page.$(selector);
            if (element) {
                logResult(`UI Element: ${name}`, 'PASS', `Found with selector: ${selector}`);
            } else {
                logResult(`UI Element: ${name}`, 'FAIL', `Not found with selector: ${selector}`);
            }
        }

    } catch (error) {
        logResult('Initial Page Load', 'FAIL', error.message);
    }
}

/**
 * Test 2: Controls Panel
 */
async function testControlsPanel(page) {
    console.log('\n=== TEST 2: Controls Panel ===');

    try {
        // Look for various control elements
        const controls = await page.evaluate(() => {
            const found = {
                inputs: document.querySelectorAll('input').length,
                buttons: document.querySelectorAll('button').length,
                textareas: document.querySelectorAll('textarea').length,
                selects: document.querySelectorAll('select').length
            };
            return found;
        });

        logResult('Controls Count', 'INFO', JSON.stringify(controls));

        // Capture controls screenshot
        await page.screenshot({
            path: path.join(SCREENSHOTS_DIR, '02_controls_panel.png'),
            fullPage: true
        });
        logResult('Controls Screenshot', 'PASS', 'Captured controls panel');

    } catch (error) {
        logResult('Controls Panel Test', 'FAIL', error.message);
    }
}

/**
 * Test 3: Default Generation
 */
async function testDefaultGeneration(page) {
    console.log('\n=== TEST 3: Default Generation Test ===');

    try {
        // Find and click generate button
        const buttonSelectors = [
            'button:has-text("Generate")',
            'button[type="submit"]',
            'button.generate',
            'button'
        ];

        let clicked = false;
        for (const selector of buttonSelectors) {
            try {
                const button = await page.$(selector);
                if (button) {
                    const text = await page.evaluate(el => el.textContent, button);
                    logResult('Found Button', 'INFO', `Text: "${text}"`);

                    // Screenshot before clicking
                    await page.screenshot({
                        path: path.join(SCREENSHOTS_DIR, '03_before_generate.png'),
                        fullPage: true
                    });

                    await button.click();
                    clicked = true;
                    logResult('Button Click', 'PASS', 'Successfully clicked generate button');
                    break;
                }
            } catch (e) {
                continue;
            }
        }

        if (!clicked) {
            logResult('Button Click', 'FAIL', 'Could not find or click generate button');
            return;
        }

        // Wait a moment and capture during generation
        await sleep(1500);
        await page.screenshot({
            path: path.join(SCREENSHOTS_DIR, '04_during_generation.png'),
            fullPage: true
        });
        logResult('During Generation Screenshot', 'PASS', 'Captured mid-generation state');

        // Wait for generation to complete (max 15 seconds)
        await sleep(15000);

        // Capture final result
        await page.screenshot({
            path: path.join(SCREENSHOTS_DIR, '05_after_generation.png'),
            fullPage: true
        });
        logResult('After Generation Screenshot', 'PASS', 'Captured post-generation state');

        // Check if any tokens/results appeared
        const hasResults = await page.evaluate(() => {
            const text = document.body.textContent;
            const svgElements = document.querySelectorAll('svg');
            const canvasElements = document.querySelectorAll('canvas');
            return {
                bodyLength: text.length,
                svgCount: svgElements.length,
                canvasCount: canvasElements.length,
                hasVisualization: svgElements.length > 0 || canvasElements.length > 0
            };
        });

        logResult('Results Check', 'INFO', JSON.stringify(hasResults));

        if (hasResults.hasVisualization) {
            logResult('Generation Result', 'PASS', 'Visualization elements found');
        } else {
            logResult('Generation Result', 'WARN', 'No visualization elements detected');
        }

    } catch (error) {
        logResult('Default Generation Test', 'FAIL', error.message);
    }
}

/**
 * Test 4: Custom Input
 */
async function testCustomInput(page) {
    console.log('\n=== TEST 4: Custom Input Test ===');

    try {
        // Reload page for fresh start
        await page.reload({ waitUntil: 'networkidle2' });
        await sleep(2000);

        // Find input field
        const input = await page.$('input[type="text"], textarea');

        if (!input) {
            logResult('Custom Input Test', 'FAIL', 'Could not find input field');
            return;
        }

        // Clear and type new prompt
        await input.click({ clickCount: 3 }); // Select all
        await page.keyboard.press('Backspace');
        await input.type('The quick brown fox', { delay: 50 });

        await page.screenshot({
            path: path.join(SCREENSHOTS_DIR, '06_custom_input.png'),
            fullPage: true
        });
        logResult('Custom Input', 'PASS', 'Entered custom prompt');

        // Try to generate with custom input
        const button = await page.$('button');
        if (button) {
            await button.click();
            await sleep(5000);

            await page.screenshot({
                path: path.join(SCREENSHOTS_DIR, '07_custom_generation.png'),
                fullPage: true
            });
            logResult('Custom Generation', 'PASS', 'Generated with custom input');
        }

    } catch (error) {
        logResult('Custom Input Test', 'FAIL', error.message);
    }
}

/**
 * Test 5: Responsive Design
 */
async function testResponsiveDesign(page) {
    console.log('\n=== TEST 5: Responsive Design Test ===');

    try {
        // Test tablet portrait (768x1024)
        await page.setViewport({ width: 768, height: 1024 });
        await sleep(1000);
        await page.screenshot({
            path: path.join(SCREENSHOTS_DIR, '08_tablet_portrait.png'),
            fullPage: true
        });
        logResult('Tablet Portrait', 'PASS', '768x1024 screenshot captured');

        // Test tablet landscape (1024x768)
        await page.setViewport({ width: 1024, height: 768 });
        await sleep(1000);
        await page.screenshot({
            path: path.join(SCREENSHOTS_DIR, '09_tablet_landscape.png'),
            fullPage: true
        });
        logResult('Tablet Landscape', 'PASS', '1024x768 screenshot captured');

        // Test mobile (375x667)
        await page.setViewport({ width: 375, height: 667 });
        await sleep(1000);
        await page.screenshot({
            path: path.join(SCREENSHOTS_DIR, '10_mobile.png'),
            fullPage: true
        });
        logResult('Mobile View', 'PASS', '375x667 screenshot captured');

        // Reset to desktop
        await page.setViewport({ width: 1920, height: 1080 });

    } catch (error) {
        logResult('Responsive Design Test', 'FAIL', error.message);
    }
}

/**
 * Test 6: D3 Visualization
 */
async function testVisualization(page) {
    console.log('\n=== TEST 6: D3 Visualization Test ===');

    try {
        // Check for SVG elements
        const vizInfo = await page.evaluate(() => {
            const svg = document.querySelector('svg');
            if (!svg) return { found: false };

            return {
                found: true,
                width: svg.getAttribute('width') || svg.getBoundingClientRect().width,
                height: svg.getAttribute('height') || svg.getBoundingClientRect().height,
                nodeCount: svg.querySelectorAll('circle, rect, g.node').length,
                linkCount: svg.querySelectorAll('path, line, g.link').length,
                hasZoom: svg.getAttribute('transform') !== null
            };
        });

        logResult('Visualization Check', 'INFO', JSON.stringify(vizInfo));

        if (vizInfo.found) {
            logResult('D3 Visualization', 'PASS', `Found SVG with ${vizInfo.nodeCount} nodes and ${vizInfo.linkCount} links`);

            // Try to test zoom/pan if visualization exists
            const svg = await page.$('svg');
            if (svg) {
                const box = await svg.boundingBox();
                if (box) {
                    // Try mouse wheel zoom
                    await page.mouse.move(box.x + box.width/2, box.y + box.height/2);
                    await page.mouse.wheel({ deltaY: -100 });
                    await sleep(500);

                    await page.screenshot({
                        path: path.join(SCREENSHOTS_DIR, '11_viz_zoomed.png'),
                        fullPage: true
                    });
                    logResult('Zoom Test', 'PASS', 'Tested zoom interaction');
                }
            }
        } else {
            logResult('D3 Visualization', 'WARN', 'No SVG visualization found');
        }

    } catch (error) {
        logResult('Visualization Test', 'FAIL', error.message);
    }
}

/**
 * Test 7: Console Errors
 */
async function testConsoleErrors(page) {
    console.log('\n=== TEST 7: Console Errors Check ===');

    const errors = [];
    const warnings = [];

    page.on('console', msg => {
        if (msg.type() === 'error') {
            errors.push(msg.text());
        } else if (msg.type() === 'warning') {
            warnings.push(msg.text());
        }
    });

    page.on('pageerror', error => {
        errors.push(error.message);
    });

    // Reload and wait
    await page.reload({ waitUntil: 'networkidle2' });
    await sleep(3000);

    if (errors.length > 0) {
        logResult('Console Errors', 'FAIL', `Found ${errors.length} errors: ${errors.join('; ')}`);
    } else {
        logResult('Console Errors', 'PASS', 'No console errors detected');
    }

    if (warnings.length > 0) {
        logResult('Console Warnings', 'WARN', `Found ${warnings.length} warnings`);
    }
}

/**
 * Main test runner
 */
async function runTests() {
    console.log('Starting TEMPO Frontend Testing...\n');

    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });

    try {
        await testInitialLoad(page);
        await testControlsPanel(page);
        await testDefaultGeneration(page);
        await testCustomInput(page);
        await testResponsiveDesign(page);
        await testVisualization(page);
        await testConsoleErrors(page);

    } catch (error) {
        console.error('Test suite error:', error);
    } finally {
        await browser.close();
    }

    // Generate report
    console.log('\n' + '='.repeat(80));
    console.log('TEST RESULTS SUMMARY');
    console.log('='.repeat(80));

    const summary = {
        total: TEST_RESULTS.length,
        passed: TEST_RESULTS.filter(r => r.status === 'PASS').length,
        failed: TEST_RESULTS.filter(r => r.status === 'FAIL').length,
        warnings: TEST_RESULTS.filter(r => r.status === 'WARN').length,
        info: TEST_RESULTS.filter(r => r.status === 'INFO').length
    };

    console.log(`Total Tests: ${summary.total}`);
    console.log(`Passed: ${summary.passed}`);
    console.log(`Failed: ${summary.failed}`);
    console.log(`Warnings: ${summary.warnings}`);
    console.log(`Info: ${summary.info}`);
    console.log('='.repeat(80));

    // Save detailed report
    const reportPath = path.join(__dirname, 'test_report.json');
    fs.writeFileSync(reportPath, JSON.stringify({
        summary,
        results: TEST_RESULTS,
        screenshotsDir: SCREENSHOTS_DIR
    }, null, 2));

    console.log(`\nDetailed report saved to: ${reportPath}`);
    console.log(`Screenshots saved to: ${SCREENSHOTS_DIR}`);

    return summary.failed === 0 ? 0 : 1;
}

// Run tests
runTests().then(exitCode => {
    process.exit(exitCode);
}).catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
});
