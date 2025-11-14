#!/usr/bin/env python3
"""Capture a screenshot of the TEMPO playground after generation."""

from playwright.sync_api import sync_playwright
import time

def capture_playground():
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={'width': 1400, 'height': 900})

        # Listen to console messages
        page.on('console', lambda msg: print(f'[BROWSER] {msg.type}: {msg.text}'))

        # Navigate to playground
        print("Navigating to playground...")
        page.goto('http://localhost:5174')

        # Wait for page to load
        page.wait_for_selector('input[type="text"]', timeout=10000)
        print("Page loaded!")

        # Wait a moment for debug indicators to update
        time.sleep(2)

        # Take screenshot of initial state with indicators
        print("Taking initial screenshot with debug indicators...")
        page.screenshot(path='playground_initial.png', full_page=True)

        # Enter prompt
        print("Entering prompt...")
        page.fill('input[type="text"]', 'Once upon a time in a magical forest')

        # Click generate button
        print("Clicking generate button...")
        page.click('button:has-text("Generate")')

        # Wait for generation to complete
        # The generation might take a while, so we'll wait for the SVG to appear
        print("Waiting for generation to complete...")

        # Wait for either error or SVG to appear
        try:
            page.wait_for_selector('svg', timeout=60000)
            print("SVG appeared! Generation complete.")
        except Exception as e:
            print(f"Timeout or error waiting for SVG: {e}")
            # Take screenshot anyway to see what happened
            page.screenshot(path='playground_error.png', full_page=True)
            raise

        # Give it extra time for all tokens to stream in (20 tokens * 200ms = 4 seconds)
        print("Waiting for tokens to fully stream in...")
        time.sleep(6)

        # Take screenshot
        print("Taking screenshot...")
        page.screenshot(path='playground_screenshot.png', full_page=True)

        print("Screenshot saved to playground_screenshot.png")

        # Keep browser open for a moment so we can see it
        time.sleep(2)

        browser.close()

if __name__ == '__main__':
    capture_playground()
