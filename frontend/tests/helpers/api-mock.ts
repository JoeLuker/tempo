import { Page } from '@playwright/test';

export async function mockAPIResponse(page: Page, response: any) {
  await page.route('**/api/v2/generate', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(response),
    });
  });
}

export async function mockAPIError(page: Page, statusCode: number = 500, message: string = 'Internal Server Error') {
  await page.route('**/api/v2/generate', async (route) => {
    await route.fulfill({
      status: statusCode,
      contentType: 'application/json',
      body: JSON.stringify({ detail: message }),
    });
  });
}

export const mockSuccessResponse = {
  generated_text: "Once upon a time in a land far away",
  prompt: "Once upon a time",
  max_tokens: 50,
  selection_threshold: 0.1,
  steps: [
    {
      position: 0,
      parallel_tokens: [
        { token_text: " in", token_id: 287, probability: 0.45 },
        { token_text: " there", token_id: 612, probability: 0.35 },
        { token_text: ",", token_id: 11, probability: 0.20 }
      ],
      pruned_tokens: [
        { token_text: " in", token_id: 287, probability: 0.45 }
      ]
    },
    {
      position: 1,
      parallel_tokens: [
        { token_text: " a", token_id: 263, probability: 0.60 },
        { token_text: " the", token_id: 278, probability: 0.25 },
        { token_text: " an", token_id: 385, probability: 0.15 }
      ],
      pruned_tokens: [
        { token_text: " a", token_id: 263, probability: 0.60 }
      ]
    }
  ],
  generation_time: 0.45,
  stats: {
    total_generated: 10,
    parallel_considered: 25,
    pruning_time: 0.05
  }
};