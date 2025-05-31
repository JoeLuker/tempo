export const testPrompts = {
  simple: 'Once upon a time',
  complex: 'In the realm of quantum computing, researchers have discovered',
  empty: '',
  long: 'A'.repeat(500),
};

export const testSettings = {
  default: {
    maxTokens: 50,
    selectionThreshold: 0.1,
    attentionThreshold: 0.02,
    useRetroactivePruning: false,
  },
  creative: {
    maxTokens: 100,
    selectionThreshold: 0.15,
    attentionThreshold: 0.03,
    useRetroactivePruning: true,
  },
  conservative: {
    maxTokens: 30,
    selectionThreshold: 0.05,
    attentionThreshold: 0.01,
    useRetroactivePruning: false,
  },
};

export const apiResponses = {
  success: {
    generated_text: "Once upon a time in a land far away, there lived a wise old wizard",
    prompt: "Once upon a time",
    max_tokens: 50,
    selection_threshold: 0.1,
    steps: generateMockSteps(10),
    generation_time: 0.45,
    stats: {
      total_generated: 15,
      parallel_considered: 45,
      pruning_time: 0.05
    }
  },
  error: {
    detail: "Model loading failed"
  },
  timeout: {
    detail: "Request timeout"
  }
};

function generateMockSteps(count: number) {
  const steps = [];
  for (let i = 0; i < count; i++) {
    steps.push({
      position: i,
      parallel_tokens: [
        { token_text: ` token${i}_1`, token_id: i * 100 + 1, probability: 0.45 },
        { token_text: ` token${i}_2`, token_id: i * 100 + 2, probability: 0.35 },
        { token_text: ` token${i}_3`, token_id: i * 100 + 3, probability: 0.20 }
      ],
      pruned_tokens: [
        { token_text: ` token${i}_1`, token_id: i * 100 + 1, probability: 0.45 }
      ]
    });
  }
  return steps;
}