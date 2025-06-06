// Comprehensive help content for all TEMPO settings
export interface SettingHelp {
  id: string;
  name: string;
  description: string;
  examples?: string[];
  warning?: string;
  relatedSettings?: string[];
  category: 'core' | 'generation' | 'mcts' | 'pruning' | 'threshold' | 'model' | 'cache' | 'advanced';
  importance: 'essential' | 'important' | 'advanced' | 'experimental';
}

export const SETTINGS_HELP: Record<string, SettingHelp> = {
  maxTokens: {
    id: 'maxTokens',
    name: 'Maximum Tokens',
    description: 'Controls how many tokens (words/word pieces) to generate. Each token ≈ 0.75 words. The model stops at this limit or when naturally complete.',
    examples: [
      '50 tokens: 1-2 sentences for quick answers',
      '150 tokens: 1-2 paragraphs for explanations',
      '500 tokens: Multiple paragraphs for essays/stories'
    ],
    category: 'core',
    importance: 'essential'
  },

  selectionThreshold: {
    id: 'selectionThreshold',
    name: 'Selection Threshold',
    description: 'Probability threshold for selecting parallel token candidates. Lower = more creative/diverse, Higher = more focused/conservative.',
    examples: [
      '0.01-0.02: Very experimental, may be incoherent',
      '0.05: Creative writing sweet spot',
      '0.15: Balanced for general use',
      '0.3+: Conservative, technical writing'
    ],
    warning: 'Values < 0.02 may produce incoherent text',
    relatedSettings: ['useRetroactiveRemoval', 'attentionThreshold'],
    category: 'core',
    importance: 'essential'
  },

  useRetroactiveRemoval: {
    id: 'useRetroactiveRemoval',
    name: 'Use Retroactive Removal',
    description: 'Removes previously selected tokens that receive little attention from later tokens, improving coherence.',
    examples: [
      'ON: Cleaner output, removes irrelevant branches',
      'OFF: Maximum diversity, keeps all paths'
    ],
    relatedSettings: ['attentionThreshold'],
    category: 'pruning',
    importance: 'important'
  },

  attentionThreshold: {
    id: 'attentionThreshold',
    name: 'Attention Threshold',
    description: 'Minimum attention score for tokens to survive retroactive pruning. Lower = more aggressive pruning.',
    examples: [
      '0.001: Very aggressive, most focused',
      '0.01: Moderate pruning, balanced',
      '0.05: Conservative, preserves diversity'
    ],
    relatedSettings: ['useRetroactivePruning'],
    category: 'pruning',
    importance: 'important'
  },

  debugMode: {
    id: 'debugMode',
    name: 'Debug Mode',
    description: 'Shows detailed internal processing information, timing data, and decision steps.',
    examples: [
      'ON: See token probabilities, pruning decisions, timing',
      'OFF: Clean output only'
    ],
    category: 'advanced',
    importance: 'advanced'
  },

  useMcts: {
    id: 'useMcts',
    name: 'Use MCTS',
    description: 'Monte Carlo Tree Search explores future token paths before deciding, improving long-term coherence but slower.',
    examples: [
      'ON: Better planning for complex reasoning',
      'OFF: Faster generation for simple tasks'
    ],
    warning: 'Significantly increases generation time',
    relatedSettings: ['mctsSimulations', 'mctsCPuct', 'mctsDepth'],
    category: 'mcts',
    importance: 'advanced'
  },

  mctsSimulations: {
    id: 'mctsSimulations',
    name: 'MCTS Simulations',
    description: 'Number of future path explorations per decision. More = better planning but slower.',
    examples: [
      '10-20: Basic planning, reasonable speed',
      '50: Good strategic depth',
      '100+: Research-grade planning, very slow'
    ],
    relatedSettings: ['useMcts'],
    category: 'mcts',
    importance: 'advanced'
  },

  mctsCPuct: {
    id: 'mctsCPuct',
    name: 'MCTS C_PUCT',
    description: 'Exploration vs exploitation balance. Higher = explore more diverse paths, Lower = focus on promising paths.',
    examples: [
      '0.5: Conservative, exploit known good paths',
      '1.4: Standard balanced approach',
      '3.0: High exploration for creativity'
    ],
    relatedSettings: ['useMcts'],
    category: 'mcts',
    importance: 'advanced'
  },

  mctsDepth: {
    id: 'mctsDepth',
    name: 'MCTS Depth',
    description: 'How many tokens ahead to look when planning. Deeper = better long-term awareness but exponentially slower.',
    examples: [
      '3-5: Short-term tactics',
      '8: Medium-term strategy',
      '15+: Long-term narrative planning'
    ],
    relatedSettings: ['useMcts'],
    category: 'mcts',
    importance: 'advanced'
  },

  dynamicThreshold: {
    id: 'dynamicThreshold',
    name: 'Dynamic Threshold',
    description: 'Gradually changes selection threshold during generation for better long-form coherence.',
    examples: [
      'Start creative → End focused: Better conclusions',
      'Start focused → End creative: Surprising endings'
    ],
    relatedSettings: ['finalThreshold', 'bezierP1', 'bezierP2'],
    category: 'threshold',
    importance: 'advanced'
  },

  finalThreshold: {
    id: 'finalThreshold',
    name: 'Final Threshold',
    description: 'Target threshold when using dynamic transitions. Determines end behavior.',
    examples: [
      '0.2: Become more conservative',
      '0.02: Become more creative'
    ],
    relatedSettings: ['dynamicThreshold'],
    category: 'threshold',
    importance: 'advanced'
  },

  useRelu: {
    id: 'useRelu',
    name: 'Use ReLU Transition',
    description: 'Sharp threshold change at activation point instead of smooth Bezier curve.',
    examples: [
      'ReLU: Distinct phases (explore then focus)',
      'Bezier: Smooth gradual transition'
    ],
    relatedSettings: ['reluActivationPoint', 'dynamicThreshold'],
    category: 'threshold',
    importance: 'experimental'
  },

  reluActivationPoint: {
    id: 'reluActivationPoint',
    name: 'ReLU Activation Point',
    description: 'When to sharply transition thresholds (0=start, 1=end of generation).',
    examples: [
      '0.3: Early shift to final strategy',
      '0.7: Maintain initial strategy longer'
    ],
    relatedSettings: ['useRelu'],
    category: 'threshold',
    importance: 'experimental'
  },

  bezierP1: {
    id: 'bezierP1',
    name: 'Bezier P1',
    description: 'First control point for smooth threshold curve. Lower = steeper initial change.',
    examples: [
      '0.1: Quick early adjustment',
      '0.5: Linear-like transition',
      '0.9: Delayed adjustment'
    ],
    relatedSettings: ['dynamicThreshold', 'bezierP2'],
    category: 'threshold',
    importance: 'experimental'
  },

  bezierP2: {
    id: 'bezierP2',
    name: 'Bezier P2',
    description: 'Second control point for smooth threshold curve. Higher = steeper final change.',
    examples: [
      '0.1: Early change then stable',
      '0.5: Linear-like transition',
      '0.9: Late dramatic change'
    ],
    relatedSettings: ['dynamicThreshold', 'bezierP1'],
    category: 'threshold',
    importance: 'experimental'
  },

  useCustomRope: {
    id: 'useCustomRope',
    name: 'Use Custom RoPE (Rotary Position Embeddings)',
    description: 'Enable modified position embeddings that allow multiple tokens to share the same logical position. This is essential for TEMPO\'s parallel processing - without it, the model processes tokens sequentially.',
    warning: 'Disabling this removes TEMPO\'s core benefits and reverts to standard generation',
    category: 'model',
    importance: 'essential'
  },

  disableKvCache: {
    id: 'disableKvCache',
    name: 'Disable KV Cache',
    description: 'Forces fresh attention computation each step. Very slow but useful for debugging.',
    warning: 'Dramatically slows generation',
    category: 'cache',
    importance: 'experimental'
  },

  disableKvCacheConsistency: {
    id: 'disableKvCacheConsistency',
    name: 'Disable KV Cache Consistency',
    description: 'Skip cache validity checks. Experimental - may cause incorrect attention.',
    warning: 'May produce incorrect results',
    category: 'cache',
    importance: 'experimental'
  },

  showTokenIds: {
    id: 'showTokenIds',
    name: 'Show Token IDs',
    description: 'Display numerical vocabulary IDs for each token in output.',
    examples: [
      'Useful for debugging tokenization issues',
      'Understanding model vocabulary'
    ],
    category: 'advanced',
    importance: 'advanced'
  },

  systemContent: {
    id: 'systemContent',
    name: 'System Content',
    description: 'System-level instructions that guide model behavior (like "Be creative" or "Be precise").',
    examples: [
      'Creative: "Be imaginative and explore unusual ideas"',
      'Technical: "Be precise, factual, and concise"',
      'Academic: "Use formal language and cite sources"'
    ],
    category: 'advanced',
    importance: 'important'
  },

  enableThinking: {
    id: 'enableThinking',
    name: 'Enable Thinking',
    description: 'Model shows step-by-step reasoning before final answer (Cogito models only).',
    examples: [
      'Math problems: Shows calculation steps',
      'Logic puzzles: Shows deduction process'
    ],
    category: 'model',
    importance: 'important'
  },

  allowIntrasetTokenVisibility: {
    id: 'allowIntrasetTokenVisibility',
    name: 'Allow Parallel Tokens to See Each Other',
    description: 'Enable attention between tokens at the same position. When off (default), parallel tokens are isolated and can\'t influence each other\'s probabilities. Turning this on allows tokens to \"compete\" directly.',
    category: 'model',
    importance: 'experimental'
  },

  noPreserveIsolatedTokens: {
    id: 'noPreserveIsolatedTokens',
    name: 'No Preserve Isolated Tokens',
    description: 'Allow pruning of tokens even if they are the only option at a position.',
    category: 'pruning',
    importance: 'experimental'
  },

  allowIsolatedTokenRemoval: {
    id: 'allowIsolatedTokenRemoval',
    name: 'Allow Isolated Token Removal',
    description: 'When enabled, tokens that don\'t interact with other parallel tokens can be removed during pruning. This can lead to more aggressive pruning but cleaner output.',
    category: 'pruning',
    importance: 'experimental'
  },

  noRelativeAttention: {
    id: 'noRelativeAttention',
    name: 'No Relative Attention',
    description: 'Use absolute instead of relative attention scores for pruning decisions.',
    relatedSettings: ['relativeThreshold'],
    category: 'pruning',
    importance: 'advanced'
  },

  disableRelativeAttention: {
    id: 'disableRelativeAttention',
    name: 'Disable Relative Attention',
    description: 'Turn off relative attention comparisons. When disabled, pruning uses absolute attention values only instead of comparing tokens relative to each other.',
    relatedSettings: ['relativeThreshold'],
    category: 'pruning',
    importance: 'advanced'
  },

  relativeThreshold: {
    id: 'relativeThreshold',
    name: 'Relative Threshold',
    description: 'Threshold for relative attention comparisons when pruning.',
    examples: [
      '0.3: Aggressive relative pruning',
      '0.7: Conservative relative pruning'
    ],
    relatedSettings: ['noRelativeAttention'],
    category: 'pruning',
    importance: 'advanced'
  },

  noMultiScaleAttention: {
    id: 'noMultiScaleAttention',
    name: 'No Multi-Scale Attention',
    description: 'Disable multi-scale attention analysis in pruning decisions.',
    category: 'pruning',
    importance: 'experimental'
  },

  disableMultiScaleAttention: {
    id: 'disableMultiScaleAttention',
    name: 'Disable Multi-Scale Attention',
    description: 'Turn off attention aggregation across multiple model layers. When disabled, only uses attention from specified layers instead of combining information from multiple scales.',
    category: 'pruning',
    importance: 'experimental'
  },

  noLciDynamicThreshold: {
    id: 'noLciDynamicThreshold',
    name: 'No LCI Dynamic Threshold',
    description: 'Disable layer-wise dynamic threshold adjustments.',
    category: 'threshold',
    importance: 'experimental'
  },

  disableLciDynamicThreshold: {
    id: 'disableLciDynamicThreshold',
    name: 'Disable LCI Dynamic Threshold',
    description: 'Turn off Linear Confidence Interval (LCI) based dynamic thresholding. LCI adjusts thresholds based on statistical confidence intervals during generation.',
    category: 'threshold',
    importance: 'experimental'
  },

  noSigmoidThreshold: {
    id: 'noSigmoidThreshold',
    name: 'No Sigmoid Threshold',
    description: 'Disable sigmoid-based threshold transitions.',
    relatedSettings: ['sigmoidSteepness'],
    category: 'threshold',
    importance: 'experimental'
  },

  disableSigmoidThreshold: {
    id: 'disableSigmoidThreshold',
    name: 'Disable Sigmoid Decision Boundary',
    description: 'Turn off sigmoid-based smoothing for pruning decisions. When disabled, uses hard thresholds instead of smooth sigmoid transitions.',
    relatedSettings: ['sigmoidSteepness'],
    category: 'threshold',
    importance: 'experimental'
  },

  sigmoidSteepness: {
    id: 'sigmoidSteepness',
    name: 'Sigmoid Steepness',
    description: 'Controls how sharp sigmoid threshold transitions are.',
    examples: [
      '5: Gradual sigmoid transition',
      '15: Sharp sigmoid transition'
    ],
    relatedSettings: ['noSigmoidThreshold'],
    category: 'threshold',
    importance: 'experimental'
  },

  completeRemovalMode: {
    id: 'completeRemovalMode',
    name: 'Complete Removal Mode',
    description: 'How to handle completely removed positions.',
    examples: [
      'keep_token: Keep at least one token',
      'keep_unattended: Keep tokens with no attention',
      'remove_position: Remove position entirely'
    ],
    category: 'pruning',
    importance: 'advanced'
  },

  numLayersToUse: {
    id: 'numLayersToUse',
    name: 'Number of Layers to Use',
    description: 'Limit attention analysis to specific number of model layers. Empty = all layers.',
    examples: [
      '8: Use only first 8 layers',
      'Empty: Use all model layers'
    ],
    category: 'pruning',
    importance: 'advanced'
  }
};

// Helper function to get help for a specific setting
export function getSettingHelp(settingId: string): SettingHelp | undefined {
  return SETTINGS_HELP[settingId];
}

// Get all settings in a category
export function getSettingsByCategory(category: string): SettingHelp[] {
  return Object.values(SETTINGS_HELP).filter(help => help.category === category);
}

// Get settings by importance level
export function getSettingsByImportance(importance: string): SettingHelp[] {
  return Object.values(SETTINGS_HELP).filter(help => help.importance === importance);
}

// Get core/essential settings only
export function getCoreSettings(): SettingHelp[] {
  return Object.values(SETTINGS_HELP).filter(help => 
    help.importance === 'essential' || 
    (help.importance === 'important' && help.category === 'core')
  );
}