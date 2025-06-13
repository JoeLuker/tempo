import { writable, get } from 'svelte/store';
import { browser } from '$app/environment';

// Define types for our settings
export type Settings = {
  // General settings
  maxTokens: number;
  selectionThreshold: number;
  useRetroactiveRemoval: boolean;
  attentionThreshold: number;
  debugMode: boolean;
  
  // MCTS settings
  useMcts: boolean;
  mctsSimulations: number;
  mctsCPuct: number;
  mctsDepth: number;
  
  // Dynamic threshold settings
  dynamicThreshold: boolean;
  finalThreshold: number;
  bezierP1: number;
  bezierP2: number;
  useRelu: boolean;
  reluActivationPoint: number;
  
  // Advanced settings
  useCustomRope: boolean;
  disableKvCache: boolean;
  showTokenIds: boolean;
  systemContent: string;
  enableThinking: boolean;
  allowIntrasetTokenVisibility: boolean;
  // Renamed from double-negatives for clarity
  allowIsolatedTokenRemoval: boolean; // was: noPreserveIsolatedTokens
  disableRelativeAttention: boolean; // was: noRelativeAttention
  relativeThreshold: number;
  disableMultiScaleAttention: boolean; // was: noMultiScaleAttention
  disableLciDynamicThreshold: boolean; // was: noLciDynamicThreshold
  disableSigmoidThreshold: boolean; // was: noSigmoidThreshold
  sigmoidSteepness: number;
  completeRemovalMode: string;
  disableKvCacheConsistency: boolean;
  numLayersToUse: number | null;
  // Legacy support
  noPreserveIsolatedTokens?: boolean;
  noRelativeAttention?: boolean;
  noMultiScaleAttention?: boolean;
  noLciDynamicThreshold?: boolean;
  noSigmoidThreshold?: boolean;
};

// Default settings
const defaultSettings: Settings = {
  // General settings
  maxTokens: 100,  // Increased from 50 - gives more room for generation
  selectionThreshold: 0.05,  // Lowered from 0.1 - shows more of TEMPO's parallel exploration
  useRetroactiveRemoval: true,  // Keep enabled - key TEMPO feature
  attentionThreshold: 0.01,  // Good default - balances quality vs exploration
  debugMode: false,
  
  // MCTS settings
  useMcts: false,
  mctsSimulations: 15,  // Increased from 10 - better exploration/exploitation balance
  mctsCPuct: 1.2,  // Slightly increased from 1.0 - encourages more exploration
  mctsDepth: 5,
  
  // Dynamic threshold settings
  dynamicThreshold: false,  // Keep disabled by default - advanced feature
  finalThreshold: 0.3,  // Lowered from 1.0 - more reasonable ending threshold
  bezierP1: 0.1,  // Lowered from 0.2 - smoother initial ramp
  bezierP2: 0.9,  // Increased from 0.8 - maintains exploration longer
  useRelu: false,
  reluActivationPoint: 0.5,
  
  // Advanced settings
  useCustomRope: true,  // Essential for TEMPO - must stay true
  disableKvCache: false,
  showTokenIds: false,
  systemContent: '',
  enableThinking: false,
  allowIntrasetTokenVisibility: false,
  // New clearer names
  allowIsolatedTokenRemoval: false,
  disableRelativeAttention: false,
  relativeThreshold: 0.5,
  disableMultiScaleAttention: false,
  disableLciDynamicThreshold: false,
  disableSigmoidThreshold: false,
  sigmoidSteepness: 10.0,
  completeRemovalMode: 'keep_token',
  disableKvCacheConsistency: false,
  numLayersToUse: null,
};

// Migrate old setting names to new ones
const migrateSettings = (settings: any): Settings => {
  const migrated = { ...settings };
  
  // Migrate double-negative names to clearer ones
  if ('noPreserveIsolatedTokens' in migrated && !('allowIsolatedTokenRemoval' in migrated)) {
    migrated.allowIsolatedTokenRemoval = migrated.noPreserveIsolatedTokens;
  }
  if ('noRelativeAttention' in migrated && !('disableRelativeAttention' in migrated)) {
    migrated.disableRelativeAttention = migrated.noRelativeAttention;
  }
  if ('noMultiScaleAttention' in migrated && !('disableMultiScaleAttention' in migrated)) {
    migrated.disableMultiScaleAttention = migrated.noMultiScaleAttention;
  }
  if ('noLciDynamicThreshold' in migrated && !('disableLciDynamicThreshold' in migrated)) {
    migrated.disableLciDynamicThreshold = migrated.noLciDynamicThreshold;
  }
  if ('noSigmoidThreshold' in migrated && !('disableSigmoidThreshold' in migrated)) {
    migrated.disableSigmoidThreshold = migrated.noSigmoidThreshold;
  }
  
  // Remove old names
  delete migrated.noPreserveIsolatedTokens;
  delete migrated.noRelativeAttention;
  delete migrated.noMultiScaleAttention;
  delete migrated.noLciDynamicThreshold;
  delete migrated.noSigmoidThreshold;
  
  return migrated;
};

// Load settings from localStorage or use defaults
const loadSettings = (): Settings => {
  if (!browser) return defaultSettings;
  
  const savedSettings = localStorage.getItem('tempo-settings');
  if (!savedSettings) return defaultSettings;
  
  try {
    // Merge saved settings with defaults to ensure all fields exist
    const parsed = JSON.parse(savedSettings);
    const migrated = migrateSettings(parsed);
    return { ...defaultSettings, ...migrated };
  } catch (e) {
    console.error('Error parsing saved settings:', e);
    return defaultSettings;
  }
};

// Create the settings store
export const settings = writable<Settings>(loadSettings());

// Save settings to localStorage whenever they change
if (browser) {
  settings.subscribe(value => {
    localStorage.setItem('tempo-settings', JSON.stringify(value));
  });
}

// Helper functions
export const resetSettings = () => {
  settings.set(defaultSettings);
};

export const updateSetting = <K extends keyof Settings>(key: K, value: Settings[K]) => {
  settings.update(s => {
    s[key] = value;
    return s;
  });
};

// Create preset configurations
export const presets = {
  default: (): void => {
    settings.set(defaultSettings);
  },
  
  creative: (): void => {
    settings.update(s => ({
      ...s,
      selectionThreshold: 0.02,  // Very low - maximum exploration
      maxTokens: 150,  // More tokens for creative output
      useRetroactiveRemoval: true,
      attentionThreshold: 0.005,  // Lower threshold for more creative retention
      dynamicThreshold: true,  // Enable dynamic threshold for varied exploration
      finalThreshold: 0.2,  // Low final threshold
      bezierP1: 0.05,  // Very smooth initial ramp
      bezierP2: 0.95,  // Maintain exploration almost to the end
      allowIntrasetTokenVisibility: true,  // Let parallel tokens influence each other
    }));
  },
  
  precise: (): void => {
    settings.update(s => ({
      ...s,
      selectionThreshold: 0.2,  // Higher threshold for more focused generation
      maxTokens: 50,  // Shorter, more controlled output
      useRetroactiveRemoval: true,
      attentionThreshold: 0.02,  // Moderate removal threshold
      dynamicThreshold: false,
      useCustomRope: true,
      disableRelativeAttention: false,
      relativeThreshold: 0.7,  // Higher relative threshold for stricter pruning
      completeRemovalMode: 'keep_token',  // Always keep best token
      disableLciDynamicThreshold: false,
      disableSigmoidThreshold: false,
      sigmoidSteepness: 15.0,  // Sharper decision boundary
    }));
  },
  
  mcts: (): void => {
    settings.update(s => ({
      ...s,
      useMcts: true,
      mctsSimulations: 30,  // More simulations for better tree exploration
      mctsCPuct: 1.5,  // Higher exploration constant
      mctsDepth: 8,  // Deeper search
      selectionThreshold: 0.1,  // Moderate threshold works well with MCTS
      maxTokens: 100,
      useRetroactiveRemoval: false,  // MCTS handles its own pruning
      dynamicThreshold: false,
    }));
  },
  
  exploration: (): void => {
    // Preset to showcase TEMPO's unique parallel exploration features
    settings.update(s => ({
      ...s,
      selectionThreshold: 0.03,  // Low threshold to see many parallel paths
      maxTokens: 80,
      useRetroactiveRemoval: true,
      attentionThreshold: 0.008,  // Lower to keep more interesting paths
      dynamicThreshold: true,  // Show how threshold changes over time
      finalThreshold: 0.15,  // Moderate final threshold
      bezierP1: 0.2,  // Standard curve
      bezierP2: 0.8,
      useCustomRope: true,  // Essential TEMPO feature
      allowIntrasetTokenVisibility: false,  // Keep tokens isolated for clearer visualization
      showTokenIds: false,  // Focus on text, not IDs
      // Enable advanced pruning features
      disableRelativeAttention: false,
      relativeThreshold: 0.4,  // Moderate relative pruning
      disableMultiScaleAttention: false,
      disableLciDynamicThreshold: false,
      disableSigmoidThreshold: false,
      sigmoidSteepness: 8.0,  // Gentler sigmoid transition
    }));
  },
};

// Export the current settings (useful for components that don't use the store directly)
export const getCurrentSettings = (): Settings => get(settings);