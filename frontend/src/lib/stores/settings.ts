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
  maxTokens: 50,
  selectionThreshold: 0.1,
  useRetroactiveRemoval: true,
  attentionThreshold: 0.01,
  debugMode: false,
  
  // MCTS settings
  useMcts: false,
  mctsSimulations: 10,
  mctsCPuct: 1.0,
  mctsDepth: 5,
  
  // Dynamic threshold settings
  dynamicThreshold: false,
  finalThreshold: 1.0,
  bezierP1: 0.2,
  bezierP2: 0.8,
  useRelu: false,
  reluActivationPoint: 0.5,
  
  // Advanced settings
  useCustomRope: true,
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
      selectionThreshold: 0.05,
      useRetroactiveRemoval: true,
      attentionThreshold: 0.005,
      dynamicThreshold: true,
      finalThreshold: 0.8,
      bezierP1: 0.1,
      bezierP2: 0.9,
    }));
  },
  
  precise: (): void => {
    settings.update(s => ({
      ...s,
      selectionThreshold: 0.3,
      useRetroactiveRemoval: true,
      attentionThreshold: 0.03,
      dynamicThreshold: false,
      useCustomRope: true,
      noRelativeAttention: false,
      relativeThreshold: 0.7,
    }));
  },
  
  mcts: (): void => {
    settings.update(s => ({
      ...s,
      useMcts: true,
      mctsSimulations: 15,
      mctsCPuct: 1.2,
      mctsDepth: 5,
      selectionThreshold: 0.15,
    }));
  },
};

// Export the current settings (useful for components that don't use the store directly)
export const getCurrentSettings = (): Settings => get(settings);