<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type * as d3 from 'd3';
  import { createBarChart } from '$lib/visualizations/barChart';
  import { theme, getAnsiColorMap, toggleTheme } from '$lib/theme';
  import { browser } from '$app/environment';
  import { get } from 'svelte/store';
  import { settings, presets, updateSetting } from '$lib/stores/settings';
  import { registerKeyboardShortcuts, getAllShortcuts } from '$lib/utils/keyboard';

  // Import shadcn components
  import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Textarea } from '$lib/components/ui/textarea';
  import { Slider } from '$lib/components/ui/slider';
  import { Switch } from '$lib/components/ui/switch';
  import { Tabs, TabsContent, TabsList, TabsTrigger } from '$lib/components/ui/tabs';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import { Toggle } from '$lib/components/ui/toggle';
  import SettingSection from '$lib/components/ui/setting-section.svelte';
  import EnhancedPresetCard from '$lib/components/ui/enhanced-preset-card.svelte';
  import RichTooltip from '$lib/components/ui/rich-tooltip.svelte';
  import { getSettingHelp, getCoreSettings } from '$lib/data/settingsHelp';
  import { PRESET_DEFINITIONS, getBeginnerPresets } from '$lib/data/presetDefinitions';

  type AnsiColorMap = Record<string, string>;

  // Adjusted Token type for clarity within Svelte processing
  type ApiToken = {
      token_text: string;
      token_id: number;
      probability: number;
  };

  // Type for data passed to the chart
  type ChartToken = {
      text: string;
      id: number;
      probability: number;
      isPruned: boolean; // Flag to indicate if it was pruned out
  };

  type Step = {
      position: number;
      parallel_tokens: ApiToken[]; // Original candidates from API
      pruned_tokens: ApiToken[];   // Kept tokens from API
  };

  // Type for Token object in the v2 API
  type Token = {
    id: number;
    text: string;
    probability: number;
  };

  // Type for TokenSetData object in the v2 API
  type TokenSetData = {
    position: number;
    original_tokens: Token[];
    pruned_tokens: Token[];
  };

  // Type for TimingInfo in the v2 API
  type TimingInfo = {
    generation_time: number;
    pruning_time: number;
    elapsed_time: number;
  };

  // Type for ModelInfo in the v2 API
  type ModelInfo = {
    model_name: string;
    is_qwen_model: boolean;
    use_custom_rope: boolean;
    device: string;
    model_type?: string;
  };

  // Full API response type from the v2 API
  type ApiResponse = {
    generated_text: string;
    raw_generated_text: string;
    steps?: any[];
    position_to_tokens?: Record<string, string[]>;
    original_parallel_positions?: number[];
    timing: TimingInfo;
    pruning?: any;
    retroactive_pruning?: any;
    model_info: ModelInfo;
    selection_threshold: number;
    max_tokens: number;
    min_steps: number;
    prompt: string;
    had_repetition_loop?: boolean;
    system_content?: string;
    // Legacy format - may still be present in some responses
    token_sets?: [number, [number, number][], [number, number][]][];
    // New v2 format for visualization
    raw_token_data?: TokenSetData[];
  };

  // --- Existing functions (ansiToHtml, stripAnsiCodes) remain the same ---

  function ansiToHtml(text: string): string {
    if (!text) return '';

    const ansiColorMap = getAnsiColorMap($theme === 'dark');

    let result = '';
    let currentColor = '';
    let inAnsi = false;
    let ansiCode = '';

    for (let i = 0; i < text.length; i++) {
      const char = text[i];

      if (char === '\x1b' && text[i + 1] === '[') {
        inAnsi = true;
        ansiCode = '';
        i++; // Skip the '['
        continue;
      }

      if (inAnsi) {
        if (char === 'm') {
          inAnsi = false;
          const color = ansiColorMap[ansiCode as keyof typeof ansiColorMap];
          if (color) {
            if (currentColor) {
              result += `</span>`;
            }
            currentColor = color;
            result += `<span style="color: ${color}">`;
          }
        } else {
          ansiCode += char;
        }
        continue;
      }

      result += char;
    }

    if (currentColor) {
      result += '</span>';
    }

    return result;
  }

  function stripAnsiCodes(text: string): string {
    if (!text) return '';
    return text.replace(/\x1b\[[0-9;]*m/g, '');
  }

  // UI state variables
  let prompt = '';
  let isGenerating = false;
  let error = '';
  let apiResponse: ApiResponse | null = null;
  let chart: { update: (data: ChartToken[]) => void; cleanup: () => void } | null = null;
  let chartContainer: HTMLElement;
  
  // Used for sliders (needs to be an array for ShadCN components)
  let maxTokensSlider = [$settings.maxTokens];
  let selectionThresholdSlider = [$settings.selectionThreshold];
  let attentionThresholdSlider = [$settings.attentionThreshold];
  let mctsSimulationsSlider = [$settings.mctsSimulations];
  let mctsCPuctSlider = [$settings.mctsCPuct];
  let mctsDepthSlider = [$settings.mctsDepth];
  let finalThresholdSlider = [$settings.finalThreshold];
  let bezierP1Slider = [$settings.bezierP1];
  let bezierP2Slider = [$settings.bezierP2];
  let reluActivationPointSlider = [$settings.reluActivationPoint];
  let relativeThresholdSlider = [$settings.relativeThreshold];
  let sigmoidSteepnessSlider = [$settings.sigmoidSteepness];
  
  // Syncing slider values with store
  $: if (maxTokensSlider[0] !== $settings.maxTokens) updateSetting('maxTokens', maxTokensSlider[0]);
  $: if (selectionThresholdSlider[0] !== $settings.selectionThreshold) updateSetting('selectionThreshold', selectionThresholdSlider[0]);
  $: if (attentionThresholdSlider[0] !== $settings.attentionThreshold) updateSetting('attentionThreshold', attentionThresholdSlider[0]);
  $: if (mctsSimulationsSlider[0] !== $settings.mctsSimulations) updateSetting('mctsSimulations', mctsSimulationsSlider[0]);
  $: if (mctsCPuctSlider[0] !== $settings.mctsCPuct) updateSetting('mctsCPuct', mctsCPuctSlider[0]);
  $: if (mctsDepthSlider[0] !== $settings.mctsDepth) updateSetting('mctsDepth', mctsDepthSlider[0]);
  $: if (finalThresholdSlider[0] !== $settings.finalThreshold) updateSetting('finalThreshold', finalThresholdSlider[0]);
  $: if (bezierP1Slider[0] !== $settings.bezierP1) updateSetting('bezierP1', bezierP1Slider[0]);
  $: if (bezierP2Slider[0] !== $settings.bezierP2) updateSetting('bezierP2', bezierP2Slider[0]);
  $: if (reluActivationPointSlider[0] !== $settings.reluActivationPoint) updateSetting('reluActivationPoint', reluActivationPointSlider[0]);
  $: if (relativeThresholdSlider[0] !== $settings.relativeThreshold) updateSetting('relativeThreshold', relativeThresholdSlider[0]);
  $: if (sigmoidSteepnessSlider[0] !== $settings.sigmoidSteepness) updateSetting('sigmoidSteepness', sigmoidSteepnessSlider[0]);
  
  // Other settings that don't need a special slider binding
  let numLayersToUse: number | null = null; // Not persisting this one as it's model-specific
  
  // Additional settings variables
  let useCustomRope = $settings.useCustomRope;
  let disableKvCache = $settings.disableKvCache;
  let disableKvCacheConsistency = $settings.disableKvCacheConsistency;
  let showTokenIds = $settings.showTokenIds;
  let systemContent = $settings.systemContent || '';
  let noRelativeAttention = $settings.noRelativeAttention;
  let noSigmoidThreshold = $settings.noSigmoidThreshold;
  let enableThinking = $settings.enableThinking || false;
  let allowIntrasetTokenVisibility = $settings.allowIntrasetTokenVisibility || false;
  let noLciDynamicThreshold = $settings.noLciDynamicThreshold || false;
  let noMultiScaleAttention = $settings.noMultiScaleAttention || false;
  let noPreserveIsolatedTokens = $settings.noPreserveIsolatedTokens || false;
  let completePruningMode = $settings.completePruningMode || 'prune_all';
  
  // Update settings when these change
  $: updateSetting('useCustomRope', useCustomRope);
  $: updateSetting('disableKvCache', disableKvCache);
  $: updateSetting('disableKvCacheConsistency', disableKvCacheConsistency);
  $: updateSetting('showTokenIds', showTokenIds);
  $: updateSetting('systemContent', systemContent);
  $: updateSetting('noRelativeAttention', noRelativeAttention);
  $: updateSetting('noSigmoidThreshold', noSigmoidThreshold);
  $: updateSetting('enableThinking', enableThinking);
  $: updateSetting('allowIntrasetTokenVisibility', allowIntrasetTokenVisibility);
  $: updateSetting('noLciDynamicThreshold', noLciDynamicThreshold);
  $: updateSetting('noMultiScaleAttention', noMultiScaleAttention);
  $: updateSetting('noPreserveIsolatedTokens', noPreserveIsolatedTokens);
  $: updateSetting('completePruningMode', completePruningMode);

  // Function to generate text
  async function generateText() {
    if (!prompt.trim()) {
      error = 'Please enter a prompt';
      return;
    }

    isGenerating = true;
    error = '';

    try {
      // Input validation with better error messages
      if (maxTokensSlider[0] <= 0) {
        throw new Error('Max tokens must be greater than 0');
      }
      
      if (selectionThresholdSlider[0] < 0 || selectionThresholdSlider[0] > 1) {
        throw new Error('Selection threshold must be between 0 and 1');
      }
      
      if ($settings.useRetroactivePruning && (attentionThresholdSlider[0] < 0 || attentionThresholdSlider[0] > 1)) {
        throw new Error('Attention threshold must be between 0 and 1');
      }
      
      // Create request body using the structured format for API v2
      const requestBody = {
        prompt,
        max_tokens: maxTokensSlider[0],
        selection_threshold: selectionThresholdSlider[0],
        min_steps: 0,
        // Organize parameters into their respective groups
        threshold_settings: {
          use_dynamic_threshold: $settings.dynamicThreshold,
          final_threshold: finalThresholdSlider[0],
          bezier_points: [bezierP1Slider[0], bezierP2Slider[0]],
          use_relu: $settings.useRelu,
          relu_activation_point: reluActivationPointSlider[0]
        },
        mcts_settings: {
          use_mcts: $settings.useMcts,
          simulations: mctsSimulationsSlider[0],
          c_puct: mctsCPuctSlider[0],
          depth: mctsDepthSlider[0]
        },
        pruning_settings: {
          enabled: $settings.useRetroactivePruning,
          attention_threshold: attentionThresholdSlider[0],
          use_relative_attention: !$settings.noRelativeAttention,
          relative_threshold: relativeThresholdSlider[0],
          use_multi_scale_attention: !$settings.noMultiScaleAttention,
          num_layers_to_use: numLayersToUse,
          use_lci_dynamic_threshold: !$settings.noLciDynamicThreshold,
          use_sigmoid_threshold: !$settings.noSigmoidThreshold,
          sigmoid_steepness: sigmoidSteepnessSlider[0],
          pruning_mode: $settings.completePruningMode
        },
        advanced_settings: {
          use_custom_rope: $settings.useCustomRope,
          disable_kv_cache: $settings.disableKvCache,
          disable_kv_cache_consistency: $settings.disableKvCacheConsistency,
          show_token_ids: $settings.showTokenIds,
          system_content: $settings.systemContent || null,
          enable_thinking: $settings.enableThinking,
          allow_intraset_token_visibility: $settings.allowIntrasetTokenVisibility,
          no_preserve_isolated_tokens: $settings.noPreserveIsolatedTokens,
          debug_mode: $settings.debugMode
        }
      };

      // Use the v2 API endpoint with timeout handling
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60-second timeout
      
      try {
        const fetchResponse = await fetch('/api/v2/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
  
        if (!fetchResponse.ok) {
          // Parse error message from API response
          let errorDetail = `HTTP error! status: ${fetchResponse.status}`;
          let statusCode = fetchResponse.status;
          
          try {
            const errorData = await fetchResponse.json();
            
            if (errorData) {
              if (errorData.message) errorDetail = errorData.message;
              else if (errorData.detail) errorDetail = errorData.detail;
              else if (errorData.error) errorDetail = errorData.error;
            }
            
            // Provide user-friendly error messages based on status code
            switch (statusCode) {
              case 400:
                throw new Error(`Invalid request: ${errorDetail}`);
              case 401:
                throw new Error('Authentication required. Please log in again.');
              case 403:
                throw new Error('You do not have permission to perform this action.');
              case 404:
                throw new Error('The requested resource was not found.');
              case 422:
                throw new Error(`Validation error: ${errorDetail}`);
              case 429:
                throw new Error('Rate limit exceeded. Please try again later.');
              case 500:
                throw new Error(`Server error: ${errorDetail}`);
              case 503:
                throw new Error('Service unavailable. The model may be loading or offline.');
              default:
                throw new Error(`Request failed: ${errorDetail}`);
            }
          } catch (parseError) {
            throw new Error(`Request failed: ${errorDetail}`);
          }
        }
  
        const data = await fetchResponse.json() as ApiResponse;
        apiResponse = data;
  
        // Update chart if we have raw token data from the API v2 format
        if (data.raw_token_data && data.raw_token_data.length > 0) {
          // Format the data for the chart
          const chartData = processTokenDataForVisualization(data.raw_token_data);
          updateChart(chartData);
        } else if (data.token_sets && data.token_sets.length > 0) {
          // Fallback to older format if available
          updateChart(data.token_sets);
        }
      } catch (fetchError) {
        if (fetchError.name === 'AbortError') {
          throw new Error('Request timed out. The server may be busy or the model may be taking too long to generate.');
        }
        throw fetchError;
      }
    } catch (e) {
      // Handle specific error types with user-friendly messages
      if (e instanceof Error) {
        if (e.message.includes('Failed to fetch') || e.message.includes('NetworkError')) {
          error = 'Network error. Please check your internet connection and try again.';
        } else if (e.message.includes('timeout') || e.message.includes('timed out')) {
          error = 'The request timed out. The server may be busy or the model may be taking too long to generate.';
        } else {
          error = e.message;
        }
      } else {
        error = 'An unexpected error occurred. Please try again.';
      }
      console.error('Generation error:', e);
    } finally {
      isGenerating = false;
    }
  }
  
  // Helper function to process the new API v2 token data format for visualization
  function processTokenDataForVisualization(rawTokenData) {
    // Convert the new structure to the old format that the chart expects
    return rawTokenData.map(tokenSet => {
      const position = tokenSet.position;
      const originalTokens = tokenSet.original_tokens.map(token => [token.id, token.probability]);
      const prunedTokens = tokenSet.pruned_tokens.map(token => [token.id, token.probability]);
      
      return [position, originalTokens, prunedTokens];
    });
  }

  // Function to update the chart
  function updateChart(tokenSets: [number, [number, number][], [number, number][]][]) {
    if (!chartContainer || !tokenSets.length) return;

    // Clear existing chart
    if (chart) {
      chart.cleanup();
    }

    // Process token sets for chart
    const chartData: ChartToken[] = [];
    tokenSets.forEach(([position, original, pruned]) => {
      // Add original tokens
      original.forEach(([id, prob]) => {
        chartData.push({
          text: `Token ${id}`,
          id,
          probability: prob,
          isPruned: false,
        });
      });

      // Add pruned tokens
      pruned.forEach(([id, prob]) => {
        chartData.push({
          text: `Token ${id}`,
          id,
          probability: prob,
          isPruned: true,
        });
      });
    });

    // Create new chart
    const chartConfig = {
      width: chartContainer.clientWidth,
      height: 400,
      margin: { top: 20, right: 20, bottom: 30, left: 40 },
    };
    chart = createBarChart(chartContainer, chartData, chartConfig);
  }

  // Track active tab for keyboard navigation
  let activeTab = 'basic';
  
  // Interface mode for progressive disclosure
  let interfaceMode = 'beginner'; // 'beginner' | 'intermediate' | 'expert'
  
  // Current active preset
  let activePreset = 'default';
  
  // Get help content for settings
  function getHelp(settingId: string) {
    return getSettingHelp(settingId);
  }
  
  // Apply preset function
  function applyPreset(presetId: string) {
    const preset = PRESET_DEFINITIONS[presetId];
    if (!preset) return;
    
    activePreset = presetId;
    
    // Apply all preset settings
    Object.entries(preset.settings).forEach(([key, value]) => {
      updateSetting(key as any, value);
    });
    
    // Update slider arrays for reactive components
    maxTokensSlider = [preset.settings.maxTokens || $settings.maxTokens];
    selectionThresholdSlider = [preset.settings.selectionThreshold || $settings.selectionThreshold];
    attentionThresholdSlider = [preset.settings.attentionThreshold || $settings.attentionThreshold];
    // ... update other sliders as needed
  }
  
  // Handle window resize and register keyboard shortcuts
  let unregisterShortcuts: () => void;
  
  onMount(() => {
    // Handle window resize
    window.addEventListener('resize', () => {
      if (apiResponse?.token_sets) {
        updateChart(apiResponse.token_sets);
      }
    });
    
    // Register keyboard shortcuts
    unregisterShortcuts = registerKeyboardShortcuts({
      generate: () => {
        if (!isGenerating && prompt.trim()) {
          generateText();
        }
      },
      reset: () => presets.default(),
      theme: () => toggleTheme(),
      basicTab: () => activeTab = 'basic',
      mctsTab: () => activeTab = 'mcts',
      advancedTab: () => activeTab = 'advanced'
    });
  });
  
  onDestroy(() => {
    if (unregisterShortcuts) {
      unregisterShortcuts();
    }
  });
  
  // Get all keyboard shortcuts for help/display
  const keyboardShortcuts = getAllShortcuts();
</script>

<main class="container mx-auto p-4">
  <h1 class="text-3xl font-bold mb-4">TEMPO Text Generation</h1>

  <!-- Responsive layout that stacks on mobile and ensures proper spacing -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-screen-xl mx-auto">
    <!-- Input Section -->
    <Card>
      <CardHeader>
        <CardTitle>Input</CardTitle>
        <CardDescription>Enter your prompt and adjust generation parameters</CardDescription>
      </CardHeader>
      <CardContent class="flex flex-col">
        <!-- Interface Mode Toggle -->
        <div class="mb-6">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-sm font-medium text-gray-700 dark:text-gray-300">Interface Mode</h3>
            <div class="flex rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 p-1">
              <button 
                class="px-3 py-1 text-xs font-medium rounded-md transition-colors {interfaceMode === 'beginner' ? 'bg-white dark:bg-gray-700 text-primary shadow-sm' : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'}"
                on:click={() => interfaceMode = 'beginner'}
              >
                Beginner
              </button>
              <button 
                class="px-3 py-1 text-xs font-medium rounded-md transition-colors {interfaceMode === 'intermediate' ? 'bg-white dark:bg-gray-700 text-primary shadow-sm' : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'}"
                on:click={() => interfaceMode = 'intermediate'}
              >
                Intermediate
              </button>
              <button 
                class="px-3 py-1 text-xs font-medium rounded-md transition-colors {interfaceMode === 'expert' ? 'bg-white dark:bg-gray-700 text-primary shadow-sm' : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'}"
                on:click={() => interfaceMode = 'expert'}
              >
                Expert
              </button>
            </div>
          </div>
        </div>

        <!-- Quick Start Section -->
        {#if interfaceMode === 'beginner' || interfaceMode === 'intermediate'}
          <div class="mb-6">
            <h3 class="text-lg font-semibold mb-3 flex items-center gap-2">
              🚀 Quick Start
            </h3>
            <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
              Choose a preset optimized for your task, then customize if needed.
            </p>
            
            <!-- Preset Cards -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
              {#each getBeginnerPresets() as preset}
                <div data-testid="preset-card">
                  <EnhancedPresetCard
                    name={preset.name}
                    description={preset.description}
                    icon={preset.icon}
                    settings={preset.settings}
                    performance={preset.performance}
                    bestFor={preset.bestFor}
                    technicalNote={preset.technicalNote}
                    difficulty={preset.difficulty}
                    estimatedTime={preset.estimatedTime}
                    isActive={activePreset === preset.id}
                    onApply={() => applyPreset(preset.id)}
                  />
                </div>
              {/each}
            </div>
          </div>
        {/if}

        <!-- Progressive Settings Interface -->
        {#if interfaceMode === 'expert'}
          <!-- Mobile-friendly tab navigation for expert mode -->
          <Tabs bind:value={activeTab} class="w-full">
            <TabsList class="grid w-full grid-cols-3 mb-4">
              <TabsTrigger value="basic" class="text-sm sm:text-base">Basic</TabsTrigger>
              <TabsTrigger value="mcts" class="text-sm sm:text-base">MCTS</TabsTrigger>
              <TabsTrigger value="advanced" class="text-sm sm:text-base">Advanced</TabsTrigger>
            </TabsList>
            <TabsContent value="basic" class="space-y-5">
            <!-- Expert mode keeps the original tabbed interface -->
            <div>
              <label for="prompt" class="block text-sm font-medium mb-2">Prompt</label>
              <Textarea
                id="prompt"
                bind:value={prompt}
                placeholder="Enter your prompt here..."
                rows={4}
                class="w-full min-h-[120px] text-base"
              />
            </div>

            <div>
              <div class="flex items-center gap-1 mb-1">
                <label for="maxTokens" class="block text-sm font-medium">Max Tokens</label>
                {#if getHelp('maxTokens')}
                  <RichTooltip helpContent={getHelp('maxTokens')} />
                {/if}
              </div>
              <Slider
                id="maxTokens"
                bind:value={maxTokensSlider}
                min={1}
                max={500}
                step={1}
              />
              <div class="text-sm text-gray-500 mt-1">{maxTokensSlider[0]} tokens</div>
            </div>

            <div>
              <div class="flex items-center gap-1 mb-1">
                <label for="selectionThreshold" class="block text-sm font-medium">Selection Threshold</label>
                {#if getHelp('selectionThreshold')}
                  <RichTooltip helpContent={getHelp('selectionThreshold')} />
                {/if}
              </div>
              <Slider
                id="selectionThreshold"
                bind:value={selectionThresholdSlider}
                min={0}
                max={1}
                step={0.01}
              />
              <div class="text-sm text-gray-500 mt-1">{selectionThresholdSlider[0].toFixed(2)}</div>
            </div>

            <div class="flex items-center space-x-2">
              <Switch 
                id="useRetroactivePruning" 
                checked={$settings.useRetroactivePruning}
                onCheckedChange={(checked) => updateSetting('useRetroactivePruning', checked)}
              />
              <label for="useRetroactivePruning" class="text-sm font-medium">Use Retroactive Pruning</label>
              {#if getHelp('useRetroactivePruning')}
                <RichTooltip helpContent={getHelp('useRetroactivePruning')} />
              {/if}
            </div>

            {#if $settings.useRetroactivePruning}
            <div class="pl-4 mt-2 space-y-2 border-l-2 border-gray-200">
              <div class="flex items-center gap-1 mb-1">
                <label for="attentionThreshold" class="block text-sm font-medium">Attention Threshold</label>
                {#if getHelp('attentionThreshold')}
                  <RichTooltip helpContent={getHelp('attentionThreshold')} />
                {/if}
              </div>
              <Slider
                id="attentionThreshold"
                bind:value={attentionThresholdSlider}
                min={0}
                max={0.1}
                step={0.001}
              />
              <div class="text-sm text-gray-500 mt-1">{attentionThresholdSlider[0].toFixed(3)}</div>
            </div>
            {/if}

            <div class="flex items-center space-x-2">
              <Switch 
                id="debugMode" 
                checked={$settings.debugMode}
                onCheckedChange={(checked) => updateSetting('debugMode', checked)}
              />
              <label for="debugMode" class="text-sm font-medium">Debug Mode</label>
              {#if getHelp('debugMode')}
                <RichTooltip helpContent={getHelp('debugMode')} />
              {/if}
            </div>
            
            <div class="mt-6 pt-4 border-t border-border">
              <h3 class="text-sm font-medium mb-3">Legacy Presets</h3>
              <div class="flex flex-wrap gap-2 justify-start">
                <Button variant="outline" size="sm" on:click={() => presets.default()} class="flex-grow sm:flex-grow-0">
                  Default
                </Button>
                <Button variant="outline" size="sm" on:click={() => presets.creative()} class="flex-grow sm:flex-grow-0">
                  Creative
                </Button>
                <Button variant="outline" size="sm" on:click={() => presets.precise()} class="flex-grow sm:flex-grow-0">
                  Precise
                </Button>
                <Button variant="outline" size="sm" on:click={() => presets.mcts()} class="flex-grow sm:flex-grow-0">
                  MCTS
                </Button>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="mcts" class="space-y-4">
            <div class="flex items-center space-x-2">
              <Switch 
                id="useMcts" 
                checked={$settings.useMcts}
                onCheckedChange={(checked) => updateSetting('useMcts', checked)}
              />
              <label for="useMcts" class="text-sm font-medium">Use MCTS</label>
              {#if getHelp('useMcts')}
                <RichTooltip helpContent={getHelp('useMcts')} />
              {/if}
            </div>

            {#if $settings.useMcts}
            <div class="pl-4 mt-2 space-y-4 border-l-2 border-gray-200">
              <div>
                <div class="flex items-center gap-1 mb-1">
                  <label for="mctsSimulations" class="block text-sm font-medium">MCTS Simulations</label>
                  {#if getHelp('mctsSimulations')}
                    <RichTooltip helpContent={getHelp('mctsSimulations')} />
                  {/if}
                </div>
                <Slider
                  id="mctsSimulations"
                  bind:value={mctsSimulationsSlider}
                  min={5}
                  max={200}
                  step={5}
                />
                <div class="text-sm text-gray-500 mt-1">{mctsSimulationsSlider[0]} simulations</div>
              </div>

              <div>
                <div class="flex items-center gap-1 mb-1">
                  <label for="mctsCPuct" class="block text-sm font-medium">MCTS C_PUCT</label>
                  {#if getHelp('mctsCPuct')}
                    <RichTooltip helpContent={getHelp('mctsCPuct')} />
                  {/if}
                </div>
                <Slider
                  id="mctsCPuct"
                  bind:value={mctsCPuctSlider}
                  min={0.1}
                  max={5.0}
                  step={0.1}
                />
                <div class="text-sm text-gray-500 mt-1">{mctsCPuctSlider[0].toFixed(1)}</div>
              </div>

              <div>
                <div class="flex items-center gap-1 mb-1">
                  <label for="mctsDepth" class="block text-sm font-medium">MCTS Depth</label>
                  {#if getHelp('mctsDepth')}
                    <RichTooltip helpContent={getHelp('mctsDepth')} />
                  {/if}
                </div>
                <Slider
                  id="mctsDepth"
                  bind:value={mctsDepthSlider}
                  min={2}
                  max={20}
                  step={1}
                />
                <div class="text-sm text-gray-500 mt-1">{mctsDepthSlider[0]} steps</div>
              </div>
            </div>
            {/if}

            <div class="flex items-center space-x-2">
              <Switch 
                id="dynamicThreshold" 
                checked={$settings.dynamicThreshold}
                onCheckedChange={(checked) => updateSetting('dynamicThreshold', checked)}
              />
              <label for="dynamicThreshold" class="text-sm font-medium">Use Dynamic Threshold</label>
              {#if getHelp('dynamicThreshold')}
                <RichTooltip helpContent={getHelp('dynamicThreshold')} />
              {/if}
            </div>

            {#if $settings.dynamicThreshold}
            <div class="pl-4 mt-2 space-y-4 border-l-2 border-gray-200">
              <div>
                <div class="flex items-center gap-1 mb-1">
                  <label for="finalThreshold" class="block text-sm font-medium">Final Threshold</label>
                  {#if getHelp('finalThreshold')}
                    <RichTooltip helpContent={getHelp('finalThreshold')} />
                  {/if}
                </div>
                <Slider
                  id="finalThreshold"
                  bind:value={finalThresholdSlider}
                  min={0.01}
                  max={0.5}
                  step={0.01}
                />
                <div class="text-sm text-gray-500 mt-1">{finalThresholdSlider[0].toFixed(2)}</div>
              </div>

              <div class="flex items-center space-x-2">
                <Switch 
                  id="useRelu" 
                  checked={$settings.useRelu}
                  onCheckedChange={(checked) => updateSetting('useRelu', checked)}
                />
                <label for="useRelu" class="text-sm font-medium">Use ReLU Transition</label>
              </div>

              {#if $settings.useRelu}
              <div>
                <label for="reluActivationPoint" class="block text-sm font-medium mb-1">ReLU Activation Point</label>
                <Slider
                  id="reluActivationPoint"
                  bind:value={reluActivationPointSlider}
                  min={0}
                  max={1}
                  step={0.01}
                />
                <div class="text-sm text-gray-500 mt-1">{reluActivationPointSlider[0].toFixed(2)}</div>
              </div>
              {:else}
              <div>
                <label for="bezierP1" class="block text-sm font-medium mb-1">Bezier P1</label>
                <Slider
                  id="bezierP1"
                  bind:value={bezierP1Slider}
                  min={0}
                  max={1}
                  step={0.01}
                />
                <div class="text-sm text-gray-500 mt-1">{bezierP1Slider[0].toFixed(2)}</div>
              </div>
              <div>
                <label for="bezierP2" class="block text-sm font-medium mb-1">Bezier P2</label>
                <Slider
                  id="bezierP2"
                  bind:value={bezierP2Slider}
                  min={0}
                  max={1}
                  step={0.01}
                />
                <div class="text-sm text-gray-500 mt-1">{bezierP2Slider[0].toFixed(2)}</div>
              </div>
              {/if}
            </div>
            {/if}
          </TabsContent>

          <TabsContent value="advanced" class="space-y-4">
            <div class="space-y-4">
              <div class="flex items-center space-x-2">
                <Switch id="useCustomRope" bind:checked={useCustomRope} />
                <label for="useCustomRope" class="text-sm font-medium">Use Custom RoPE</label>
                {#if getHelp('useCustomRope')}
                  <RichTooltip helpContent={getHelp('useCustomRope')} />
                {/if}
              </div>

              <div class="flex items-center space-x-2">
                <Switch id="disableKvCache" bind:checked={disableKvCache} />
                <label for="disableKvCache" class="text-sm font-medium">Disable KV Cache</label>
                {#if getHelp('disableKvCache')}
                  <RichTooltip helpContent={getHelp('disableKvCache')} />
                {/if}
              </div>

              <div class="flex items-center space-x-2">
                <Switch id="showTokenIds" bind:checked={showTokenIds} />
                <label for="showTokenIds" class="text-sm font-medium">Show Token IDs</label>
                {#if getHelp('showTokenIds')}
                  <RichTooltip helpContent={getHelp('showTokenIds')} />
                {/if}
              </div>

              <div>
                <div class="flex items-center gap-1 mb-1">
                  <label for="systemContent" class="block text-sm font-medium">System Content</label>
                  {#if getHelp('systemContent')}
                    <RichTooltip helpContent={getHelp('systemContent')} />
                  {/if}
                </div>
                <Textarea
                  id="systemContent"
                  bind:value={systemContent}
                  placeholder="Enter system content..."
                  rows={2}
                />
              </div>

              <div class="flex items-center space-x-2">
                <Switch id="enableThinking" bind:checked={enableThinking} />
                <label for="enableThinking" class="text-sm font-medium">Enable Thinking</label>
                {#if getHelp('enableThinking')}
                  <RichTooltip helpContent={getHelp('enableThinking')} />
                {/if}
              </div>

              <div class="flex items-center space-x-2">
                <Switch id="allowIntrasetTokenVisibility" bind:checked={allowIntrasetTokenVisibility} />
                <label for="allowIntrasetTokenVisibility" class="text-sm font-medium">Allow Intraset Token Visibility</label>
              </div>

              <div class="flex items-center space-x-2">
                <Switch id="noPreserveIsolatedTokens" bind:checked={noPreserveIsolatedTokens} />
                <label for="noPreserveIsolatedTokens" class="text-sm font-medium">No Preserve Isolated Tokens</label>
              </div>

              <div class="flex items-center space-x-2">
                <Switch id="noRelativeAttention" bind:checked={noRelativeAttention} />
                <label for="noRelativeAttention" class="text-sm font-medium">No Relative Attention</label>
              </div>

              {#if !noRelativeAttention}
              <div>
                <label for="relativeThreshold" class="block text-sm font-medium mb-1">Relative Threshold</label>
                <Slider
                  id="relativeThreshold"
                  bind:value={relativeThresholdSlider}
                  min={0}
                  max={1}
                  step={0.01}
                />
                <div class="text-sm text-gray-500 mt-1">{relativeThresholdSlider[0].toFixed(2)}</div>
              </div>
              {/if}

              <div class="flex items-center space-x-2">
                <Switch id="noMultiScaleAttention" bind:checked={noMultiScaleAttention} />
                <label for="noMultiScaleAttention" class="text-sm font-medium">No Multi-Scale Attention</label>
              </div>

              <div>
                <label for="numLayersToUse" class="block text-sm font-medium mb-1">Number of Layers to Use</label>
                <Input
                  id="numLayersToUse"
                  type="number"
                  bind:value={numLayersToUse}
                  placeholder="Leave empty for all layers"
                  min={1}
                />
              </div>

              <div class="flex items-center space-x-2">
                <Switch id="noLciDynamicThreshold" bind:checked={noLciDynamicThreshold} />
                <label for="noLciDynamicThreshold" class="text-sm font-medium">No LCI Dynamic Threshold</label>
              </div>

              <div class="flex items-center space-x-2">
                <Switch id="noSigmoidThreshold" bind:checked={noSigmoidThreshold} />
                <label for="noSigmoidThreshold" class="text-sm font-medium">No Sigmoid Threshold</label>
              </div>

              {#if !noSigmoidThreshold}
              <div>
                <label for="sigmoidSteepness" class="block text-sm font-medium mb-1">Sigmoid Steepness</label>
                <Slider
                  id="sigmoidSteepness"
                  bind:value={sigmoidSteepnessSlider}
                  min={1}
                  max={20}
                  step={0.1}
                />
                <div class="text-sm text-gray-500 mt-1">{sigmoidSteepnessSlider[0].toFixed(1)}</div>
              </div>
              {/if}

              <div>
                <label for="completePruningMode" class="block text-sm font-medium mb-1">Complete Pruning Mode</label>
                <select
                  id="completePruningMode"
                  bind:value={completePruningMode}
                  class="w-full p-2 border rounded"
                >
                  <option value="keep_token">Keep Token</option>
                  <option value="keep_unattended">Keep Unattended</option>
                  <option value="remove_position">Remove Position</option>
                </select>
              </div>

              <div class="flex items-center space-x-2">
                <Switch id="disableKvCacheConsistency" bind:checked={disableKvCacheConsistency} />
                <label for="disableKvCacheConsistency" class="text-sm font-medium">Disable KV Cache Consistency</label>
              </div>
            </div>
          </TabsContent>
          </Tabs>
        {:else}
          <!-- Progressive disclosure for beginner/intermediate -->
          <div class="space-y-4">

            <!-- Prompt Input - Always Visible -->
            <SettingSection 
              title="Prompt & Generation" 
              description="Enter your prompt and control basic generation parameters"
              icon="💬"
              category="core"
              importance="essential"
              defaultOpen={true}
            >
              <div class="space-y-4">
                <div>
                  <label for="prompt" class="block text-sm font-medium mb-2">Prompt</label>
                  <Textarea
                    id="prompt"
                    bind:value={prompt}
                    placeholder="Enter your prompt here..."
                    rows={4}
                    class="w-full min-h-[120px] text-base"
                  />
                </div>

                <div>
                  <div class="flex items-center gap-1 mb-1">
                    <label for="maxTokens" class="block text-sm font-medium">Max Tokens</label>
                    {#if getHelp('maxTokens')}
                      <RichTooltip helpContent={getHelp('maxTokens')} />
                    {/if}
                  </div>
                  <Slider
                    id="maxTokens"
                    bind:value={maxTokensSlider}
                    min={1}
                    max={500}
                    step={1}
                  />
                  <div class="text-sm text-gray-500 mt-1">{maxTokensSlider[0]} tokens (~{Math.round(maxTokensSlider[0] * 0.75)} words)</div>
                </div>

                <div>
                  <div class="flex items-center gap-1 mb-1">
                    <label for="selectionThreshold" class="block text-sm font-medium">Selection Threshold</label>
                    {#if getHelp('selectionThreshold')}
                      <RichTooltip helpContent={getHelp('selectionThreshold')} />
                    {/if}
                  </div>
                  <Slider
                    id="selectionThreshold"
                    bind:value={selectionThresholdSlider}
                    min={0.01}
                    max={0.5}
                    step={0.01}
                  />
                  <div class="text-sm text-gray-500 mt-1">{selectionThresholdSlider[0].toFixed(2)}</div>
                </div>
              </div>
            </SettingSection>

            <!-- Core Settings for Intermediate+ -->
            {#if interfaceMode !== 'beginner'}
              <SettingSection 
                title="Pruning & Refinement" 
                description="Control how TEMPO refines and improves generated content"
                icon="✂️"
                category="pruning"
                importance="important"
                defaultOpen={interfaceMode === 'intermediate'}
              >
                <div class="space-y-4">
                  <div class="flex items-center space-x-2">
                    <Switch 
                      id="useRetroactivePruning" 
                      checked={$settings.useRetroactivePruning}
                      onCheckedChange={(checked) => updateSetting('useRetroactivePruning', checked)}
                    />
                    <label for="useRetroactivePruning" class="text-sm font-medium">Use Retroactive Pruning</label>
                    {#if getHelp('useRetroactivePruning')}
                      <RichTooltip helpContent={getHelp('useRetroactivePruning')} />
                    {/if}
                  </div>

                  {#if $settings.useRetroactivePruning}
                    <div class="pl-4 mt-2 space-y-2 border-l-2 border-gray-200 dark:border-gray-700">
                      <div class="flex items-center gap-1 mb-1">
                        <label for="attentionThreshold" class="block text-sm font-medium">Attention Threshold</label>
                        {#if getHelp('attentionThreshold')}
                          <RichTooltip helpContent={getHelp('attentionThreshold')} />
                        {/if}
                      </div>
                      <Slider
                        id="attentionThreshold"
                        bind:value={attentionThresholdSlider}
                        min={0.001}
                        max={0.1}
                        step={0.001}
                      />
                      <div class="text-sm text-gray-500 mt-1">{attentionThresholdSlider[0].toFixed(3)}</div>
                    </div>
                  {/if}
                </div>
              </SettingSection>
            {/if}
          </div>
        {/if}

        <Button
          on:click={generateText}
          disabled={isGenerating}
          class="w-full mt-6 py-6 text-lg relative"
          variant={isGenerating ? "outline" : "default"}
          data-testid="generate-button"
        >
          {#if isGenerating}
            <div class="flex items-center justify-center gap-2">
              <svg class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span>Generating...</span>
            </div>
          {:else}
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd" />
            </svg>
            Generate
          {/if}
        </Button>
      </CardContent>
    </Card>

    <!-- Output Section -->
    <Card>
      <CardHeader>
        <CardTitle>Output</CardTitle>
        <CardDescription>Generated text and visualization</CardDescription>
      </CardHeader>
      <CardContent>
        {#if error}
          <div class="text-red-500 mb-4 p-3 border border-red-200 rounded bg-red-50 dark:bg-red-900/20 dark:border-red-800">
            <div class="flex items-start gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-red-500 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
              </svg>
              <div>{error}</div>
            </div>
          </div>
        {/if}

        {#if isGenerating}
          <div class="space-y-4 animate-pulse">
            <div>
              <h3 class="text-lg font-medium mb-2">Generating...</h3>
              <div class="bg-gray-100 dark:bg-gray-800 p-4 rounded h-48 flex flex-col items-center justify-center">
                <svg class="animate-spin h-10 w-10 mb-4 text-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <div class="text-sm text-gray-500">
                  Processing your request. This may take a moment depending on the model size and settings.
                </div>
              </div>
            </div>
            
            <!-- Loading placeholders for other sections -->
            <div>
              <h3 class="text-lg font-medium mb-2">Model Information</h3>
              <div class="grid grid-cols-2 gap-2">
                <div class="h-6 bg-gray-200 dark:bg-gray-700 rounded"></div>
                <div class="h-6 bg-gray-200 dark:bg-gray-700 rounded"></div>
                <div class="h-6 bg-gray-200 dark:bg-gray-700 rounded"></div>
                <div class="h-6 bg-gray-200 dark:bg-gray-700 rounded"></div>
              </div>
            </div>
            
            <div>
              <h3 class="text-lg font-medium mb-2">Visualization</h3>
              <div class="h-[300px] bg-gray-200 dark:bg-gray-700 rounded"></div>
            </div>
          </div>
        {:else if apiResponse}
          <div class="space-y-4">
            <div>
              <h3 class="text-lg font-medium mb-2">Generated Text</h3>
              <div class="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-auto max-h-[300px]" data-testid="generated-text">
                {@html ansiToHtml(apiResponse.generated_text)}
              </div>
            </div>

            <!-- Model info from v2 API -->
            {#if apiResponse.model_info}
              <div>
                <h3 class="text-lg font-medium mb-2">Model Information</h3>
                <div class="text-sm grid grid-cols-2 gap-2">
                  <div>Model: <span class="font-semibold">{apiResponse.model_info.model_name}</span></div>
                  <div>Device: <span class="font-semibold">{apiResponse.model_info.device}</span></div>
                  {#if apiResponse.model_info.model_type}
                    <div>Type: <span class="font-semibold">{apiResponse.model_info.model_type}</span></div>
                  {/if}
                  <div>Custom RoPE: <span class="font-semibold">{apiResponse.model_info.use_custom_rope ? 'Enabled' : 'Disabled'}</span></div>
                </div>
              </div>
            {/if}

            {#if apiResponse.timing}
              <div>
                <h3 class="text-lg font-medium mb-2">Timing</h3>
                <div class="text-sm grid grid-cols-2 gap-2">
                  <div>Generation Time: <span class="font-semibold">{apiResponse.timing.generation_time.toFixed(2)}s</span></div>
                  <div>Pruning Time: <span class="font-semibold">{apiResponse.timing.pruning_time.toFixed(2)}s</span></div>
                  <div>Total Time: <span class="font-semibold">{apiResponse.timing.elapsed_time.toFixed(2)}s</span></div>
                </div>
              </div>
            {/if}

            {#if apiResponse.retroactive_pruning}
              <div>
                <h3 class="text-lg font-medium mb-2">Pruning Settings</h3>
                <div class="text-sm grid grid-cols-2 gap-2">
                  <div>Attention Threshold: <span class="font-semibold">{apiResponse.retroactive_pruning.attention_threshold.toFixed(3)}</span></div>
                  <div>Pruning Mode: <span class="font-semibold">{apiResponse.retroactive_pruning.pruning_mode}</span></div>
                  <div>Relative Attention: <span class="font-semibold">{apiResponse.retroactive_pruning.use_relative_attention ? 'Enabled' : 'Disabled'}</span></div>
                  {#if apiResponse.retroactive_pruning.use_relative_attention}
                    <div>Relative Threshold: <span class="font-semibold">{apiResponse.retroactive_pruning.relative_threshold.toFixed(2)}</span></div>
                  {/if}
                </div>
              </div>
            {/if}

            <!-- Token visualization - works with both v1 and v2 API formats -->
            {#if (apiResponse.raw_token_data && apiResponse.raw_token_data.length > 0) || (apiResponse.token_sets && apiResponse.token_sets.length > 0)}
              <div>
                <h3 class="text-lg font-medium mb-2">Token Visualization</h3>
                <div
                  bind:this={chartContainer}
                  class="w-full h-[400px] chart-container"
                  data-testid="visualization-chart"
                />
              </div>
            {/if}
          </div>
        {/if}
      </CardContent>
    </Card>
  </div>
</main>

<style>
  :global(.code-output) {
    scrollbar-width: thin;
    scrollbar-color: hsl(var(--muted)) transparent; /* Use theme variable */
  }
  :global(.code-output::-webkit-scrollbar) { width: 8px; height: 8px; }
  :global(.code-output::-webkit-scrollbar-track) { background: transparent; }
  :global(.code-output::-webkit-scrollbar-thumb) {
    background-color: hsl(var(--muted)); /* Use theme variable */
    border-radius: 4px;
    border: 2px solid hsl(var(--background)); /* Match background */
  }
</style>