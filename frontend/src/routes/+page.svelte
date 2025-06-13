<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Textarea } from '$lib/components/ui/textarea';
  import SimplifiedSettings from '$lib/components/SimplifiedSettings.svelte';
  import OutputVisualization from '$lib/components/OutputVisualization.svelte';
  import GenerationHistory from '$lib/components/GenerationHistory.svelte';
  import { settings, updateSetting, presets } from '$lib/stores/settings';
  import { generationHistory } from '$lib/stores/history';
  import { registerKeyboardShortcuts } from '$lib/utils/keyboard';
  
  let prompt = '';
  let isGenerating = false;
  let error = '';
  let apiResponse: any = null;
  let showHistory = false;
  let promptTextarea: HTMLTextAreaElement;
  let outputVisualization: OutputVisualization;
  
  // Generate text with current settings
  async function generateText() {
    if (!prompt.trim()) {
      error = 'Please enter a prompt';
      return;
    }

    isGenerating = true;
    error = '';

    try {
      const requestBody = {
        prompt,
        max_tokens: $settings.maxTokens,
        selection_threshold: $settings.selectionThreshold,
        use_custom_rope: $settings.useCustomRope,
        use_retroactive_removal: $settings.useRetroactiveRemoval,
        attention_threshold: $settings.attentionThreshold,
        use_mcts: $settings.useMcts,
        mcts_simulations: $settings.mctsSimulations,
        mcts_c_puct: $settings.mctsCPuct,
        mcts_depth: $settings.mctsDepth,
        dynamic_threshold: $settings.dynamicThreshold,
        final_threshold: $settings.finalThreshold,
        bezier_p1: $settings.bezierP1,
        bezier_p2: $settings.bezierP2,
        use_relu: $settings.useRelu,
        relu_activation_point: $settings.reluActivationPoint,
        debug_mode: $settings.debugMode,
        show_token_ids: $settings.showTokenIds,
        system_content: $settings.systemContent || null,
        enable_thinking: $settings.enableThinking,
        disable_kv_cache: $settings.disableKvCache,
        disable_kv_cache_consistency: $settings.disableKvCacheConsistency,
        allow_intraset_token_visibility: $settings.allowIntrasetTokenVisibility,
        no_preserve_isolated_tokens: $settings.allowIsolatedTokenRemoval,
        no_relative_attention: $settings.disableRelativeAttention,
        relative_threshold: $settings.relativeThreshold,
        no_multi_scale_attention: $settings.disableMultiScaleAttention,
        no_lci_dynamic_threshold: $settings.disableLciDynamicThreshold,
        no_sigmoid_threshold: $settings.disableSigmoidThreshold,
        sigmoid_steepness: $settings.sigmoidSteepness,
        complete_removal_mode: $settings.completeRemovalMode,
        num_layers_to_use: $settings.numLayersToUse || null
      };

      const response = await fetch('/api/v2/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        throw new Error(`Request failed: ${response.statusText}`);
      }

      apiResponse = await response.json();
      
      // Save to history
      if (apiResponse) {
        generationHistory.addGeneration(apiResponse, $settings);
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'An error occurred';
    } finally {
      isGenerating = false;
    }
  }
  
  // Preset configurations
  const applyPreset = (presetName: string) => {
    if (typeof presets[presetName] === 'function') {
      presets[presetName]();
    }
  };
  
  // Load generation from history
  function handleHistoryLoad(event: CustomEvent) {
    const { apiResponse: response, settings: historySettings, prompt: historyPrompt } = event.detail;
    
    // Update the UI with the loaded data
    apiResponse = response;
    prompt = historyPrompt;
    
    // Optionally update settings to match the historical generation
    // You might want to ask the user if they want to restore settings
    if (confirm('Also restore the settings used for this generation?')) {
      Object.entries(historySettings).forEach(([key, value]) => {
        updateSetting(key, value);
      });
    }
    
    // Close history panel
    showHistory = false;
  }
  
  // Set up keyboard shortcuts
  onMount(() => {
    const unregister = registerKeyboardShortcuts({
      generate: () => {
        if (prompt.trim() && !isGenerating) {
          generateText();
        }
      },
      clearPrompt: () => {
        prompt = '';
        promptTextarea?.focus();
      },
      toggleHistory: () => {
        showHistory = !showHistory;
      },
      export: () => {
        if (apiResponse && outputVisualization) {
          // Trigger export dialog through OutputVisualization
          const exportButton = document.querySelector('.stats-bar button');
          if (exportButton instanceof HTMLElement) {
            exportButton.click();
          }
        }
      },
      focusPrompt: () => {
        promptTextarea?.focus();
        // Prevent the '/' from being typed
        setTimeout(() => {
          if (promptTextarea && prompt.endsWith('/')) {
            prompt = prompt.slice(0, -1);
          }
        }, 0);
      },
      nextTab: () => {
        // Handle in OutputVisualization component
      },
      prevTab: () => {
        // Handle in OutputVisualization component
      },
      reset: () => {
        applyPreset('default');
      },
      theme: () => {
        document.documentElement.classList.toggle('dark');
      },
      showShortcuts: () => {
        // Show shortcuts dialog (to be implemented)
        alert('Keyboard Shortcuts:\n\n' +
          'Ctrl+Enter - Generate text\n' +
          'Escape - Clear prompt\n' +
          'Ctrl+H - Toggle history\n' +
          'Ctrl+E - Export results\n' +
          '/ - Focus prompt\n' +
          'Alt+R - Reset settings\n' +
          'Alt+T - Toggle theme\n' +
          'Alt+? - Show this help'
        );
      },
      basicTab: () => {},
      mctsTab: () => {},
      advancedTab: () => {}
    });
    
    return unregister;
  });
</script>

<main class="container mx-auto p-4 max-w-7xl">
  <div class="header">
    <div class="header-content">
      <h1 class="title">TEMPO</h1>
      <p class="subtitle">Threshold-Enabled Multipath Parallel Output</p>
    </div>
    <div class="header-actions">
      <Button 
        variant="outline"
        on:click={() => showHistory = !showHistory}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="mr-2">
          <circle cx="12" cy="12" r="10"></circle>
          <polyline points="16 12 12 8 8 12"></polyline>
          <line x1="12" y1="16" x2="12" y2="8"></line>
        </svg>
        History
      </Button>
    </div>
  </div>

  <div class="main-grid">
    <!-- Input & Settings Column -->
    <div class="input-column">
      <Card>
        <CardHeader>
          <CardTitle>Generate Text</CardTitle>
        </CardHeader>
        <CardContent>
          <!-- Prompt Input -->
          <div class="prompt-section">
            <label for="prompt">Prompt</label>
            <Textarea
              id="prompt"
              bind:this={promptTextarea}
              bind:value={prompt}
              placeholder="Enter your prompt here..."
              rows={4}
              class="w-full"
            />
          </div>
          
          <!-- Quick Presets -->
          <div class="presets">
            <button 
              class="preset-button"
              class:active={!$settings.useCustomRope}
              on:click={() => applyPreset('default')}
            >
              Standard
            </button>
            <button 
              class="preset-button"
              class:active={$settings.useCustomRope && $settings.selectionThreshold > 0.1}
              on:click={() => applyPreset('precise')}
            >
              Precise TEMPO
            </button>
            <button 
              class="preset-button"
              class:active={$settings.useCustomRope && $settings.selectionThreshold <= 0.1}
              on:click={() => applyPreset('exploration')}
            >
              Exploratory
            </button>
            <button 
              class="preset-button"
              class:active={$settings.useMcts}
              on:click={() => applyPreset('mcts')}
            >
              MCTS Search
            </button>
          </div>
          
          <!-- Settings -->
          <div class="settings-container">
            <SimplifiedSettings 
              settings={$settings} 
              onUpdate={updateSetting}
            />
          </div>
          
          <!-- Generate Button -->
          <Button
            on:click={generateText}
            disabled={isGenerating || !prompt.trim()}
            size="lg"
            class="w-full generate-button"
          >
            {#if isGenerating}
              <span class="generating">Generating...</span>
            {:else}
              Generate Text
            {/if}
          </Button>
          
          {#if error}
            <div class="error-message">{error}</div>
          {/if}
        </CardContent>
      </Card>
    </div>
    
    <!-- Output Column -->
    <div class="output-column">
      <Card class="h-full">
        <CardHeader>
          <CardTitle>Output</CardTitle>
        </CardHeader>
        <CardContent>
          <OutputVisualization 
            bind:this={outputVisualization}
            {apiResponse} 
            {isGenerating}
            settings={$settings}
          />
        </CardContent>
      </Card>
    </div>
    
    <!-- History Panel -->
    {#if showHistory}
      <div class="history-column">
        <GenerationHistory on:load={handleHistoryLoad} />
      </div>
    {/if}
  </div>
</main>

<style>
  .container {
    min-height: 100vh;
  }
  
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
  }
  
  .header-content {
    text-align: left;
  }
  
  .header-actions {
    display: flex;
    gap: 0.5rem;
  }
  
  .title {
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, hsl(var(--primary)), hsl(var(--primary) / 0.7));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  .subtitle {
    font-size: 1.125rem;
    color: hsl(var(--muted-foreground));
    margin: 0.5rem 0 0 0;
  }
  
  .main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    transition: grid-template-columns 0.3s ease;
  }
  
  .main-grid:has(.history-column) {
    grid-template-columns: 1fr 1fr 350px;
  }
  
  .history-column {
    height: calc(100vh - 200px);
    overflow: hidden;
  }
  
  @media (max-width: 1024px) {
    .main-grid {
      grid-template-columns: 1fr;
    }
    
    .main-grid:has(.history-column) {
      grid-template-columns: 1fr;
    }
    
    .history-column {
      position: fixed;
      right: 0;
      top: 0;
      height: 100vh;
      width: 90%;
      max-width: 400px;
      z-index: 1000;
      box-shadow: -4px 0 12px rgba(0, 0, 0, 0.15);
    }
  }
  
  .prompt-section {
    margin-bottom: 1.5rem;
  }
  
  .prompt-section label {
    display: block;
    font-weight: 500;
    margin-bottom: 0.5rem;
  }
  
  .presets {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
  }
  
  .preset-button {
    padding: 0.5rem 1rem;
    border: 1px solid hsl(var(--border));
    border-radius: 0.375rem;
    background: hsl(var(--card));
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .preset-button:hover {
    background: hsl(var(--muted));
  }
  
  .preset-button.active {
    background: hsl(var(--primary));
    color: hsl(var(--primary-foreground));
    border-color: hsl(var(--primary));
  }
  
  .settings-container {
    margin-bottom: 1.5rem;
  }
  
  .generate-button {
    font-size: 1.125rem;
    font-weight: 600;
  }
  
  .generating {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  
  .generating::before {
    content: '';
    width: 16px;
    height: 16px;
    border: 2px solid currentColor;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  
  .error-message {
    margin-top: 1rem;
    padding: 0.75rem;
    background: hsl(var(--destructive) / 0.1);
    color: hsl(var(--destructive));
    border-radius: 0.375rem;
    font-size: 0.875rem;
  }
  
  .output-column :global(.card) {
    display: flex;
    flex-direction: column;
  }
  
  .output-column :global(.card-content) {
    flex: 1;
  }
</style>