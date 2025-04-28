<script lang="ts">
  import { onMount } from 'svelte';
  import type * as d3 from 'd3';
  import { createBarChart } from '$lib/visualizations/barChart';
  import { theme, getAnsiColorMap } from '$lib/theme';
  import { browser } from '$app/environment';
  import { get } from 'svelte/store';
  
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
  
  type AnsiColorMap = Record<string, string>;
  type Token = {
    token_text: string;
    token_id: number;
    probability: number;
    isPruned?: boolean;
  };
  
  type Step = {
    generated_text: string;
    parallel_tokens: Token[];
    pruned_tokens: Token[];
  };
  
  // Function to convert ANSI color codes to HTML styling
  function ansiToHtml(text: string): string {
    if (!text) return '';
    
    const ansiColorMap = getAnsiColorMap($theme === 'dark');
    
    let result = '';
    let parts = text.split(/(\u001b\[\d+m|\[\d+m)/);
    let currentColor: string | null = null;
    
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      if (part.match(/\u001b\[(\d+)m|\[(\d+)m/)) {
        const colorMatch = part.match(/\u001b\[(\d+)m/) || part.match(/\[(\d+)m/);
        const colorCode = colorMatch ? colorMatch[1] : '0';
        
        if (colorCode === '0') {
          // Reset color
          if (currentColor) {
            result += '</span>';
            currentColor = null;
          }
        } else if (ansiColorMap[colorCode as keyof typeof ansiColorMap]) {
          // Close previous color span if there was one
          if (currentColor) {
            result += '</span>';
          }
          
          currentColor = ansiColorMap[colorCode as keyof typeof ansiColorMap];
          result += `<span style="color: ${currentColor};">`;
        }
      } else {
        result += part;
      }
    }
    
    // Close any open span
    if (currentColor) {
      result += '</span>';
    }
    
    return result;
  }
  
  // Function to strip ANSI color codes from text (fallback for token display)
  function stripAnsiCodes(text: string): string {
    if (!text) return '';
    // This regex pattern matches ANSI escape codes
    return text.replace(/\u001b\[\d+m|\[\d+m/g, '');
  }
  
  // INVARIANT: All parameters must have valid default values
  let prompt = "Is a hotdog a sandwich?";
  let selectionThreshold = [0.05];
  let maxTokens = 30;
  let minSteps = 0;
  let usePruning = true;
  let useDiversityPruning = true;
  let useRetroactivePruning = true;
  let diversityClusters = 3;
  let retroactivePruningThreshold = [0.01];
  
  // New pruning strategy and parameters
  let pruningStrategy = "coherence";
  let coherenceThreshold = 0.3;
  let diversitySteps = 5;
  
  // Dynamic threshold parameters
  let useDynamicThreshold = true;
  let finalThreshold = [1.0];
  let useRelu = false;
  let reluActivationPoint = [0.5];
  let bezierP1 = [0.2];
  let bezierP2 = [0.8];
  
  // Advanced retroactive pruning options
  let noRelativeAttention = false;
  let relativeThreshold = [0.5];
  let noMultiScaleAttention = false;
  let numLayersToUse = "";  // Empty string means null/None
  let noLciDynamicThreshold = false;
  let noSigmoidThreshold = false;
  let sigmoidSteepness = 10.0;
  let completePruningMode = "keep_token";
  
  // Additional parameters
  let noPreserveIsolatedTokens = false;
  let disableKvCacheConsistency = false;
  
  // MCTS parameters
  let useMCTS = false;
  let mctsSimulations = 10;
  let mctsCPuct = 1.0;
  let mctsDepth = 5;
  
  // Additional parameters
  let enableThinking = false;
  let allowIntrasetTokenVisibility = false;
  let useCustomRope = true;
  let disableKVCache = false;
  
  // State variables
  let isGenerating = false;
  let error: string | null = null;
  let generatedText = "";
  let steps: Step[] = [];
  let currentStep = [0];
  
  // Chart reference
  let chartContainer: HTMLElement;
  let chart: any;
  
  // UI State
  let showAdvancedSettings = false;
  
  // INVARIANT: Parameters must always be within these bounds
  $: {
    // Enforce threshold bounds
    selectionThreshold[0] = Math.max(0, Math.min(1, selectionThreshold[0]));
    
    // Enforce token bounds
    maxTokens = Math.max(1, Math.min(600, maxTokens));
    
    // Enforce step bounds
    minSteps = Math.max(0, minSteps);
    
    // Enforce threshold bounds
    retroactivePruningThreshold[0] = Math.max(0, Math.min(1, retroactivePruningThreshold[0]));
    
    // Enforce cluster bounds
    diversityClusters = Math.max(1, Math.min(10, diversityClusters));
    
    // Enforce MCTS bounds
    mctsSimulations = Math.max(1, Math.min(50, mctsSimulations));
    mctsCPuct = Math.max(0.1, Math.min(5.0, mctsCPuct));
    mctsDepth = Math.max(1, Math.min(10, mctsDepth));
  }
  
  // When theme changes, update the generated text to use the right colors
  $: if (generatedText && $theme) {
    generatedText = ansiToHtml(steps.length > 0 ? steps[steps.length - 1].generated_text : "");
  }
  
  let resultData = {};
  let chartData = null;
  let selectedViewMode = "default";
  let selectedPromptIndex = 0;
  
  onMount(() => {
    initializeVisualization();
  });
  
  // Update the visualization when steps or currentStep changes
  $: if (chart && steps.length > 0 && steps[currentStep[0]]) {
    try {
      const stepData = steps[currentStep[0]];
      
      // INVARIANT: Step data must contain parallel_tokens and pruned_tokens
      if (stepData.parallel_tokens && stepData.pruned_tokens) {
        // Create data for visualization
        const originalTokens = stepData.parallel_tokens.map(t => ({
          text: t.token_text,
          id: t.token_id,
          probability: t.probability,
          isPruned: false
        }));
        
        // Get IDs of tokens that were kept after pruning
        const keptIds = new Set(stepData.pruned_tokens.map(t => t.token_id));
        
        // Mark pruned tokens
        originalTokens.forEach(token => {
          token.isPruned = !keptIds.has(token.id);
        });
        
        // Sort by probability descending
        originalTokens.sort((a, b) => b.probability - a.probability);
        
        // Update chart with new data
        chart.update(originalTokens);
      } else {
        console.error("Invalid step data format", stepData);
      }
    } catch (err) {
      console.error("Error updating visualization:", err);
    }
  }
  
  async function generateText() {
    try {
      // Reset state
      isGenerating = true;
      error = null;
      generatedText = "";
      steps = [];
      currentStep = [0];
      
      // INVARIANT: API request parameters must be valid
      const requestData = {
        prompt: prompt.trim(),
        threshold: selectionThreshold[0].toString(),
        max_tokens: maxTokens.toString(),
        min_steps: minSteps.toString(),
        use_pruning: usePruning,
        use_diversity_pruning: useDiversityPruning,
        use_retroactive_pruning: useRetroactivePruning,
        pruning_strategy: pruningStrategy,
        coherence_threshold: coherenceThreshold.toString(),
        diversity_clusters: diversityClusters.toString(),
        diversity_steps: diversitySteps.toString(),
        attention_threshold: retroactivePruningThreshold[0].toString(),
        
        // Dynamic threshold parameters
        dynamic_threshold: useDynamicThreshold,
        final_threshold: finalThreshold[0].toString(),
        use_relu: useRelu,
        relu_activation_point: reluActivationPoint[0].toString(),
        bezier_points: [bezierP1[0], bezierP2[0]],
        
        // Advanced retroactive pruning parameters
        no_relative_attention: noRelativeAttention,
        relative_threshold: relativeThreshold[0].toString(),
        no_multi_scale_attention: noMultiScaleAttention,
        num_layers_to_use: numLayersToUse ? parseInt(numLayersToUse) : null,
        no_lci_dynamic_threshold: noLciDynamicThreshold,
        no_sigmoid_threshold: noSigmoidThreshold,
        sigmoid_steepness: sigmoidSteepness.toString(),
        complete_pruning_mode: completePruningMode,
        
        // Additional parameters
        no_preserve_isolated_tokens: noPreserveIsolatedTokens,
        disable_kv_cache_consistency: disableKvCacheConsistency,
        
        // MCTS parameters
        use_mcts: useMCTS,
        mcts_simulations: mctsSimulations.toString(),
        mcts_c_puct: mctsCPuct.toString(),
        mcts_depth: mctsDepth.toString(),
        
        // Additional parameters
        enable_thinking: enableThinking,
        allow_intraset_token_visibility: allowIntrasetTokenVisibility,
        use_custom_rope: useCustomRope,
        disable_kv_cache: disableKVCache
      };
      
      // Validate request before sending
      if (!requestData.prompt) {
        throw new Error("Prompt cannot be empty");
      }
      
      // Send request to API
      const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        const errorMsg = errorData.detail || "Generation failed";
        
        // Check if the error is related to threshold
        if (errorMsg.includes("No tokens above threshold")) {
          throw new Error(`Generation failed: No tokens above threshold. Please try lowering the threshold value further (current: ${selectionThreshold[0]}). Try 0.01 or lower.`);
        } else {
          throw new Error(errorMsg);
        }
      }
      
      const data = await response.json();
      
      // INVARIANT: Response must contain generated_text and steps
      if (!data.generated_text || !Array.isArray(data.steps)) {
        throw new Error("Invalid response format from API");
      }
      
      // Update state with results
      const processedText = ansiToHtml(data.generated_text);
      generatedText = processedText;
      steps = data.steps;
      
      // Clean token text in each step - just strip for tokens since they're small
      steps = steps.map(step => ({
        ...step,
        parallel_tokens: step.parallel_tokens.map(token => ({
          ...token,
          token_text: stripAnsiCodes(token.token_text)
        })),
        pruned_tokens: step.pruned_tokens.map(token => ({
          ...token,
          token_text: stripAnsiCodes(token.token_text)
        }))
      }));
      
      // Handle case where we got data but no steps
      if (steps.length === 0) {
        error = "No generation steps returned. Try lowering the threshold value.";
      } else {
        currentStep = [0];
        
        // Update visualization with first step
        if (chart) {
          const firstStep = steps[0];
          const tokens = firstStep.parallel_tokens.map(t => ({
            text: t.token_text,
            id: t.token_id,
            probability: t.probability,
            isPruned: !firstStep.pruned_tokens.some(pt => pt.token_id === t.token_id)
          }));
          
          tokens.sort((a, b) => b.probability - a.probability);
          chart.update(tokens);
        }
      }
      
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : "An error occurred during generation";
      error = errorMessage;
      console.error("Generation error:", err);
    } finally {
      isGenerating = false;
    }
  }
  
  function initializeVisualization() {
    if (chart) return;
    if (!chartContainer) return;
    
    try {
      chart = createBarChart(chartContainer);
    } catch (err) {
      console.error("Failed to initialize visualization:", err);
    }
  }
  
  function goToStep(step: number) {
    currentStep = [step];
  }
  
  function prevStep() {
    if (currentStep[0] > 0) {
      currentStep = [currentStep[0] - 1];
    }
  }
  
  function nextStep() {
    if (currentStep[0] < steps.length - 1) {
      currentStep = [currentStep[0] + 1];
    }
  }
</script>

<div class="space-y-8">
  <!-- Main parameters card -->
  <Card>
    <CardContent class="pt-6">
      <div class="flex flex-col space-y-6">
        <div>
          <label for="prompt" class="text-sm font-medium mb-1 block">Prompt</label>
          <Textarea 
            id="prompt"
            bind:value={prompt} 
            rows={3}
            placeholder="Enter your prompt here..."
          />
        </div>
        
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          <div>
            <label for="threshold-slider" class="text-sm font-medium mb-1 block">
              Selection Threshold <span class="text-primary font-mono">{selectionThreshold[0].toFixed(2)}</span>
            </label>
            <Slider 
              bind:value={selectionThreshold} 
              id="threshold-slider"
            />
            <div class="text-xs text-muted-foreground mt-1">
              Lower values = more diverse options
            </div>
          </div>
          
          <div>
            <label for="maxTokens" class="text-sm font-medium mb-1 block">
              Max Tokens
            </label>
            <Input 
              id="maxTokens"
              type="number" 
              bind:value={maxTokens} 
              min={1} 
              max={600} 
            />
          </div>
          
          <div>
            <label for="minSteps" class="text-sm font-medium mb-1 block">
              Min Steps
            </label>
            <Input 
              id="minSteps"
              type="number" 
              bind:value={minSteps} 
              min={0} 
            />
          </div>
        </div>
        
        <!-- Pruning Settings -->
        <div>
          <div class="flex justify-between items-center mb-4">
            <h3 class="text-lg font-medium">Pruning Settings</h3>
            <div class="flex items-center space-x-2">
              <label for="usePruning" class="text-sm font-medium">Use Pruning</label>
              <Switch id="usePruning" bind:checked={usePruning} />
            </div>
          </div>
          
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-6 {!usePruning ? 'opacity-50' : ''}">
            <div class="flex items-center space-x-2">
              <Checkbox 
                id="useDiversityPruning"
                bind:checked={useDiversityPruning} 
                disabled={!usePruning}
              />
              <label for="useDiversityPruning" class="text-sm">
                Use Diversity Pruning
              </label>
            </div>
            
            <div class="flex items-center space-x-2">
              <Checkbox 
                id="useRetroactivePruning"
                bind:checked={useRetroactivePruning} 
                disabled={!usePruning}
              />
              <label for="useRetroactivePruning" class="text-sm">
                Use Retroactive Pruning
              </label>
            </div>
            
            <div>
              <label for="pruningStrategy" class="text-sm font-medium mb-1 block">
                Pruning Strategy
              </label>
              <select
                id="pruningStrategy"
                class="w-full px-3 py-2 border border-input bg-background text-sm rounded-md"
                bind:value={pruningStrategy}
                disabled={!usePruning}
              >
                <option value="coherence">Coherence</option>
                <option value="diversity">Diversity</option>
                <option value="hybrid">Hybrid</option>
              </select>
            </div>
            
            <div>
              <label for="coherenceThreshold" class="text-sm font-medium mb-1 block">
                Coherence Threshold
              </label>
              <Input 
                id="coherenceThreshold"
                type="number" 
                bind:value={coherenceThreshold} 
                min={0}
                max={1}
                step={0.01}
                disabled={!usePruning || pruningStrategy === "diversity"}
              />
            </div>
            
            <div>
              <label for="diversityClusters" class="text-sm font-medium mb-1 block">
                Diversity Clusters
              </label>
              <Input 
                id="diversityClusters"
                type="number" 
                bind:value={diversityClusters} 
                min={1} 
                max={10} 
                disabled={!usePruning || pruningStrategy === "coherence"}
              />
            </div>
            
            <div>
              <label for="diversitySteps" class="text-sm font-medium mb-1 block">
                Diversity Steps
              </label>
              <Input 
                id="diversitySteps"
                type="number" 
                bind:value={diversitySteps} 
                min={0}
                disabled={!usePruning || pruningStrategy !== "hybrid"}
              />
            </div>
            
            <div>
              <label for="retroThreshold" class="text-sm font-medium mb-1 block">
                Retroactive Threshold <span class="text-primary font-mono">{retroactivePruningThreshold[0].toFixed(3)}</span>
              </label>
              <Slider 
                id="retroThreshold"
                bind:value={retroactivePruningThreshold} 
                disabled={!usePruning || !useRetroactivePruning}
              />
            </div>
            
            <div class="sm:col-span-2 pt-3 border-t border-border">
              <div class="flex items-center justify-between">
                <label for="useDynamicThreshold" class="text-sm font-medium">
                  Use Dynamic Threshold
                </label>
                <Switch 
                  id="useDynamicThreshold" 
                  bind:checked={useDynamicThreshold}
                  disabled={!usePruning} 
                />
              </div>
              
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-3 {(!usePruning || !useDynamicThreshold) ? 'opacity-50' : ''}">
                <div>
                  <label for="finalThreshold" class="text-sm font-medium mb-1 block">
                    Final Threshold <span class="text-primary font-mono">{finalThreshold[0].toFixed(2)}</span>
                  </label>
                  <Slider 
                    id="finalThreshold"
                    bind:value={finalThreshold}
                    disabled={!usePruning || !useDynamicThreshold}
                  />
                </div>
                
                <div class="flex items-center justify-between">
                  <label for="useRelu" class="text-sm font-medium">
                    Use ReLU Transition
                  </label>
                  <Switch 
                    id="useRelu" 
                    bind:checked={useRelu}
                    disabled={!usePruning || !useDynamicThreshold} 
                  />
                </div>
                
                <div>
                  <label for="reluActivationPoint" class="text-sm font-medium mb-1 block">
                    ReLU Activation Point <span class="text-primary font-mono">{reluActivationPoint[0].toFixed(2)}</span>
                  </label>
                  <Slider 
                    id="reluActivationPoint"
                    bind:value={reluActivationPoint}
                    disabled={!usePruning || !useDynamicThreshold || !useRelu}
                  />
                </div>
                
                <div>
                  <label for="bezierP1" class="text-sm font-medium mb-1 block">
                    Bezier P1 <span class="text-primary font-mono">{bezierP1[0].toFixed(2)}</span>
                  </label>
                  <Slider 
                    id="bezierP1"
                    bind:value={bezierP1}
                    disabled={!usePruning || !useDynamicThreshold || useRelu}
                  />
                </div>
                
                <div>
                  <label for="bezierP2" class="text-sm font-medium mb-1 block">
                    Bezier P2 <span class="text-primary font-mono">{bezierP2[0].toFixed(2)}</span>
                  </label>
                  <Slider 
                    id="bezierP2"
                    bind:value={bezierP2}
                    disabled={!usePruning || !useDynamicThreshold || useRelu}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Advanced Settings Toggle -->
        <div>
          <Button 
            variant="ghost"
            on:click={() => showAdvancedSettings = !showAdvancedSettings}
            class="text-sm flex items-center text-muted-foreground hover:text-foreground"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
            {showAdvancedSettings ? 'Hide' : 'Show'} Advanced Settings
          </Button>
        
          {#if showAdvancedSettings}
            <div class="mt-4 pt-4 border-t border-border space-y-6">
              <!-- MCTS Settings -->
              <div>
                <div class="flex justify-between items-center mb-4">
                  <h3 class="text-lg font-medium">MCTS Settings</h3>
                  <div class="flex items-center space-x-2">
                    <label class="text-sm font-medium">Use MCTS</label>
                    <Switch id="useMCTS" bind:checked={useMCTS} />
                  </div>
                </div>
                
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-6 {!useMCTS ? 'opacity-50' : ''}">
                  <div>
                    <label class="text-sm font-medium mb-1 block">
                      MCTS Simulations
                    </label>
                    <Input 
                      type="number" 
                      bind:value={mctsSimulations} 
                      min="1" 
                      max="50" 
                      disabled={!useMCTS}
                    />
                  </div>
                  
                  <div>
                    <label class="text-sm font-medium mb-1 block">
                      Exploration (c_puct)
                    </label>
                    <Input 
                      type="number" 
                      bind:value={mctsCPuct} 
                      min="0.1" 
                      max="5.0" 
                      step="0.1"
                      disabled={!useMCTS}
                    />
                  </div>
                  
                  <div>
                    <label class="text-sm font-medium mb-1 block">
                      MCTS Depth
                    </label>
                    <Input 
                      type="number" 
                      bind:value={mctsDepth} 
                      min="1" 
                      max="10" 
                      disabled={!useMCTS}
                    />
                  </div>
                </div>
              </div>
              
              <!-- Additional Settings -->
              <div>
                <h3 class="text-lg font-medium mb-4">Additional Settings</h3>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div class="flex items-center space-x-2">
                    <Checkbox 
                      id="enableThinking"
                      bind:checked={enableThinking} 
                    />
                    <label for="enableThinking" class="text-sm">
                      Enable Thinking Mode
                    </label>
                  </div>
                  
                  <div class="flex items-center space-x-2">
                    <Checkbox 
                      id="allowIntrasetTokenVisibility"
                      bind:checked={allowIntrasetTokenVisibility} 
                    />
                    <label for="allowIntrasetTokenVisibility" class="text-sm">
                      Allow Intraset Token Visibility
                    </label>
                  </div>
                  
                  <div class="flex items-center space-x-2">
                    <Checkbox 
                      id="noPreserveIsolatedTokens"
                      bind:checked={noPreserveIsolatedTokens} 
                      disabled={allowIntrasetTokenVisibility}
                    />
                    <label for="noPreserveIsolatedTokens" class="text-sm">
                      Don't Preserve Isolated Tokens
                    </label>
                  </div>
                  
                  <div class="flex items-center space-x-2">
                    <Checkbox 
                      id="useCustomRope"
                      bind:checked={useCustomRope} 
                    />
                    <label for="useCustomRope" class="text-sm">
                      Use Custom RoPE Modifications
                    </label>
                  </div>
                  
                  <div class="flex items-center space-x-2">
                    <Checkbox 
                      id="disableKVCache"
                      bind:checked={disableKVCache} 
                    />
                    <label for="disableKVCache" class="text-sm">
                      Disable KV Cache
                    </label>
                  </div>
                  
                  <div class="flex items-center space-x-2">
                    <Checkbox 
                      id="disableKvCacheConsistency"
                      bind:checked={disableKvCacheConsistency} 
                    />
                    <label for="disableKvCacheConsistency" class="text-sm">
                      Disable KV Cache Consistency
                    </label>
                  </div>
                </div>
              </div>
              
              <!-- Advanced Retroactive Pruning Settings -->
              <div class="{!usePruning || !useRetroactivePruning ? 'opacity-50' : ''}">
                <h3 class="text-lg font-medium mb-4">Advanced Retroactive Pruning</h3>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div class="flex items-center space-x-2">
                    <Checkbox 
                      id="noRelativeAttention"
                      bind:checked={noRelativeAttention} 
                      disabled={!usePruning || !useRetroactivePruning}
                    />
                    <label for="noRelativeAttention" class="text-sm">
                      Disable Relative Attention
                    </label>
                  </div>
                  
                  <div>
                    <label for="relativeThreshold" class="text-sm font-medium mb-1 block">
                      Relative Threshold <span class="text-primary font-mono">{relativeThreshold[0].toFixed(2)}</span>
                    </label>
                    <Slider 
                      id="relativeThreshold"
                      bind:value={relativeThreshold}
                      disabled={!usePruning || !useRetroactivePruning || noRelativeAttention}
                    />
                  </div>
                  
                  <div class="flex items-center space-x-2">
                    <Checkbox 
                      id="noMultiScaleAttention"
                      bind:checked={noMultiScaleAttention} 
                      disabled={!usePruning || !useRetroactivePruning}
                    />
                    <label for="noMultiScaleAttention" class="text-sm">
                      Disable Multi-Scale Attention
                    </label>
                  </div>
                  
                  <div>
                    <label for="numLayersToUse" class="text-sm font-medium mb-1 block">
                      Number of Layers to Use
                    </label>
                    <Input 
                      id="numLayersToUse"
                      type="number" 
                      bind:value={numLayersToUse}
                      placeholder="All Layers" 
                      disabled={!usePruning || !useRetroactivePruning}
                    />
                    <div class="text-xs text-muted-foreground mt-1">
                      Leave empty to use all layers
                    </div>
                  </div>
                  
                  <div class="flex items-center space-x-2">
                    <Checkbox 
                      id="noLciDynamicThreshold"
                      bind:checked={noLciDynamicThreshold} 
                      disabled={!usePruning || !useRetroactivePruning}
                    />
                    <label for="noLciDynamicThreshold" class="text-sm">
                      Disable LCI Dynamic Threshold
                    </label>
                  </div>
                  
                  <div class="flex items-center space-x-2">
                    <Checkbox 
                      id="noSigmoidThreshold"
                      bind:checked={noSigmoidThreshold} 
                      disabled={!usePruning || !useRetroactivePruning}
                    />
                    <label for="noSigmoidThreshold" class="text-sm">
                      Disable Sigmoid Threshold
                    </label>
                  </div>
                  
                  <div>
                    <label for="sigmoidSteepness" class="text-sm font-medium mb-1 block">
                      Sigmoid Steepness
                    </label>
                    <Input 
                      id="sigmoidSteepness"
                      type="number" 
                      bind:value={sigmoidSteepness} 
                      min={1}
                      step={0.5}
                      disabled={!usePruning || !useRetroactivePruning || noSigmoidThreshold}
                    />
                  </div>
                  
                  <div>
                    <label for="completePruningMode" class="text-sm font-medium mb-1 block">
                      Complete Pruning Mode
                    </label>
                    <select
                      id="completePruningMode"
                      class="w-full px-3 py-2 border border-input bg-background text-sm rounded-md"
                      bind:value={completePruningMode}
                      disabled={!usePruning || !useRetroactivePruning}
                    >
                      <option value="keep_token">Keep Token</option>
                      <option value="keep_unattended">Keep Unattended</option>
                      <option value="remove_position">Remove Position</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>
          {/if}
        </div>
        
        <div class="pt-4">
          <Button 
            variant="default"
            on:click={generateText} 
            disabled={isGenerating || !prompt.trim()} 
            class={isGenerating ? 'opacity-70 cursor-not-allowed' : ''}
          >
            {#if isGenerating}
              <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Generating...
            {:else}
              Generate Text
            {/if}
          </Button>
        </div>
        
        {#if error}
          <div class="mt-4 p-3 bg-destructive/10 text-destructive rounded-md text-sm">
            {error}
          </div>
        {/if}
      </div>
    </CardContent>
  </Card>
  
  {#if steps.length > 0}
    <!-- Generated Text Display -->
    <Card>
      <CardHeader>
        <CardTitle>Generated Result</CardTitle>
      </CardHeader>
      <CardContent>
        <div class="p-4 bg-muted text-foreground rounded-md whitespace-pre-wrap font-mono text-sm code-output overflow-auto max-h-96" 
             style="line-height: 1.6;">
          {@html generatedText}
        </div>
      </CardContent>
    </Card>
    
    <!-- Visualization Section -->
    <Card>
      <CardHeader>
        <CardTitle>Token Visualization</CardTitle>
      </CardHeader>
      <CardContent>
        <div class="flex flex-col items-center justify-center mb-6">
          <span class="mb-2 text-sm font-medium">Step {currentStep[0] + 1} of {steps.length}</span>
          <Slider 
            bind:value={currentStep} 
            min={0} 
            max={steps.length - 1} 
            step={1}
            disabled={steps.length <= 1}
            class="mb-4"
          />
          
          <div class="flex items-center justify-center space-x-4 w-full">
            <Button 
              variant="outline"
              size="icon"
              on:click={prevStep} 
              disabled={currentStep[0] === 0 || steps.length <= 1} 
              class="disabled:opacity-50"
            >
              <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clip-rule="evenodd" />
              </svg>
            </Button>
            
            <div class="inline-flex bg-muted rounded-md p-1 overflow-x-auto max-w-full">
              {#each Array(Math.min(steps.length, 10)) as _, i}
                {@const stepIndex = Math.floor(i * (steps.length / Math.min(10, steps.length)))}
                <Button 
                  variant={currentStep[0] === stepIndex ? "default" : "ghost"}
                  size="sm"
                  on:click={() => goToStep(stepIndex)} 
                  class="px-3 py-1 m-1"
                >
                  {stepIndex + 1}
                </Button>
              {/each}
            </div>
            
            <Button 
              variant="outline"
              size="icon"
              on:click={nextStep} 
              disabled={currentStep[0] === steps.length - 1 || steps.length <= 1} 
              class="disabled:opacity-50"
            >
              <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
              </svg>
            </Button>
          </div>
        </div>
        
        <div bind:this={chartContainer} class="w-full h-80 bg-background border border-border rounded-lg mb-6"></div>
        
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h3 class="font-medium mb-2">Parallel Tokens</h3>
            <div class="p-3 border border-border rounded-lg h-60 overflow-y-auto bg-background">
              {#if steps[currentStep[0]]}
                <table class="w-full text-sm">
                  <thead class="bg-muted">
                    <tr>
                      <th class="px-2 py-1 text-left">Token</th>
                      <th class="px-2 py-1 text-right">Probability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {#each steps[currentStep[0]].parallel_tokens as token}
                      <tr class="{token.isPruned ? 'text-muted-foreground' : ''} border-t border-border">
                        <td class="px-2 py-1">{token.token_text}</td>
                        <td class="px-2 py-1 text-right">{token.probability.toFixed(5)}</td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              {/if}
            </div>
          </div>
          
          <div>
            <h3 class="font-medium mb-2">Pruned Tokens</h3>
            <div class="p-3 border border-border rounded-lg h-60 overflow-y-auto bg-background">
              {#if steps[currentStep[0]]}
                <table class="w-full text-sm">
                  <thead class="bg-muted">
                    <tr>
                      <th class="px-2 py-1 text-left">Token</th>
                      <th class="px-2 py-1 text-right">Probability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {#each steps[currentStep[0]].pruned_tokens as token}
                      <tr class="border-t border-border">
                        <td class="px-2 py-1">{token.token_text}</td>
                        <td class="px-2 py-1 text-right">{token.probability.toFixed(5)}</td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              {/if}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  {/if}
</div>

<style>
  /* Custom styles */
  :global(.code-output) {
    scrollbar-width: thin;
    scrollbar-color: rgba(var(--muted), 0.2) transparent;
  }
  
  :global(.code-output::-webkit-scrollbar) {
    width: 8px;
    height: 8px;
  }
  
  :global(.code-output::-webkit-scrollbar-track) {
    background: transparent;
  }
  
  :global(.code-output::-webkit-scrollbar-thumb) {
    background-color: rgba(var(--muted), 0.2);
    border-radius: 4px;
  }
</style> 