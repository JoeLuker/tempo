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

  type ApiResponse = {
    generated_text: string;
    token_sets?: [number, [number, number][], [number, number][]][];
    timing?: {
      generation_time: number;
      pruning_time: number;
      elapsed_time: number;
    };
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

  // State variables
  let prompt = '';
  let maxTokens = [50];
  let selectionThreshold = [0.1];
  let useRetroactivePruning = true;
  let attentionThreshold = [0.01];
  let debugMode = false;
  let isGenerating = false;
  let error = '';
  let apiResponse: ApiResponse | null = null;
  let chart: { update: (data: ChartToken[]) => void; cleanup: () => void } | null = null;
  let chartContainer: HTMLElement;

  // MCTS parameters
  let useMcts = false;
  let mctsSimulations = [10];
  let mctsCPuct = [1.0];
  let mctsDepth = [5];

  // Dynamic threshold parameters
  let dynamicThreshold = false;
  let finalThreshold = [1.0];
  let bezierP1 = [0.2];
  let bezierP2 = [0.8];
  let useRelu = false;
  let reluActivationPoint = [0.5];

  // Advanced parameters
  let useCustomRope = true;
  let disableKvCache = false;
  let showTokenIds = false;
  let systemContent = '';
  let enableThinking = false;
  let allowIntrasetTokenVisibility = false;
  let noPreserveIsolatedTokens = false;
  let noRelativeAttention = false;
  let relativeThreshold = [0.5];
  let noMultiScaleAttention = false;
  let numLayersToUse: number | null = null;
  let noLciDynamicThreshold = false;
  let noSigmoidThreshold = false;
  let sigmoidSteepness = [10.0];
  let completePruningMode = 'keep_token';
  let disableKvCacheConsistency = false;

  // Function to generate text
  async function generateText() {
    if (!prompt.trim()) {
      error = 'Please enter a prompt';
      return;
    }

    isGenerating = true;
    error = '';

    try {
      const fetchResponse = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          max_tokens: maxTokens[0],
          selection_threshold: selectionThreshold[0],
          use_retroactive_pruning: useRetroactivePruning,
          attention_threshold: attentionThreshold[0],
          debug_mode: debugMode,
          // MCTS parameters
          use_mcts: useMcts,
          mcts_simulations: mctsSimulations[0],
          mcts_c_puct: mctsCPuct[0],
          mcts_depth: mctsDepth[0],
          // Dynamic threshold parameters
          dynamic_threshold: dynamicThreshold,
          final_threshold: finalThreshold[0],
          bezier_p1: bezierP1[0],
          bezier_p2: bezierP2[0],
          use_relu: useRelu,
          relu_activation_point: reluActivationPoint[0],
          // Advanced parameters
          use_custom_rope: useCustomRope,
          disable_kv_cache: disableKvCache,
          show_token_ids: showTokenIds,
          system_content: systemContent,
          enable_thinking: enableThinking,
          allow_intraset_token_visibility: allowIntrasetTokenVisibility,
          no_preserve_isolated_tokens: noPreserveIsolatedTokens,
          no_relative_attention: noRelativeAttention,
          relative_threshold: relativeThreshold[0],
          no_multi_scale_attention: noMultiScaleAttention,
          num_layers_to_use: numLayersToUse,
          no_lci_dynamic_threshold: noLciDynamicThreshold,
          no_sigmoid_threshold: noSigmoidThreshold,
          sigmoid_steepness: sigmoidSteepness[0],
          complete_pruning_mode: completePruningMode,
          disable_kv_cache_consistency: disableKvCacheConsistency,
        }),
      });

      if (!fetchResponse.ok) {
        throw new Error(`HTTP error! status: ${fetchResponse.status}`);
      }

      const data = await fetchResponse.json() as ApiResponse;
      apiResponse = data;

      // Update chart if we have token data
      if (data.token_sets && data.token_sets.length > 0) {
        updateChart(data.token_sets);
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'An error occurred';
    } finally {
      isGenerating = false;
    }
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

  // Handle window resize
  onMount(() => {
    window.addEventListener('resize', () => {
      if (apiResponse?.token_sets) {
        updateChart(apiResponse.token_sets);
      }
    });
  });
</script>

<main class="container mx-auto p-4">
  <h1 class="text-3xl font-bold mb-4">TEMPO Text Generation</h1>

  <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
    <!-- Input Section -->
    <Card>
      <CardHeader>
        <CardTitle>Input</CardTitle>
        <CardDescription>Enter your prompt and adjust generation parameters</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value="basic">
          <TabsList class="grid w-full grid-cols-3">
            <TabsTrigger value="basic">Basic</TabsTrigger>
            <TabsTrigger value="mcts">MCTS</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>

          <TabsContent value="basic" class="space-y-4">
            <div>
              <label for="prompt" class="block text-sm font-medium mb-1">Prompt</label>
              <Textarea
                id="prompt"
                bind:value={prompt}
                placeholder="Enter your prompt here..."
                rows={4}
              />
            </div>

            <div>
              <label for="maxTokens" class="block text-sm font-medium mb-1">Max Tokens</label>
              <Slider
                id="maxTokens"
                bind:value={maxTokens}
                min={1}
                max={200}
                step={1}
              />
              <div class="text-sm text-gray-500 mt-1">{maxTokens[0]} tokens</div>
            </div>

            <div>
              <label for="selectionThreshold" class="block text-sm font-medium mb-1">Selection Threshold</label>
              <Slider
                id="selectionThreshold"
                bind:value={selectionThreshold}
                min={0}
                max={1}
                step={0.01}
              />
              <div class="text-sm text-gray-500 mt-1">{selectionThreshold[0].toFixed(2)}</div>
              <p class="text-xs text-gray-400 mt-1">Probability threshold for initial token candidate selection</p>
            </div>

            <div class="flex items-center space-x-2">
              <Switch id="useRetroactivePruning" bind:checked={useRetroactivePruning} />
              <label for="useRetroactivePruning" class="text-sm font-medium">Use Retroactive Pruning</label>
            </div>

            {#if useRetroactivePruning}
            <div class="pl-4 mt-2 space-y-2 border-l-2 border-gray-200">
              <label for="attentionThreshold" class="block text-sm font-medium mb-1">Attention Threshold</label>
              <Slider
                id="attentionThreshold"
                bind:value={attentionThreshold}
                min={0}
                max={0.1}
                step={0.001}
              />
              <div class="text-sm text-gray-500 mt-1">{attentionThreshold[0].toFixed(3)}</div>
              <p class="text-xs text-gray-400 mt-1">Threshold for pruning past steps based on attention</p>
            </div>
            {/if}

            <div class="flex items-center space-x-2">
              <Switch id="debugMode" bind:checked={debugMode} />
              <label for="debugMode" class="text-sm font-medium">Debug Mode</label>
            </div>
          </TabsContent>

          <TabsContent value="mcts" class="space-y-4">
            <div class="flex items-center space-x-2">
              <Switch id="useMcts" bind:checked={useMcts} />
              <label for="useMcts" class="text-sm font-medium">Use MCTS</label>
            </div>

            {#if useMcts}
            <div class="pl-4 mt-2 space-y-4 border-l-2 border-gray-200">
              <div>
                <label for="mctsSimulations" class="block text-sm font-medium mb-1">MCTS Simulations</label>
                <Slider
                  id="mctsSimulations"
                  bind:value={mctsSimulations}
                  min={1}
                  max={100}
                  step={1}
                />
                <div class="text-sm text-gray-500 mt-1">{mctsSimulations[0]} simulations</div>
              </div>

              <div>
                <label for="mctsCPuct" class="block text-sm font-medium mb-1">MCTS C_PUCT</label>
                <Slider
                  id="mctsCPuct"
                  bind:value={mctsCPuct}
                  min={0.1}
                  max={5.0}
                  step={0.1}
                />
                <div class="text-sm text-gray-500 mt-1">{mctsCPuct[0].toFixed(1)}</div>
              </div>

              <div>
                <label for="mctsDepth" class="block text-sm font-medium mb-1">MCTS Depth</label>
                <Slider
                  id="mctsDepth"
                  bind:value={mctsDepth}
                  min={1}
                  max={20}
                  step={1}
                />
                <div class="text-sm text-gray-500 mt-1">{mctsDepth[0]} steps</div>
              </div>
            </div>
            {/if}

            <div class="flex items-center space-x-2">
              <Switch id="dynamicThreshold" bind:checked={dynamicThreshold} />
              <label for="dynamicThreshold" class="text-sm font-medium">Use Dynamic Threshold</label>
            </div>

            {#if dynamicThreshold}
            <div class="pl-4 mt-2 space-y-4 border-l-2 border-gray-200">
              <div>
                <label for="finalThreshold" class="block text-sm font-medium mb-1">Final Threshold</label>
                <Slider
                  id="finalThreshold"
                  bind:value={finalThreshold}
                  min={0}
                  max={1}
                  step={0.01}
                />
                <div class="text-sm text-gray-500 mt-1">{finalThreshold[0].toFixed(2)}</div>
              </div>

              <div class="flex items-center space-x-2">
                <Switch id="useRelu" bind:checked={useRelu} />
                <label for="useRelu" class="text-sm font-medium">Use ReLU Transition</label>
              </div>

              {#if useRelu}
              <div>
                <label for="reluActivationPoint" class="block text-sm font-medium mb-1">ReLU Activation Point</label>
                <Slider
                  id="reluActivationPoint"
                  bind:value={reluActivationPoint}
                  min={0}
                  max={1}
                  step={0.01}
                />
                <div class="text-sm text-gray-500 mt-1">{reluActivationPoint[0].toFixed(2)}</div>
              </div>
              {:else}
              <div>
                <label for="bezierP1" class="block text-sm font-medium mb-1">Bezier P1</label>
                <Slider
                  id="bezierP1"
                  bind:value={bezierP1}
                  min={0}
                  max={1}
                  step={0.01}
                />
                <div class="text-sm text-gray-500 mt-1">{bezierP1[0].toFixed(2)}</div>
              </div>
              <div>
                <label for="bezierP2" class="block text-sm font-medium mb-1">Bezier P2</label>
                <Slider
                  id="bezierP2"
                  bind:value={bezierP2}
                  min={0}
                  max={1}
                  step={0.01}
                />
                <div class="text-sm text-gray-500 mt-1">{bezierP2[0].toFixed(2)}</div>
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
              </div>

              <div class="flex items-center space-x-2">
                <Switch id="disableKvCache" bind:checked={disableKvCache} />
                <label for="disableKvCache" class="text-sm font-medium">Disable KV Cache</label>
              </div>

              <div class="flex items-center space-x-2">
                <Switch id="showTokenIds" bind:checked={showTokenIds} />
                <label for="showTokenIds" class="text-sm font-medium">Show Token IDs</label>
              </div>

              <div>
                <label for="systemContent" class="block text-sm font-medium mb-1">System Content</label>
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
                  bind:value={relativeThreshold}
                  min={0}
                  max={1}
                  step={0.01}
                />
                <div class="text-sm text-gray-500 mt-1">{relativeThreshold[0].toFixed(2)}</div>
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
                  bind:value={sigmoidSteepness}
                  min={1}
                  max={20}
                  step={0.1}
                />
                <div class="text-sm text-gray-500 mt-1">{sigmoidSteepness[0].toFixed(1)}</div>
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

        <Button
          on:click={generateText}
          disabled={isGenerating}
          class="w-full mt-4"
        >
          {isGenerating ? 'Generating...' : 'Generate'}
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
          <div class="text-red-500 mb-4">{error}</div>
        {/if}

        {#if apiResponse}
          <div class="space-y-4">
            <div>
              <h3 class="text-lg font-medium mb-2">Generated Text</h3>
              <div class="bg-gray-100 dark:bg-gray-800 p-4 rounded">
                {@html ansiToHtml(apiResponse.generated_text)}
              </div>
            </div>

            {#if apiResponse.timing}
              <div>
                <h3 class="text-lg font-medium mb-2">Timing</h3>
                <div class="text-sm">
                  <div>Generation Time: {apiResponse.timing.generation_time.toFixed(2)}s</div>
                  <div>Pruning Time: {apiResponse.timing.pruning_time.toFixed(2)}s</div>
                  <div>Total Time: {apiResponse.timing.elapsed_time.toFixed(2)}s</div>
                </div>
              </div>
            {/if}

            {#if apiResponse.token_sets && apiResponse.token_sets.length > 0}
              <div>
                <h3 class="text-lg font-medium mb-2">Token Visualization</h3>
                <div
                  bind:this={chartContainer}
                  class="w-full h-[400px]"
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