<script>
  import { onMount } from 'svelte';
  import * as d3 from 'd3';
  import { createBarChart } from '$lib/visualizations/barChart';
  
  // INVARIANT: All parameters must have valid default values
  let prompt = "Is a hotdog a sandwich?";
  let threshold = 0.05;
  let maxTokens = 30;
  let minSteps = 0;
  let usePruning = true;
  let pruningStrategy = "coherence";
  let diversitySteps = 5;
  let coherenceThreshold = 0.7;
  let diversityClusters = 3;
  
  // State variables
  let isGenerating = false;
  let error = null;
  let generatedText = "";
  let steps = [];
  let currentStep = 0;
  
  // Chart reference
  let chartContainer;
  let chart;
  
  // INVARIANT: Parameters must always be within these bounds
  $: {
    // Enforce threshold bounds
    threshold = Math.max(0, Math.min(1, threshold));
    
    // Enforce token bounds
    maxTokens = Math.max(1, Math.min(200, maxTokens));
    
    // Enforce step bounds
    minSteps = Math.max(0, minSteps);
    diversitySteps = Math.max(0, diversitySteps);
    
    // Enforce threshold bounds
    coherenceThreshold = Math.max(0, Math.min(1, coherenceThreshold));
    
    // Enforce cluster bounds
    diversityClusters = Math.max(1, Math.min(10, diversityClusters));
  }
  
  onMount(() => {
    // Initialize visualization
    if (chartContainer) {
      chart = createBarChart(chartContainer);
    }
  });
  
  // Update the visualization when steps or currentStep changes
  $: if (chart && steps.length > 0 && steps[currentStep]) {
    try {
      const stepData = steps[currentStep];
      
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
      currentStep = 0;
      
      // INVARIANT: API request parameters must be valid
      const requestData = {
        prompt: prompt.trim(),
        threshold: parseFloat(threshold),
        max_tokens: parseInt(maxTokens),
        min_steps: parseInt(minSteps),
        use_pruning: usePruning,
        pruning_strategy: pruningStrategy,
        diversity_steps: parseInt(diversitySteps),
        coherence_threshold: parseFloat(coherenceThreshold),
        diversity_clusters: parseInt(diversityClusters)
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
          throw new Error(`Generation failed: No tokens above threshold. Please try lowering the threshold value further (current: ${threshold}). Try 0.01 or lower.`);
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
      generatedText = data.generated_text;
      steps = data.steps;
      
      // Handle case where we got data but no steps
      if (steps.length === 0) {
        error = "No generation steps returned. Try lowering the threshold value.";
      } else {
        currentStep = 0;
        
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
      
    } catch (err) {
      error = err.message || "An error occurred during generation";
      console.error("Generation error:", err);
    } finally {
      isGenerating = false;
    }
  }
  
  function goToStep(step) {
    // INVARIANT: Step index must be within bounds
    currentStep = Math.max(0, Math.min(steps.length - 1, step));
  }
  
  function prevStep() {
    if (currentStep > 0) {
      currentStep--;
    }
  }
  
  function nextStep() {
    if (currentStep < steps.length - 1) {
      currentStep++;
    }
  }
</script>

<div class="container mx-auto px-4 py-8">
  <h1 class="text-3xl font-bold mb-6">TEMPO Visualization</h1>
  
  <div class="mb-8 p-6 bg-gray-50 rounded-lg shadow">
    <h2 class="text-xl font-semibold mb-4">Generation Parameters</h2>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div class="col-span-1 md:col-span-2">
        <label class="block text-sm font-medium text-gray-700 mb-1">Prompt</label>
        <textarea 
          bind:value={prompt} 
          class="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500" 
          rows="3"
        ></textarea>
      </div>
      
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1">
          Threshold ({threshold})
        </label>
        <input 
          type="range" 
          bind:value={threshold} 
          min="0" 
          max="1" 
          step="0.05" 
          class="w-full"
        />
      </div>
      
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1">
          Max Tokens
        </label>
        <input 
          type="number" 
          bind:value={maxTokens} 
          min="1" 
          max="200" 
          class="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1">
          Min Steps
        </label>
        <input 
          type="number" 
          bind:value={minSteps} 
          min="0" 
          class="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1" for="usePruning">
          Use Pruning
        </label>
        <input 
          id="usePruning"
          type="checkbox" 
          bind:checked={usePruning} 
          class="mr-2"
        />
      </div>
      
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1">
          Pruning Strategy
        </label>
        <select 
          bind:value={pruningStrategy} 
          class="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={!usePruning}
        >
          <option value="hybrid">Hybrid</option>
          <option value="coherence">Coherence</option>
          <option value="diversity">Diversity</option>
        </select>
      </div>
      
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1">
          Diversity Steps
        </label>
        <input 
          type="number" 
          bind:value={diversitySteps} 
          min="0" 
          class="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={!usePruning || pruningStrategy !== 'hybrid'}
        />
      </div>
      
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1">
          Coherence Threshold ({coherenceThreshold})
        </label>
        <input 
          type="range" 
          bind:value={coherenceThreshold} 
          min="0" 
          max="1" 
          step="0.05" 
          class="w-full"
          disabled={!usePruning || pruningStrategy === 'diversity'}
        />
      </div>
      
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1">
          Diversity Clusters
        </label>
        <input 
          type="number" 
          bind:value={diversityClusters} 
          min="1" 
          max="10" 
          class="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={!usePruning || pruningStrategy === 'coherence'}
        />
      </div>
    </div>
    
    <div class="mt-4">
      <button 
        on:click={generateText} 
        disabled={isGenerating || !prompt.trim()} 
        class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
      >
        {isGenerating ? 'Generating...' : 'Generate Text'}
      </button>
    </div>
    
    {#if error}
      <div class="mt-4 p-3 bg-red-100 text-red-800 rounded-md">
        {error}
      </div>
    {/if}
  </div>
  
  {#if steps.length > 0}
    <div class="mb-8">
      <h2 class="text-xl font-semibold mb-4">Generated Text</h2>
      <div class="p-4 bg-white border border-gray-200 rounded-md">
        {generatedText}
      </div>
    </div>
    
    <div class="mb-8">
      <h2 class="text-xl font-semibold mb-4">
        Token Visualization (Step {currentStep + 1} of {steps.length})
      </h2>
      
      <div class="flex items-center justify-center mb-4">
        <button 
          on:click={prevStep} 
          disabled={currentStep === 0} 
          class="px-3 py-1 bg-gray-200 rounded-md mr-2 disabled:opacity-50"
        >
          Previous
        </button>
        
        <span class="mx-2">Step {currentStep + 1}</span>
        
        <button 
          on:click={nextStep} 
          disabled={currentStep === steps.length - 1} 
          class="px-3 py-1 bg-gray-200 rounded-md ml-2 disabled:opacity-50"
        >
          Next
        </button>
      </div>
      
      <div bind:this={chartContainer} class="w-full h-80 bg-white border border-gray-200 rounded-md"></div>
      
      <div class="mt-4 grid grid-cols-2 gap-4">
        <div>
          <h3 class="font-medium mb-2">Parallel Tokens</h3>
          <div class="p-3 bg-white border border-gray-200 rounded-md h-60 overflow-y-auto">
            {#if steps[currentStep]}
              <table class="w-full text-sm">
                <thead>
                  <tr>
                    <th class="text-left font-medium">Token</th>
                    <th class="text-left font-medium">ID</th>
                    <th class="text-right font-medium">Probability</th>
                  </tr>
                </thead>
                <tbody>
                  {#each steps[currentStep].parallel_tokens.sort((a, b) => b.probability - a.probability) as token}
                    <tr class="border-t border-gray-100">
                      <td class="py-1">{token.token_text}</td>
                      <td class="py-1">{token.token_id}</td>
                      <td class="py-1 text-right">{token.probability.toFixed(4)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            {/if}
          </div>
        </div>
        
        <div>
          <h3 class="font-medium mb-2">Pruned Result</h3>
          <div class="p-3 bg-white border border-gray-200 rounded-md h-60 overflow-y-auto">
            {#if steps[currentStep]}
              <table class="w-full text-sm">
                <thead>
                  <tr>
                    <th class="text-left font-medium">Token</th>
                    <th class="text-left font-medium">ID</th>
                    <th class="text-right font-medium">Probability</th>
                  </tr>
                </thead>
                <tbody>
                  {#each steps[currentStep].pruned_tokens.sort((a, b) => b.probability - a.probability) as token}
                    <tr class="border-t border-gray-100">
                      <td class="py-1">{token.token_text}</td>
                      <td class="py-1">{token.token_id}</td>
                      <td class="py-1 text-right">{token.probability.toFixed(4)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            {/if}
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  :global(body) {
    background-color: #f9fafb;
    font-family: 'Inter', sans-serif;
  }
  
  /* Bar chart styles */
  :global(.bar) {
    fill: #3b82f6;
    transition: fill 0.3s;
  }
  
  :global(.bar.pruned) {
    fill: #ef4444;
  }
  
  :global(.bar-label) {
    font-size: 10px;
    fill: #374151;
  }
</style> 