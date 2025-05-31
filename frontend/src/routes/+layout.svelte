<script lang="ts">
  import '../app.css';
  import { onMount, onDestroy } from 'svelte';
  import { theme, toggleTheme, initTheme } from '$lib/theme';
  import { Button } from '$lib/components/ui/button';
  import { Card } from '$lib/components/ui/card';
  import ShortcutsDialog from '$lib/components/ShortcutsDialog.svelte';
  import { registerKeyboardShortcuts } from '$lib/utils/keyboard';
  
  // State for shortcuts dialog
  let showShortcutsDialog = false;
  let unregisterShortcuts: () => void;
  
  // Initialize theme and keyboard shortcuts on mount
  onMount(() => {
    initTheme();
    
    // Register keyboard shortcuts just for the dialog
    unregisterShortcuts = registerKeyboardShortcuts({
      // Just handle the shortcuts dialog here, the rest in +page.svelte
      showShortcuts: () => showShortcutsDialog = true,
      generate: () => {}, // No-op for other shortcuts
      reset: () => {},
      theme: () => {},
      basicTab: () => {},
      mctsTab: () => {},
      advancedTab: () => {},
    });
  });
  
  onDestroy(() => {
    if (unregisterShortcuts) {
      unregisterShortcuts();
    }
  });
</script>

<ShortcutsDialog bind:open={showShortcutsDialog} />

<div class="min-h-screen flex flex-col">
  <header class="border-b border-border bg-background">
    <div class="container px-4 py-3 flex justify-between items-center">
      <div class="flex items-center space-x-2">
        <span class="text-xl font-bold text-primary">TEMPO</span>
        <span class="text-muted-foreground text-sm">Visualization</span>
      </div>
      
      <Button 
        on:click={toggleTheme} 
        variant="outline"
        size="icon"
        aria-label="Toggle theme"
      >
        {#if $theme === 'dark'}
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-yellow-500" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clip-rule="evenodd" />
          </svg>
        {:else}
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-700" viewBox="0 0 20 20" fill="currentColor">
            <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
          </svg>
        {/if}
      </Button>
    </div>
  </header>

  <main class="container px-4 py-6 flex-1">
    <slot />
  </main>

  <footer class="border-t border-border py-6">
    <div class="container px-4 mx-auto max-w-screen-xl">
      <div class="flex flex-col sm:flex-row justify-between items-center">
        <div class="text-sm text-muted-foreground mb-4 sm:mb-0">
          TEMPO Visualization Tool
        </div>
        
        <div class="text-xs text-muted-foreground flex items-center">
          <button
            class="inline-flex items-center gap-1 hover:text-primary transition-colors"
            on:click={() => showShortcutsDialog = true}
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="2" y="4" width="20" height="16" rx="2" ry="2" />
              <path d="M6 8h.001M10 8h.001M14 8h.001M18 8h.001M8 12h.001M12 12h.001M16 12h.001M7 16h10" />
            </svg>
            <span>Keyboard Shortcuts</span>
            <kbd class="px-1.5 py-0.5 rounded bg-muted font-mono ml-1">Alt+?</kbd>
          </button>
        </div>
      </div>
    </div>
  </footer>
</div> 