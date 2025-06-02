// Format TEMPO output for cleaner display

export function formatCleanText(text: string): string {
  if (!text) return '';
  
  // Remove ANSI codes if any
  const cleanText = text.replace(/\x1b\[[0-9;]*m/g, '');
  
  // Clean up the TEMPO bracket notation
  return cleanText
    // Remove empty brackets
    .replace(/\[\s*\]/g, '')
    // Clean up multiple spaces
    .replace(/\s+/g, ' ')
    // Clean up space before punctuation
    .replace(/\s+([.,!?;:])/g, '$1')
    // Clean up multiple punctuation
    .replace(/([.,!?;:])\1+/g, '$1')
    // Trim
    .trim();
}

interface OutputSection {
  type: 'text' | 'parallel';
  content?: string;
  tokens?: string[];
}

export function formatParallelTokens(text: string): OutputSection[] {
  if (!text) return [];
  
  // Extract all bracketed sections [token1/token2/token3]
  const sections: OutputSection[] = [];
  let current = '';
  let inBracket = false;
  let bracketContent = '';
  
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    
    if (char === '[' && !inBracket) {
      // Start of bracket
      if (current.trim()) {
        sections.push({ type: 'text', content: current });
      }
      current = '';
      inBracket = true;
      bracketContent = '';
    } else if (char === ']' && inBracket) {
      // End of bracket
      if (bracketContent.trim()) {
        const tokens = bracketContent.split('/').map(t => t.trim()).filter(t => t);
        sections.push({ type: 'parallel', tokens });
      }
      inBracket = false;
      bracketContent = '';
    } else if (inBracket) {
      bracketContent += char;
    } else {
      current += char;
    }
  }
  
  // Add any remaining text
  if (current.trim()) {
    sections.push({ type: 'text', content: current });
  }
  
  return sections;
}

export function renderFormattedOutput(text: string): string {
  const sections = formatParallelTokens(text);
  let html = '';
  
  sections.forEach(section => {
    if (section.type === 'text') {
      html += `<span class="text-content">${section.content}</span>`;
    } else if (section.type === 'parallel') {
      html += '<span class="parallel-tokens">';
      html += '<span class="bracket">[</span>';
      section.tokens.forEach((token, idx) => {
        html += `<span class="token ${idx === 0 ? 'primary' : 'alternative'}">${token}</span>`;
        if (idx < section.tokens.length - 1) {
          html += '<span class="separator">/</span>';
        }
      });
      html += '<span class="bracket">]</span>';
      html += '</span>';
    }
  });
  
  return html;
}

export function renderInteractiveTokens(text: string): string {
  const sections = formatParallelTokens(text);
  let html = '';
  
  sections.forEach(section => {
    if (section.type === 'text') {
      html += section.content;
    } else if (section.type === 'parallel' && section.tokens && section.tokens.length > 0) {
      const primaryToken = section.tokens[0];
      const alternatives = section.tokens.slice(1);
      
      if (alternatives.length === 0) {
        // No alternatives, just show the token
        html += primaryToken;
      } else {
        // Has alternatives, make it interactive
        html += `<span class="token-choice" data-alternatives="${alternatives.map(t => t.replace(/"/g, '&quot;')).join('|')}">`;
        html += `<span class="chosen-token">${primaryToken}</span>`;
        html += `<span class="alternatives-popup">`;
        html += `<div class="popup-header">Alternative tokens considered:</div>`;
        html += `<div class="alternatives-list">`;
        alternatives.forEach(alt => {
          html += `<div class="alternative-token">${alt}</div>`;
        });
        html += `</div>`;
        html += `</span>`;
        html += `</span>`;
      }
    }
  });
  
  return html;
}