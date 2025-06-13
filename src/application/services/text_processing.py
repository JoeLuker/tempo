"""Text processing utilities for TEMPO output."""

import re
import logging

logger = logging.getLogger(__name__)


class TextProcessingService:
    """Service for processing and cleaning generated text."""
    
    @staticmethod
    def strip_ansi_codes(text: str) -> str:
        """Remove ANSI color codes from text.
        
        Args:
            text: Text potentially containing ANSI escape codes
            
        Returns:
            Text with ANSI codes removed
        """
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
    
    @staticmethod
    def extract_clean_text(text_with_brackets: str) -> str:
        """Extract clean text by taking only the first token from each bracket group.
        
        This method processes TEMPO's output format where parallel tokens are shown
        in brackets like [token1/token2/token3] and extracts only the first token
        from each group to create readable text.
        
        Args:
            text_with_brackets: Text containing bracketed token groups
            
        Returns:
            Clean text with only the first token from each bracket group
        """
        if not text_with_brackets:
            return ""
        
        # Remove ANSI codes first
        text = TextProcessingService.strip_ansi_codes(text_with_brackets)
        
        result = []
        i = 0
        while i < len(text):
            if text[i] == '[':
                # Find the matching closing bracket
                j = i + 1
                bracket_depth = 1
                while j < len(text) and bracket_depth > 0:
                    if text[j] == '[':
                        bracket_depth += 1
                    elif text[j] == ']':
                        bracket_depth -= 1
                    j += 1
                
                if bracket_depth == 0:
                    # Extract content between brackets
                    bracket_content = text[i+1:j-1]
                    # Split by '/' and take the first non-empty token
                    tokens = [t.strip() for t in bracket_content.split('/')]
                    if tokens and tokens[0]:
                        # Add space before token if needed (unless it's punctuation)
                        if result and result[-1] not in ' \n' and tokens[0] not in '.,;:!?"\'':
                            result.append(' ')
                        result.append(tokens[0])
                    i = j
                else:
                    # Unclosed bracket, just append it
                    result.append(text[i])
                    i += 1
            else:
                # Regular character
                result.append(text[i])
                i += 1
        
        return ''.join(result)
    
    @staticmethod
    def process_generation_output(generated_text_with_colors: str, raw_text: str = "") -> dict:
        """Process generation output to provide multiple text formats.
        
        Args:
            generated_text_with_colors: Text with ANSI color codes and brackets
            raw_text: Raw generated text without formatting
            
        Returns:
            Dictionary with different text formats
        """
        # Extract clean text with fallbacks
        clean_text = TextProcessingService.extract_clean_text(generated_text_with_colors) if generated_text_with_colors else ""
        
        # If clean_text is empty but we have other text, use raw_text as fallback
        if not clean_text and raw_text:
            clean_text = TextProcessingService.strip_ansi_codes(raw_text)
        
        # Final fallback: use stripped version of generated_text
        if not clean_text and generated_text_with_colors:
            clean_text = TextProcessingService.strip_ansi_codes(generated_text_with_colors)
        
        # Log if clean_text is empty
        if not clean_text:
            logger.warning(
                f"Clean text is empty - generated_text: {len(generated_text_with_colors) if generated_text_with_colors else 0} chars, "
                f"raw_text: {len(raw_text) if raw_text else 0} chars"
            )
        
        return {
            "generated_text": generated_text_with_colors,
            "raw_generated_text": raw_text,
            "clean_text": clean_text
        }