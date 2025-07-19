import ollama # type: ignore
from typing import List, Dict, Any, Tuple
import logging
import re
from dataclasses import dataclass
import tiktoken

@dataclass
class SemanticChunk:
    content: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any]
    quality_score: float = 0.0
    
class SemanticChunkerWithLLMJudge:
    """Semantic chunking with LLM quality assessment"""
    
    def __init__(self, 
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 1000,
                 overlap: int = 50,
                 judge_model: str = "mistral"):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.judge_model = judge_model
        self.logger = logging.getLogger("SemanticChunker")
        
        # Initialize tokenizer for size estimation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_document(self, text: str, metadata: Dict[str, Any] = None) -> List[SemanticChunk]:
        """Chunk document into semantic units"""
        
        # Step 1: Initial segmentation by paragraphs and sections
        initial_segments = self._initial_segmentation(text)
        
        # Step 2: Merge or split segments to meet size constraints
        sized_segments = self._apply_size_constraints(initial_segments)
        
        # Step 3: Create chunks with overlap
        chunks = self._create_chunks_with_overlap(sized_segments)
        
        # Step 4: Judge chunk quality with LLM
        evaluated_chunks = self._evaluate_chunks(chunks, metadata)
        
        # Step 5: Re-chunk low quality chunks if needed
        final_chunks = self._improve_low_quality_chunks(evaluated_chunks, text)
        
        return final_chunks
    
    def _initial_segmentation(self, text: str) -> List[str]:
        """Segment text by natural boundaries"""
        
        # Split by multiple newlines (paragraphs)
        segments = re.split(r'\n\s*\n', text)
        
        # Further split by headers (markdown style)
        refined_segments = []
        for segment in segments:
            if re.match(r'^#+\s', segment):  # Markdown header
                refined_segments.append(segment)
            else:
                # Split long paragraphs by sentences
                if len(segment) > self.max_chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', segment)
                    current = ""
                    for sentence in sentences:
                        if len(current) + len(sentence) < self.max_chunk_size:
                            current += " " + sentence if current else sentence
                        else:
                            if current:
                                refined_segments.append(current)
                            current = sentence
                    if current:
                        refined_segments.append(current)
                else:
                    refined_segments.append(segment)
        
        return [s.strip() for s in refined_segments if s.strip()]
    
    def _apply_size_constraints(self, segments: List[str]) -> List[str]:
        """Merge small segments and split large ones"""
        
        sized_segments = []
        current_segment = ""
        
        for segment in segments:
            segment_tokens = len(self.tokenizer.encode(segment))
            
            if segment_tokens > self.max_chunk_size:
                # Split large segment
                if current_segment:
                    sized_segments.append(current_segment)
                    current_segment = ""
                
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', segment)
                for sentence in sentences:
                    if len(self.tokenizer.encode(current_segment + " " + sentence)) < self.max_chunk_size:
                        current_segment += " " + sentence if current_segment else sentence
                    else:
                        if current_segment:
                            sized_segments.append(current_segment)
                        current_segment = sentence
                        
            elif len(self.tokenizer.encode(current_segment + " " + segment)) < self.max_chunk_size:
                # Merge with current
                current_segment += "\n\n" + segment if current_segment else segment
            else:
                # Start new segment
                if current_segment:
                    sized_segments.append(current_segment)
                current_segment = segment
        
        if current_segment:
            sized_segments.append(current_segment)
        
        return sized_segments
    
    def _create_chunks_with_overlap(self, segments: List[str]) -> List[SemanticChunk]:
        """Create chunks with overlap for context continuity"""
        
        chunks = []
        full_text = "\n\n".join(segments)
        
        for i, segment in enumerate(segments):
            # Find position in full text
            start_index = full_text.find(segment)
            end_index = start_index + len(segment)
            
            # Add overlap from previous segment
            if i > 0 and self.overlap > 0:
                prev_segment = segments[i-1]
                overlap_text = prev_segment[-self.overlap:] if len(prev_segment) > self.overlap else prev_segment
                segment = overlap_text + "\n\n" + segment
                start_index = max(0, start_index - len(overlap_text) - 2)
            
            # Add overlap from next segment
            if i < len(segments) - 1 and self.overlap > 0:
                next_segment = segments[i+1]
                overlap_text = next_segment[:self.overlap] if len(next_segment) > self.overlap else next_segment
                segment = segment + "\n\n" + overlap_text
                end_index = min(len(full_text), end_index + len(overlap_text) + 2)
            
            chunk = SemanticChunk(
                content=segment.strip(),
                start_index=start_index,
                end_index=end_index,
                metadata={"segment_index": i}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _evaluate_chunks(self, chunks: List[SemanticChunk], 
                        doc_metadata: Dict[str, Any] = None) -> List[SemanticChunk]:
        """Use LLM to judge chunk quality"""
        
        evaluated_chunks = []
        
        for chunk in chunks:
            prompt = f"""You are a quality judge for document chunks. Evaluate this chunk and provide a quality score.

Chunk content:
{chunk.content}

Evaluate based on:
1. Completeness - Does it contain complete thoughts/ideas?
2. Context - Is there enough context to understand it standalone?
3. Coherence - Does it flow logically?
4. Relevance - Is the content meaningful and not just fragments?

Respond with ONLY a JSON object like this:
{{"score": 0.85, "complete": true, "has_context": true, "coherent": true, "issues": []}}

The score should be between 0.0 and 1.0."""

            try:
                response = ollama.generate(model=self.judge_model, prompt=prompt)
                
                # Parse response
                import json
                eval_result = json.loads(response['response'])
                
                chunk.quality_score = eval_result.get('score', 0.5)
                chunk.metadata['quality_evaluation'] = eval_result
                
            except Exception as e:
                self.logger.warning(f"Chunk evaluation failed: {e}")
                chunk.quality_score = 0.5  # Default score
                
            evaluated_chunks.append(chunk)
        
        return evaluated_chunks
    
    def _improve_low_quality_chunks(self, chunks: List[SemanticChunk], 
                                   original_text: str) -> List[SemanticChunk]:
        """Re-chunk low quality chunks with different boundaries"""
        
        final_chunks = []
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            
            if chunk.quality_score < 0.6:  # Low quality threshold
                self.logger.info(f"Improving low quality chunk at index {i}")
                
                # Try to merge with adjacent chunks
                if i > 0 and chunks[i-1].quality_score >= 0.7:
                    # Merge with previous
                    merged_content = chunks[i-1].content + "\n\n" + chunk.content
                    if len(self.tokenizer.encode(merged_content)) <= self.max_chunk_size * 1.2:
                        # Remove last chunk and create merged
                        final_chunks.pop()
                        merged_chunk = SemanticChunk(
                            content=merged_content,
                            start_index=chunks[i-1].start_index,
                            end_index=chunk.end_index,
                            metadata={**chunks[i-1].metadata, **chunk.metadata}
                        )
                        # Re-evaluate merged chunk
                        final_chunks.extend(self._evaluate_chunks([merged_chunk]))
                        i += 1
                        continue
                
                # If can't merge, try to expand context
                expanded_start = max(0, chunk.start_index - 100)
                expanded_end = min(len(original_text), chunk.end_index + 100)
                expanded_content = original_text[expanded_start:expanded_end]
                
                expanded_chunk = SemanticChunk(
                    content=expanded_content,
                    start_index=expanded_start,
                    end_index=expanded_end,
                    metadata=chunk.metadata
                )
                
                # Re-evaluate
                evaluated = self._evaluate_chunks([expanded_chunk])
                if evaluated[0].quality_score > chunk.quality_score:
                    final_chunks.append(evaluated[0])
                else:
                    final_chunks.append(chunk)  # Keep original
            else:
                final_chunks.append(chunk)
            
            i += 1
        
        return final_chunks
    
    def chunk_confluence_document(self, document: Dict[str, Any]) -> List[SemanticChunk]:
        """Special handling for Confluence documents"""
        
        chunks = []
        
        # Chunk main content
        main_chunks = self.chunk_document(
            document['content'],
            metadata={
                'doc_id': document['id'],
                'doc_title': document['title'],
                'section': 'main_content'
            }
        )
        chunks.extend(main_chunks)
        
        # Chunk tables separately
        for i, table in enumerate(document.get('extracted_elements', {}).get('tables', [])):
            table_text = self._table_to_text(table)
            table_chunk = SemanticChunk(
                content=table_text,
                start_index=-1,  # Not from main text
                end_index=-1,
                metadata={
                    'doc_id': document['id'],
                    'doc_title': document['title'],
                    'section': 'table',
                    'table_index': i
                },
                quality_score=0.9  # Tables are usually complete
            )
            chunks.append(table_chunk)
        
        # Chunk code blocks
        for i, code_block in enumerate(document.get('extracted_elements', {}).get('code_blocks', [])):
            code_chunk = SemanticChunk(
                content=f"Code ({code_block['language']}):\n{code_block['code']}",
                start_index=-1,
                end_index=-1,
                metadata={
                    'doc_id': document['id'],
                    'doc_title': document['title'],
                    'section': 'code',
                    'code_index': i,
                    'language': code_block['language']
                },
                quality_score=0.95  # Code blocks are self-contained
            )
            chunks.append(code_chunk)
        
        return chunks
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table to readable text"""
        lines = []
        
        # Headers
        if table.get('headers'):
            lines.append(" | ".join(table['headers']))
            lines.append("-" * 50)
        
        # Rows
        for row in table.get('rows', []):
            lines.append(" | ".join(str(cell) for cell in row))
        
        return "\n".join(lines)