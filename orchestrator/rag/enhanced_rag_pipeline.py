import logging
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime
import hashlib
import requests
from sentence_transformers import SentenceTransformer # type: ignore
import chromadb # type: ignore
import os

from orchestrator.memory.graph.neo4j_manager import Neo4jManager # type: ignore
from integrations.docling.confluence_extractor import ConfluenceDoclingExtractor
from orchestrator.rag.semantic_chunker import SemanticChunkerWithLLMJudge, SemanticChunk # type: ignore

class EnhancedRAGPipeline:
    """Enhanced RAG with multi-level embeddings and incremental learning"""
    
    def __init__(self, 
                 neo4j_uri: str,
                 neo4j_user: str,
                 neo4j_password: str,
                 confluence_url: str,
                 confluence_user: str,
                 confluence_password: str,
                 chroma_persist_dir: str = "./chroma_data_enhanced"):
        
        self.logger = logging.getLogger("EnhancedRAGPipeline")
        
        # Initialize components
        self.neo4j_manager = Neo4jManager(neo4j_uri, neo4j_user, neo4j_password)
        self.confluence_extractor = ConfluenceDoclingExtractor(
            confluence_url, confluence_user, confluence_password
        )
        self.semantic_chunker = SemanticChunkerWithLLMJudge()
        
        # Initialize embedding models
        self.chunk_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.doc_embedder = SentenceTransformer('all-mpnet-base-v2')  # Different model for doc-level
        
        # Initialize ChromaDB with multiple collections
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
        
        # Collection for chunk-level embeddings
        self.chunk_collection = self.chroma_client.get_or_create_collection(
            name="confluence_chunks_enhanced",
            metadata={"description": "Semantic chunks with quality scores"}
        )
        
        # Collection for document-level embeddings
        self.doc_collection = self.chroma_client.get_or_create_collection(
            name="confluence_docs_enhanced",
            metadata={"description": "Document-level embeddings"}
        )
        
        self.logger.info("Enhanced RAG Pipeline initialized")
    
    def ingest_confluence_space(self, space_key: str, limit: int = 100):
        """Ingest all documents from a Confluence space"""
        self.logger.info(f"Starting ingestion of space: {space_key}")
        
        # Extract documents
        documents = self.confluence_extractor.extract_space_documents(space_key, limit)
        
        for doc in documents:
            try:
                self.ingest_document(doc)
            except Exception as e:
                self.logger.error(f"Failed to ingest document {doc['id']}: {e}")
                continue
        
        self.logger.info(f"Ingestion complete. Processed {len(documents)} documents")
    
    def ingest_document(self, document: Dict[str, Any]):
        """Ingest a single document with all enhancements"""
        doc_id = document['id']
        self.logger.info(f"Ingesting document: {doc_id} - {document['title']}")
        
        # 1. Add document to Neo4j
        self.neo4j_manager.add_document(
            doc_id=doc_id,
            title=document['title'],
            content=document['content'],
            metadata=document['metadata']
        )
        
        # 2. Create document-level embedding
        doc_embedding = self.doc_embedder.encode(
            f"{document['title']} {document['content'][:1000]}"
        )
        
        # Store document-level embedding
        self.doc_collection.add(
            embeddings=[doc_embedding.tolist()],
            documents=[document['content'][:1000]],
            metadatas=[{
                'doc_id': doc_id,
                'title': document['title'],
                'space_key': document['space_key'],
                'type': 'document',
                'created_date': document['created_date']
            }],
            ids=[doc_id]
        )
        
        # 3. Semantic chunking
        chunks = self.semantic_chunker.chunk_confluence_document(document)
        
        # 4. Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # Create chunk embedding
            chunk_embedding = self.chunk_embedder.encode(chunk.content)
            
            # Store in ChromaDB
            self.chunk_collection.add(
                embeddings=[chunk_embedding.tolist()],
                documents=[chunk.content],
                metadatas=[{
                    'chunk_id': chunk_id,
                    'doc_id': doc_id,
                    'doc_title': document['title'],
                    'chunk_index': i,
                    'quality_score': chunk.quality_score,
                    'section': chunk.metadata.get('section', 'main'),
                    **chunk.metadata
                }],
                ids=[chunk_id]
            )
            
            # Add chunk to Neo4j
            self.neo4j_manager.add_chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=chunk.content[:500],  # Store preview
                chunk_index=i,
                embedding_id=chunk_id
            )
        
        # 5. Extract and link ticket references
        ticket_refs = self.confluence_extractor.find_ticket_references(document)
        for ticket_key in ticket_refs:
            self.neo4j_manager.link_document_to_ticket(
                doc_id=doc_id,
                ticket_key=ticket_key,
                relationship_type="REFERENCES"
            )
        
        self.logger.info(f"Successfully ingested {doc_id} with {len(chunks)} chunks")
    
    def hybrid_search(self, 
                     query: str,
                     ticket_context: Optional[Dict[str, Any]] = None,
                     k: int = 10) -> List[Dict[str, Any]]:
        """Hybrid search combining vector similarity and graph relationships"""
        
        results = []
        
        # 1. Vector search at document level
        doc_query_embedding = self.doc_embedder.encode(query)
        doc_results = self.doc_collection.query(
            query_embeddings=[doc_query_embedding.tolist()],
            n_results=k // 2
        )
        
        # 2. Vector search at chunk level
        chunk_query_embedding = self.chunk_embedder.encode(query)
        chunk_results = self.chunk_collection.query(
            query_embeddings=[chunk_query_embedding.tolist()],
            n_results=k
        )
        
        # 3. Graph-based search if ticket context provided
        graph_results = []
        if ticket_context and ticket_context.get('ticket_key'):
            graph_results = self.neo4j_manager.find_related_documents(
                ticket_key=ticket_context['ticket_key'],
                limit=k // 2
            )
        
        # 4. Combine and re-rank results
        all_results = self._combine_search_results(
            doc_results, chunk_results, graph_results, query
        )
        
        # 5. Apply incremental learning if we have feedback
        if ticket_context and ticket_context.get('user_feedback'):
            self._apply_incremental_learning(
                all_results, 
                ticket_context['user_feedback']
            )
        
        return all_results[:k]
    
    def _combine_search_results(self,
                               doc_results: Dict[str, Any],
                               chunk_results: Dict[str, Any], 
                               graph_results: List[Dict[str, Any]],
                               query: str) -> List[Dict[str, Any]]:
        """Combine results from different sources with intelligent ranking"""
        
        combined = {}
        
        # Process document-level results
        if doc_results['ids']:
            for i, doc_id in enumerate(doc_results['ids'][0]):
                if doc_id not in combined:
                    combined[doc_id] = {
                        'id': doc_id,
                        'content': doc_results['documents'][0][i],
                        'metadata': doc_results['metadatas'][0][i],
                        'scores': {
                            'doc_similarity': 1.0 - doc_results['distances'][0][i],
                            'chunk_similarity': 0.0,
                            'graph_relevance': 0.0
                        },
                        'source_types': ['doc_embedding']
                    }
        
        # Process chunk-level results
        if chunk_results['ids']:
            for i, chunk_id in enumerate(chunk_results['ids'][0]):
                doc_id = chunk_results['metadatas'][0][i]['doc_id']
                
                if doc_id not in combined:
                    combined[doc_id] = {
                        'id': doc_id,
                        'content': chunk_results['documents'][0][i],
                        'metadata': chunk_results['metadatas'][0][i],
                        'scores': {
                            'doc_similarity': 0.0,
                            'chunk_similarity': 0.0,
                            'graph_relevance': 0.0
                        },
                        'source_types': []
                    }
                
                # Update chunk similarity (take max)
                chunk_score = 1.0 - chunk_results['distances'][0][i]
                quality_score = chunk_results['metadatas'][0][i].get('quality_score', 0.5)
                adjusted_score = chunk_score * (0.7 + 0.3 * quality_score)  # Quality bonus
                
                combined[doc_id]['scores']['chunk_similarity'] = max(
                    combined[doc_id]['scores']['chunk_similarity'],
                    adjusted_score
                )
                
                if 'chunk_embedding' not in combined[doc_id]['source_types']:
                    combined[doc_id]['source_types'].append('chunk_embedding')
        
        # Process graph results
        for result in graph_results:
            doc_id = result['id']
            
            if doc_id not in combined:
                combined[doc_id] = {
                    'id': doc_id,
                    'content': result['content'],
                    'metadata': {'title': result['title']},
                    'scores': {
                        'doc_similarity': 0.0,
                        'chunk_similarity': 0.0,
                        'graph_relevance': 0.0
                    },
                    'source_types': []
                }
            
            combined[doc_id]['scores']['graph_relevance'] = result['relevance']
            if 'knowledge_graph' not in combined[doc_id]['source_types']:
                combined[doc_id]['source_types'].append('knowledge_graph')
        
        # Calculate final scores
        results = []
        for doc_id, data in combined.items():
            # Weighted combination of scores
            final_score = (
                data['scores']['doc_similarity'] * 0.3 +
                data['scores']['chunk_similarity'] * 0.4 +
                data['scores']['graph_relevance'] * 0.3
            )
            
            # Bonus for multiple sources
            source_bonus = len(data['source_types']) * 0.1
            final_score = min(final_score + source_bonus, 1.0)
            
            results.append({
                'id': doc_id,
                'content': data['content'],
                'metadata': data['metadata'],
                'relevance_score': final_score,
                'score_breakdown': data['scores'],
                'sources': data['source_types']
            })
        
        # Sort by final score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results
    
    def _apply_incremental_learning(self, 
                                   results: List[Dict[str, Any]],
                                   feedback: Dict[str, Any]):
        """Apply user feedback to improve future rankings"""
        
        for result in results:
            doc_id = result['id']
            
            # Update graph relationships based on feedback
            if feedback.get('ticket_key') and feedback.get('helpful_docs'):
                if doc_id in feedback['helpful_docs']:
                    self.neo4j_manager.update_relationship_feedback(
                        doc_id=doc_id,
                        ticket_key=feedback['ticket_key'],
                        helpful=True
                    )
                    
                    # If very helpful, create RESOLVES relationship
                    if feedback.get('resolved_ticket'):
                        self.neo4j_manager.link_document_to_ticket(
                            doc_id=doc_id,
                            ticket_key=feedback['ticket_key'],
                            relationship_type="RESOLVES",
                            confidence=0.9
                        )
    
    def get_recommendations_for_ticket(self, ticket_key: str, 
                                     ticket_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get document recommendations specifically for a ticket"""
        
        # Build enhanced query from ticket data
        query_parts = [
            ticket_data.get('summary', ''),
            ticket_data.get('description', '')[:500],
            ticket_data.get('issue_type', ''),
            f"project {ticket_data.get('project_key', '')}"
        ]
        
        enhanced_query = " ".join(filter(None, query_parts))
        
        # Search with ticket context
        results = self.hybrid_search(
            query=enhanced_query,
            ticket_context={
                'ticket_key': ticket_key,
                'project_key': ticket_data.get('project_key'),
                'issue_type': ticket_data.get('issue_type'),
                'status': ticket_data.get('status')
            }
        )
        
        # Look for solution patterns
        if ticket_data.get('project_key'):
            patterns = self.neo4j_manager.find_solution_patterns(
                project_key=ticket_data['project_key'],
                issue_type=ticket_data.get('issue_type')
            )
            
            # Boost documents that solved similar tickets
            for pattern in patterns:
                for doc in pattern['resolving_documents']:
                    for result in results:
                        if result['id'] == doc['id']:
                            result['relevance_score'] *= 1.2  # Boost
                            result['metadata']['solved_similar'] = True
                            break
        
        # Re-sort after boosting
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results
    
    def add_jira_tickets(self, tickets: List[Dict[str, Any]]):
        """Add Jira tickets to knowledge graph"""
        self.logger.info(f"Adding {len(tickets)} tickets to knowledge graph")
        
        for ticket in tickets:
            try:
                fields = ticket.get('fields', {})
                
                self.neo4j_manager.add_ticket(
                    ticket_key=ticket['key'],
                    summary=fields.get('summary', ''),
                    project_key=fields.get('project', {}).get('key', ''),
                    status=fields.get('status', {}).get('name', ''),
                    metadata={
                        'issue_type': fields.get('issuetype', {}).get('name', ''),
                        'priority': fields.get('priority', {}).get('name', ''),
                        'assignee': fields.get('assignee', {}).get('displayName', '') if fields.get('assignee') else '',
                        'created': fields.get('created', ''),
                        'updated': fields.get('updated', '')
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Failed to add ticket {ticket.get('key')}: {e}")
    
    def update_document_impact_scores(self):
        """Update impact scores for all documents based on their ticket relationships"""
        self.logger.info("Updating document impact scores")
        
        # Get all documents from ChromaDB
        all_docs = self.doc_collection.get()
        
        for i, doc_id in enumerate(all_docs['ids']):
            impact_score = self.neo4j_manager.get_document_impact_score(doc_id)
            
            # Update metadata in ChromaDB
            metadata = all_docs['metadatas'][i]
            metadata['impact_score'] = impact_score
            
            # Update the document
            self.doc_collection.update(
                ids=[doc_id],
                metadatas=[metadata]
            )
        
        self.logger.info("Impact scores updated")

    # In orchestrator/rag/enhanced_rag_pipeline.py, add this method:

    def publish_article_to_confluence(self, 
                                    article: Dict[str, Any], 
                                    ticket_id: str,
                                    project_key: str) -> Dict[str, Any]:
        """Publish an approved article to Confluence"""
        try:
            # Determine space key from project
            space_key = project_key  # Or map to specific space if different
            
            # Create page content in Confluence storage format
            content_html = self._markdown_to_confluence_html(article['content'])
            
            # Create the page
            url = f"{self.confluence_extractor.confluence_url}/rest/api/content"
            
            page_data = {
                "type": "page",
                "title": f"{ticket_id} - {article.get('title', 'Resolution Guide')}",
                "space": {"key": space_key},
                "body": {
                    "storage": {
                        "value": content_html,
                        "representation": "storage"
                    }
                },
                "metadata": {
                    "labels": [
                        {"name": "jurix-generated"},
                        {"name": f"ticket-{ticket_id}"},
                        {"name": "knowledge-article"}
                    ]
                }
            }
            
            response = requests.post(
                url, 
                json=page_data,
                auth=self.confluence_extractor.auth,
                headers={'Content-Type': 'application/json'},
                verify=False
            )
            
            if response.status_code == 200:
                page_info = response.json()
                page_id = page_info['id']
                page_url = f"{self.confluence_extractor.confluence_url}/pages/viewpage.action?pageId={page_id}"
                
                self.logger.info(f"✅ Published article to Confluence: {page_url}")
                
                # Now ingest this document into the RAG system
                confluence_doc = {
                    'id': f"confluence_{page_id}",
                    'title': page_info['title'],
                    'content': article['content'],
                    'space_key': space_key,
                    'created_date': datetime.now().isoformat(),
                    'url': page_url,
                    'metadata': {
                        'source': 'jurix_generated',
                        'ticket_id': ticket_id,
                        'auto_published': True
                    }
                }
                
                # Ingest into RAG
                self.ingest_document(confluence_doc)
                
                # Create the ticket relationship in Neo4j
                self._create_article_ticket_relationship(
                    doc_id=f"confluence_{page_id}",
                    ticket_id=ticket_id,
                    page_url=page_url
                )
                
                return {
                    "status": "success",
                    "page_id": page_id,
                    "page_url": page_url,
                    "doc_id": f"confluence_{page_id}"
                }
            else:
                self.logger.error(f"Failed to publish: {response.text}")
                return {
                    "status": "error",
                    "error": response.text
                }
                
        except Exception as e:
            self.logger.error(f"Error publishing to Confluence: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _markdown_to_confluence_html(self, markdown_content: str) -> str:
        """Convert markdown to Confluence storage format"""
        import markdown
        
        # Basic conversion
        html = markdown.markdown(markdown_content)
        
        # Convert some markdown elements to Confluence macros
        # Headers
        html = re.sub(r'<h1>(.*?)</h1>', r'<h1>\1</h1>', html)
        html = re.sub(r'<h2>(.*?)</h2>', r'<h2>\1</h2>', html)
        
        # Code blocks - convert to Confluence code macro
        html = re.sub(
            r'<pre><code class="language-(\w+)">(.*?)</code></pre>',
            r'<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">\1</ac:parameter><ac:plain-text-body><![CDATA[\2]]></ac:plain-text-body></ac:structured-macro>',
            html,
            flags=re.DOTALL
        )
        
        # Info panels
        html = re.sub(
            r'<blockquote>(.*?)</blockquote>',
            r'<ac:structured-macro ac:name="info"><ac:rich-text-body>\1</ac:rich-text-body></ac:structured-macro>',
            html,
            flags=re.DOTALL
        )
        
        return html

    def _create_article_ticket_relationship(self, doc_id: str, ticket_id: str, page_url: str):
        """Create RESOLVES relationship between article and ticket in Neo4j"""
        try:
            self.logger.info(f"🔗 Creating Neo4j relationship: {ticket_id} -> {doc_id}")
            
            # First, check if Neo4j is connected
            if not self.neo4j_manager:
                self.logger.error("❌ Neo4j manager not initialized!")
                return
                
            # Test Neo4j connection
            try:
                with self.neo4j_manager.driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    self.logger.info("✅ Neo4j connection successful")
            except Exception as e:
                self.logger.error(f"❌ Neo4j connection failed: {e}")
                return
            
            # Ensure the ticket exists in Neo4j
            self.logger.info(f"📝 Ensuring ticket {ticket_id} exists in Neo4j")
            self.neo4j_manager.add_ticket(
                ticket_key=ticket_id,
                summary=f"Ticket {ticket_id}",
                project_key=ticket_id.split('-')[0] if '-' in ticket_id else "PROJ",
                status="Resolved",
                metadata={
                    "has_documentation": True,
                    "documentation_url": page_url,
                    "documentation_id": doc_id
                }
            )
            self.logger.info(f"✅ Ticket {ticket_id} added/updated in Neo4j")
            
            # Ensure the document exists in Neo4j
            self.logger.info(f"📄 Ensuring document {doc_id} exists in Neo4j")
            self.neo4j_manager.add_document(
                doc_id=doc_id,
                title=f"{ticket_id} - Resolution Guide",
                content="Auto-generated article from JURIX",
                metadata={
                    "confluence_url": page_url,
                    "ticket_id": ticket_id,
                    "auto_generated": True
                }
            )
            self.logger.info(f"✅ Document {doc_id} added to Neo4j")
            
            # Create the relationship
            self.logger.info(f"🔗 Creating RESOLVES relationship")
            self.neo4j_manager.link_document_to_ticket(
                doc_id=doc_id,
                ticket_key=ticket_id,
                relationship_type="RESOLVES",
                confidence=1.0,
                metadata={
                    "auto_generated": True,
                    "published_date": datetime.now().isoformat(),
                    "confluence_url": page_url
                }
            )
            self.logger.info(f"✅ RESOLVES relationship created: {ticket_id} -> {doc_id}")
            
            # Verify the relationship was created
            with self.neo4j_manager.driver.session() as session:
                verify_query = """
                MATCH (t:Ticket {key: $ticket_key})-[r:RESOLVES]-(d:Document {id: $doc_id})
                RETURN t.key as ticket, d.id as doc, type(r) as rel_type
                """
                result = session.run(verify_query, ticket_key=ticket_id, doc_id=doc_id)
                record = result.single()
                
                if record:
                    self.logger.info(f"✅ Verified relationship in Neo4j: {record['ticket']} -[{record['rel_type']}]-> {record['doc']}")
                else:
                    self.logger.error(f"❌ Relationship verification failed - not found in Neo4j")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create relationship: {e}", exc_info=True)