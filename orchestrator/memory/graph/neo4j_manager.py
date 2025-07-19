import neo4j # type: ignore
from neo4j import GraphDatabase # type: ignore
import logging
from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime
import hashlib

class Neo4jManager:
    """Manages knowledge graph in Neo4j with documents, tickets, and projects"""
    
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.logger = logging.getLogger("Neo4jManager")
        
        # Initialize schema
        self._initialize_schema()
        
    def _initialize_schema(self):
        """Create indexes and constraints"""
        with self.driver.session() as session:
            # Constraints
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Ticket) REQUIRE t.key IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.key IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            ]
            
            # Indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.title)",
                "CREATE INDEX IF NOT EXISTS FOR (t:Ticket) ON (t.status)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.embedding_id)",
                "CREATE INDEX IF NOT EXISTS FOR ()-[r:RESOLVED_BY]-() ON (r.confidence)",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    self.logger.warning(f"Constraint already exists or error: {e}")
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    self.logger.warning(f"Index already exists or error: {e}")
                    
        self.logger.info("Neo4j schema initialized")
    
    def add_document(self, doc_id: str, title: str, content: str, 
                     metadata: Dict[str, Any]) -> str:
        """Add a document to the graph"""
        with self.driver.session() as session:
            query = """
            MERGE (d:Document {id: $doc_id})
            SET d.title = $title,
                d.content = $content,
                d.created_at = datetime(),
                d.updated_at = datetime(),
                d.source = $source,
                d.doc_type = $doc_type,
                d.metadata = $metadata
            RETURN d.id as id
            """
            
            result = session.run(query, 
                doc_id=doc_id,
                title=title,
                content=content[:1000],  # Store preview
                source=metadata.get("source", "confluence"),
                doc_type=metadata.get("type", "article"),
                metadata=json.dumps(metadata)
            )
            
            record = result.single()
            self.logger.info(f"Added document {doc_id} to graph")
            return record["id"]
    
    def add_chunk(self, chunk_id: str, doc_id: str, content: str, 
                  chunk_index: int, embedding_id: str) -> str:
        """Add a document chunk with its embedding reference"""
        with self.driver.session() as session:
            query = """
            MATCH (d:Document {id: $doc_id})
            CREATE (c:Chunk {
                id: $chunk_id,
                content: $content,
                chunk_index: $chunk_index,
                embedding_id: $embedding_id,
                created_at: datetime()
            })
            CREATE (d)-[:HAS_CHUNK {index: $chunk_index}]->(c)
            RETURN c.id as id
            """
            
            result = session.run(query,
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                chunk_index=chunk_index,
                embedding_id=embedding_id
            )
            
            record = result.single()
            return record["id"]
    
    def add_ticket(self, ticket_key: str, summary: str, project_key: str,
                   status: str, metadata: Dict[str, Any]) -> str:
        """Add a ticket to the graph"""
        with self.driver.session() as session:
            # Create ticket and link to project
            query = """
            MERGE (p:Project {key: $project_key})
            ON CREATE SET p.name = $project_key, p.created_at = datetime()
            
            MERGE (t:Ticket {key: $ticket_key})
            SET t.summary = $summary,
                t.status = $status,
                t.updated_at = datetime(),
                t.metadata = $metadata
                
            MERGE (t)-[:BELONGS_TO]->(p)
            RETURN t.key as key
            """
            
            result = session.run(query,
                ticket_key=ticket_key,
                summary=summary,
                project_key=project_key,
                status=status,
                metadata=json.dumps(metadata)
            )
            
            record = result.single()
            self.logger.info(f"Added ticket {ticket_key} to graph")
            return record["key"]
    
    def link_document_to_ticket(self, doc_id: str, ticket_key: str, 
                           relationship_type: str = "REFERENCES",
                           confidence: float = 1.0,
                           metadata: Dict[str, Any] = None):
        """Create a relationship between a document and a ticket"""
        with self.driver.session() as session:
            # First ensure both nodes exist
            ensure_query = """
            MERGE (d:Document {id: $doc_id})
            MERGE (t:Ticket {key: $ticket_key})
            """
            session.run(ensure_query, doc_id=doc_id, ticket_key=ticket_key)
            
            # Then create the relationship
            query = """
            MATCH (d:Document {id: $doc_id})
            MATCH (t:Ticket {key: $ticket_key})
            MERGE (d)-[r:%s {confidence: $confidence}]->(t)
            SET r.created_at = datetime()
            SET r.metadata = $metadata
            RETURN d, r, t
            """ % relationship_type  # Using string formatting for relationship type
            
            result = session.run(
                query, 
                doc_id=doc_id, 
                ticket_key=ticket_key,
                confidence=confidence,
                metadata=metadata or {}
            )
            
            record = result.single()
            if record:
                self.logger.info(f"Created {relationship_type} relationship: {doc_id} -> {ticket_key}")
            else:
                self.logger.error(f"Failed to create relationship: {doc_id} -> {ticket_key}")
        
        return record
    
    def find_related_documents(self, ticket_key: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find documents related to a ticket through various paths"""
        with self.driver.session() as session:
            query = """
            MATCH (t:Ticket {key: $ticket_key})
            OPTIONAL MATCH (t)<-[:REFERENCES|RESOLVES|MENTIONS]-(d:Document)
            OPTIONAL MATCH (t)-[:BELONGS_TO]->(:Project)<-[:BELONGS_TO]-(similar:Ticket)
                          <-[:REFERENCES|RESOLVES]-(d2:Document)
            WHERE similar.key <> t.key AND similar.status = t.status
            
            WITH COLLECT(DISTINCT d) + COLLECT(DISTINCT d2) as documents
            UNWIND documents as doc
            
            WITH doc, 
                 CASE 
                   WHEN (doc)-[:RESOLVES]->(:Ticket {key: $ticket_key}) THEN 1.0
                   WHEN (doc)-[:REFERENCES]->(:Ticket {key: $ticket_key}) THEN 0.8
                   ELSE 0.5
                 END as relevance
                 
            WHERE doc IS NOT NULL
            RETURN doc.id as id, 
                   doc.title as title, 
                   doc.content as content,
                   relevance
            ORDER BY relevance DESC
            LIMIT $limit
            """
            
            result = session.run(query, ticket_key=ticket_key, limit=limit)
            
            documents = []
            for record in result:
                documents.append({
                    "id": record["id"],
                    "title": record["title"],
                    "content": record["content"],
                    "relevance": record["relevance"],
                    "source": "knowledge_graph"
                })
            
            return documents
    
    def find_solution_patterns(self, project_key: str, issue_type: str = None) -> List[Dict[str, Any]]:
        """Find common solution patterns in a project"""
        with self.driver.session() as session:
            query = """
            MATCH (p:Project {key: $project_key})<-[:BELONGS_TO]-(t:Ticket)
            WHERE t.status IN ['Done', 'Resolved', 'Closed']
            AND ($issue_type IS NULL OR t.metadata CONTAINS $issue_type)
            
            MATCH (t)<-[:RESOLVES]-(d:Document)
            
            WITH t, COLLECT(d) as docs
            RETURN t.key as ticket_key,
                   t.summary as summary,
                   [d IN docs | {id: d.id, title: d.title}] as resolving_documents
            ORDER BY t.updated_at DESC
            LIMIT 10
            """
            
            result = session.run(query, 
                project_key=project_key,
                issue_type=issue_type
            )
            
            patterns = []
            for record in result:
                patterns.append({
                    "ticket_key": record["ticket_key"],
                    "summary": record["summary"],
                    "resolving_documents": record["resolving_documents"]
                })
            
            return patterns
    
    def get_document_impact_score(self, doc_id: str) -> float:
        """Calculate how many tickets a document has helped resolve"""
        with self.driver.session() as session:
            query = """
            MATCH (d:Document {id: $doc_id})
            OPTIONAL MATCH (d)-[:RESOLVES]->(t:Ticket)
            WHERE t.status IN ['Done', 'Resolved', 'Closed']
            
            WITH d, COUNT(DISTINCT t) as resolved_count
            
            OPTIONAL MATCH (d)-[:REFERENCES]->(t2:Ticket)
            WITH d, resolved_count, COUNT(DISTINCT t2) as referenced_count
            
            RETURN resolved_count * 1.0 + referenced_count * 0.5 as impact_score
            """
            
            result = session.run(query, doc_id=doc_id)
            record = result.single()
            
            return record["impact_score"] if record else 0.0
        
    def update_ticket_metadata(self, ticket_key: str, metadata: Dict[str, Any]):
        """Update ticket metadata"""
        with self.driver.session() as session:
            query = """
            MATCH (t:Ticket {key: $ticket_key})
            SET t += $metadata
            SET t.updated_at = datetime()
            RETURN t
            """
            
            result = session.run(query, ticket_key=ticket_key, metadata=metadata)
            return result.single()

    def get_ticket_documentation(self, ticket_key: str) -> List[Dict[str, Any]]:
        """Get all documentation for a ticket"""
        with self.driver.session() as session:
            query = """
            MATCH (t:Ticket {key: $ticket_key})-[r:RESOLVES]-(d:Document)
            RETURN d.id as doc_id, 
                d.title as title, 
                r.confidence as confidence,
                r.metadata as relationship_metadata,
                d.metadata as doc_metadata
            ORDER BY r.confidence DESC
            """
            
            result = session.run(query, ticket_key=ticket_key)
            
            docs = []
            for record in result:
                docs.append({
                    "doc_id": record["doc_id"],
                    "title": record["title"],
                    "confidence": record["confidence"],
                    "relationship_metadata": record["relationship_metadata"],
                    "doc_metadata": record["doc_metadata"]
                })
            
            return docs
    
    def update_relationship_feedback(self, doc_id: str, ticket_key: str, 
                                   helpful: bool):
        """Update relationship strength based on feedback"""
        with self.driver.session() as session:
            # Adjust confidence based on feedback
            adjustment = 0.1 if helpful else -0.1
            
            query = """
            MATCH (d:Document {id: $doc_id})-[r:REFERENCES|RESOLVES]->(t:Ticket {key: $ticket_key})
            SET r.confidence = CASE 
                WHEN r.confidence + $adjustment > 1.0 THEN 1.0
                WHEN r.confidence + $adjustment < 0.0 THEN 0.0
                ELSE r.confidence + $adjustment
            END,
            r.feedback_count = COALESCE(r.feedback_count, 0) + 1,
            r.last_feedback = datetime()
            """
            
            session.run(query,
                doc_id=doc_id,
                ticket_key=ticket_key,
                adjustment=adjustment
            )
    
    def close(self):
        """Close the driver connection"""
        self.driver.close()