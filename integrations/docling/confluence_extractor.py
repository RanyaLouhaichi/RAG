import requests
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import re
from bs4 import BeautifulSoup
import hashlib
import json

class ConfluenceDoclingExtractor:
    """Extract and process Confluence documents using BeautifulSoup (simplified from Docling)"""
    
    def __init__(self, confluence_url: str, username: str, password: str):
        self.confluence_url = confluence_url.rstrip('/')
        self.auth = (username, password)
        self.logger = logging.getLogger("ConfluenceDoclingExtractor")
        
    def get_all_spaces(self) -> List[Dict[str, Any]]:
        """Get all Confluence spaces"""
        url = f"{self.confluence_url}/rest/api/space"
        
        try:
            response = requests.get(url, auth=self.auth, verify=False)  # Add verify=False for local instances
            response.raise_for_status()
            
            data = response.json()
            spaces = []
            
            for space in data.get('results', []):
                spaces.append({
                    'key': space['key'],
                    'name': space['name'],
                    'type': space['type']
                })
            
            self.logger.info(f"Found {len(spaces)} spaces")
            return spaces
            
        except Exception as e:
            self.logger.error(f"Failed to get spaces: {e}")
            return []
    
    def get_space_content(self, space_key: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all content from a space"""
        url = f"{self.confluence_url}/rest/api/content"
        params = {
            'spaceKey': space_key,
            'limit': limit,
            'expand': 'body.storage,version,space'
        }
        
        try:
            response = requests.get(url, auth=self.auth, params=params, verify=False)
            response.raise_for_status()
            
            data = response.json()
            return data.get('results', [])
            
        except Exception as e:
            self.logger.error(f"Failed to get space content: {e}")
            return []
    
    def extract_document(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Extract a single Confluence document"""
        url = f"{self.confluence_url}/rest/api/content/{content_id}"
        params = {
            'expand': 'body.storage,version,space,ancestors'
        }
        
        try:
            response = requests.get(url, auth=self.auth, params=params, verify=False)
            response.raise_for_status()
            
            content = response.json()
            
            # Extract HTML content
            html_content = content.get('body', {}).get('storage', {}).get('value', '')
            
            if not html_content:
                return None
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract text and structure
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Extract specific elements
            tables = self._extract_tables(soup)
            code_blocks = self._extract_code_blocks(soup)
            images = self._extract_images(soup)
            links = self._extract_links(soup)
            
            # Generate document ID
            doc_id = f"confluence_{content['id']}"
            
            # Create document structure
            document = {
                'id': doc_id,
                'title': content.get('title', ''),
                'content': text_content,
                'html_content': html_content,
                'space_key': content.get('space', {}).get('key', ''),
                'space_name': content.get('space', {}).get('name', ''),
                'version': content.get('version', {}).get('number', 1),
                'created_date': content.get('version', {}).get('when', ''),
                'created_by': content.get('version', {}).get('by', {}).get('displayName', ''),
                'url': f"{self.confluence_url}/pages/viewpage.action?pageId={content_id}",
                'ancestors': [
                    {'id': a['id'], 'title': a['title']} 
                    for a in content.get('ancestors', [])
                ],
                'extracted_elements': {
                    'tables': tables,
                    'code_blocks': code_blocks,
                    'images': images,
                    'links': links
                },
                'metadata': {
                    'source': 'confluence',
                    'type': content.get('type', 'page'),
                    'extraction_date': datetime.now().isoformat()
                }
            }
            
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to extract document {content_id}: {e}")
            return None
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract tables from HTML"""
        tables = []
        
        for table in soup.find_all('table'):
            headers = []
            rows = []
            
            # Extract headers
            for th in table.find_all('th'):
                headers.append(th.get_text(strip=True))
            
            # Extract rows
            for tr in table.find_all('tr'):
                cells = []
                for td in tr.find_all('td'):
                    cells.append(td.get_text(strip=True))
                if cells:
                    rows.append(cells)
            
            if headers or rows:
                tables.append({
                    'headers': headers,
                    'rows': rows
                })
        
        return tables
    
    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract code blocks"""
        code_blocks = []
        
        # Confluence uses ac:structured-macro for code blocks
        for macro in soup.find_all('ac:structured-macro', {'ac:name': 'code'}):
            code_elem = macro.find('ac:plain-text-body')
            if code_elem:
                language = 'unknown'
                for param in macro.find_all('ac:parameter', {'ac:name': 'language'}):
                    language = param.get_text(strip=True)
                
                code_blocks.append({
                    'code': code_elem.get_text(strip=True),
                    'language': language
                })
        
        # Also check for regular pre tags
        for pre in soup.find_all('pre'):
            code = pre.get_text(strip=True)
            if code and not any(code in cb['code'] for cb in code_blocks):
                language = pre.get('class', [''])[0] if pre.get('class') else 'unknown'
                code_blocks.append({
                    'code': code,
                    'language': language
                })
        
        return code_blocks
    
    def _extract_images(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract image references"""
        images = []
        
        # Regular img tags
        for img in soup.find_all('img'):
            images.append({
                'src': img.get('src', ''),
                'alt': img.get('alt', ''),
                'title': img.get('title', '')
            })
        
        # Confluence image macros
        for macro in soup.find_all('ac:image'):
            url = macro.find('ri:url')
            if url:
                images.append({
                    'src': url.get('ri:value', ''),
                    'alt': '',
                    'title': ''
                })
        
        return images
    
    def _extract_links(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract links and references"""
        links = []
        
        for a in soup.find_all('a'):
            href = a.get('href', '')
            
            # Classify link type
            link_type = 'external'
            if '/pages/viewpage.action' in href:
                link_type = 'confluence_page'
            elif '/browse/' in href:
                link_type = 'jira_ticket'
            
            links.append({
                'href': href,
                'text': a.get_text(strip=True),
                'type': link_type
            })
        
        return links
    
    def extract_space_documents(self, space_key: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Extract all documents from a Confluence space"""
        self.logger.info(f"Extracting documents from space: {space_key}")
        
        # Get all content in space
        contents = self.get_space_content(space_key, limit)
        
        if not contents:
            self.logger.warning(f"No content found in space {space_key}")
            return []
        
        documents = []
        for content in contents:
            try:
                doc = self.extract_document(content['id'])
                if doc:
                    documents.append(doc)
                    self.logger.info(f"Extracted: {doc['title']}")
            except Exception as e:
                self.logger.error(f"Failed to extract content {content.get('id')}: {e}")
                continue
        
        self.logger.info(f"Extracted {len(documents)} documents from {space_key}")
        return documents
    
    def find_ticket_references(self, document: Dict[str, Any]) -> List[str]:
        """Find Jira ticket references in document"""
        ticket_pattern = r'[A-Z]{2,}-\d+'
        
        tickets = set()
        
        # Search in content
        content = document.get('content', '')
        tickets.update(re.findall(ticket_pattern, content))
        
        # Search in links
        for link in document.get('extracted_elements', {}).get('links', []):
            if link['type'] == 'jira_ticket':
                match = re.search(ticket_pattern, link['href'])
                if match:
                    tickets.add(match.group())
            # Also check link text
            tickets.update(re.findall(ticket_pattern, link.get('text', '')))
        
        return list(tickets)