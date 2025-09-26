# Confluence Extractor Documentation

## Overview

The `ConfluenceDoclingExtractor` class is a tool for extracting and processing documents from Confluence. It can retrieve all spaces and their content, and then extract the title, content, and other metadata from each document. It uses `BeautifulSoup` to parse the HTML content and extract text, tables, code blocks, images, and links.

## `ConfluenceDoclingExtractor` Class

### Initialization

The class is initialized with the Confluence URL and authentication credentials.

### Methods

- **`get_all_spaces()`**: Retrieves a list of all spaces in the Confluence instance.

- **`get_space_content(space_key: str, limit: int = 100)`**: Retrieves the content of a specific space, with an optional limit on the number of documents to retrieve.

- **`extract_document(content_id: str)`**: This is the core method of the class. It takes a content ID as input and extracts the document's title, content, and other metadata. It uses `BeautifulSoup` to parse the HTML content and extract text, tables, code blocks, images, and links.

- **`extract_space_documents(space_key: str, limit: int = 100)`**: Extracts all documents from a given space, with an optional limit.

- **`find_ticket_references(document: Dict[str, Any])`**: Finds all Jira ticket references in a given document.

## Usage

The `ConfluenceDoclingExtractor` can be used to build a knowledge base from Confluence pages. For example:

```python
from integrations.docling.confluence_extractor import ConfluenceDoclingExtractor

extractor = ConfluenceDoclingExtractor(confluence_url, username, password)
spaces = extractor.get_all_spaces()
for space in spaces:
    documents = extractor.extract_space_documents(space['key'])
    # Process the documents...
```
