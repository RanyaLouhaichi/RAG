from enum import Enum

class QueryType(Enum):
    SEARCH = "search"  
    SUMMARIZE = "summarize"  
    FOLLOWUP = "followup"  
    CONVERSATION = "conversation"  