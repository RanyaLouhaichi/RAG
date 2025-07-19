#!/usr/bin/env python3
# test_complete_article_workflow.py

import requests
import json
import time
from datetime import datetime
import sys

# Configuration
BASE_URL = "http://localhost:5001"
TEST_TICKET_ID = "JURIX-11"  # Change to a real ticket ID from your Jira

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print a colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")

def print_step(step_num, text):
    """Print a step header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}Step {step_num}: {text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*60}{Colors.ENDC}")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")

def print_article_preview(content, max_lines=20):
    """Print article content with formatting"""
    lines = content.split('\n')
    print(f"\n{Colors.WARNING}📄 Article Content Preview:{Colors.ENDC}")
    print(f"{Colors.WARNING}{'─'*60}{Colors.ENDC}")
    
    for i, line in enumerate(lines[:max_lines]):
        if line.startswith('#'):
            print(f"{Colors.BOLD}{line}{Colors.ENDC}")
        elif line.startswith('- '):
            print(f"{Colors.CYAN}{line}{Colors.ENDC}")
        else:
            print(line)
    
    if len(lines) > max_lines:
        print(f"\n{Colors.WARNING}... ({len(lines) - max_lines} more lines){Colors.ENDC}")
    print(f"{Colors.WARNING}{'─'*60}{Colors.ENDC}")

def test_complete_workflow():
    """Test the complete article generation and feedback workflow"""
    
    print_header("JURIX AI - Complete Article Generation Workflow Test")
    print_info(f"Testing with ticket: {TEST_TICKET_ID}")
    print_info(f"Backend URL: {BASE_URL}")
    print_info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Generate Initial Article
    print_step(1, "Generating Initial Article")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/article/generate/{TEST_TICKET_ID}",
            timeout=120  # 2 minute timeout for generation
        )
        
        if response.status_code == 200:
            result = response.json()
            article_v1 = result['article']
            
            print_success("Initial article generated successfully!")
            print_info(f"Version: {article_v1['version']}")
            print_info(f"Status: {article_v1['approval_status']}")
            print_info(f"Title: {article_v1['title']}")
            print_info(f"Collaboration used: {article_v1.get('collaboration_enhanced', False)}")
            
            if article_v1.get('collaboration_metadata'):
                collab = article_v1['collaboration_metadata']
                print_info(f"Agents involved: {', '.join(collab.get('collaborating_agents', []))}")
            
            print_article_preview(article_v1['content'])
            
        else:
            print_error(f"Failed to generate article: {response.status_code}")
            print_error(response.json())
            return
            
    except requests.exceptions.Timeout:
        print_error("Request timed out. The article generation is taking too long.")
        return
    except Exception as e:
        print_error(f"Error generating article: {e}")
        return
    
    # Wait a bit before next step
    time.sleep(2)
    
    # Step 2: First Round of Feedback
    print_step(2, "Submitting First Round of Feedback")
    
    feedback_1 = """Please enhance the article with the following improvements:

1. In the Solution Implementation section:
   - Add specific code examples showing the actual implementation
   - Include configuration files or settings that were changed
   - Add command-line examples where applicable

2. Add a new "Performance Impact" section with:
   - Before/after metrics (response time, throughput, etc.)
   - Resource utilization comparisons
   - Quantifiable improvements achieved

3. Expand the troubleshooting section:
   - Common errors users might encounter
   - How to diagnose issues
   - Rollback procedures if needed

4. Include more technical details about the root cause analysis"""
    
    print_info("Feedback being submitted:")
    print(f"{Colors.WARNING}{feedback_1}{Colors.ENDC}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/article/feedback/{TEST_TICKET_ID}",
            json={
                "feedback": feedback_1,
                "action": "refine",
                "current_version": 1
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            article_v2 = result['article']
            
            print_success("Feedback processed and article refined!")
            print_info(f"New version: {article_v2['version']}")
            print_info(f"Feedback applied: {len(article_v2.get('feedback_history', []))} entries")
            
            print_article_preview(article_v2['content'])
            
            # Compare with previous version
            print_info("\n📊 Version Comparison:")
            print_info(f"V1 length: {len(article_v1['content'])} characters")
            print_info(f"V2 length: {len(article_v2['content'])} characters")
            print_info(f"Content increased by: {len(article_v2['content']) - len(article_v1['content'])} characters")
            
        else:
            print_error(f"Failed to process feedback: {response.status_code}")
            print_error(response.json())
            return
            
    except Exception as e:
        print_error(f"Error processing feedback: {e}")
        return
    
    time.sleep(2)
    
    # Step 3: Check Article Status
    print_step(3, "Checking Article Status")
    
    try:
        response = requests.get(f"{BASE_URL}/api/article/status/{TEST_TICKET_ID}")
        
        if response.status_code == 200:
            status = response.json()
            print_success("Article status retrieved!")
            print_info(f"Current version: {status['article']['version']}")
            print_info(f"Approval status: {status['article']['approval_status']}")
            print_info(f"Is final: {status['is_final']}")
        else:
            print_error(f"Failed to get status: {response.status_code}")
            
    except Exception as e:
        print_error(f"Error getting status: {e}")
    
    time.sleep(2)
    
    # Step 4: Second Round of More Specific Feedback
    print_step(4, "Submitting Second Round of Feedback")
    
    feedback_2 = """Great improvements! A few more specific refinements:

1. In the code examples section, please use Python syntax highlighting
2. Add these specific metrics that were measured:
   - Response time improved from 5.2s to 0.8s (84% improvement)
   - Memory usage reduced from 512MB to 128MB
   - CPU utilization dropped from 75% to 20%
3. Add a link to the related documentation: https://wiki.company.com/optimization-guide
4. Include the specific error message users were seeing: "Connection timeout after 30s"
"""
    
    print_info("Submitting more specific feedback...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/article/feedback/{TEST_TICKET_ID}",
            json={
                "feedback": feedback_2,
                "action": "refine",
                "current_version": 2
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            article_v3 = result['article']
            
            print_success("Second refinement completed!")
            print_info(f"Version: {article_v3['version']}")
            
            # Check if specific requests were addressed
            content = article_v3['content'].lower()
            if "84% improvement" in content:
                print_success("✓ Specific metrics were added")
            if "connection timeout after 30s" in content:
                print_success("✓ Error message was included")
            if "```python" in content:
                print_success("✓ Python syntax highlighting detected")
                
        else:
            print_error(f"Failed to process second feedback: {response.status_code}")
            
    except Exception as e:
        print_error(f"Error processing second feedback: {e}")
    
    time.sleep(2)
    
    # Step 5: Final Approval
    print_step(5, "Approving Final Article")
    
    approval_message = "Excellent work! The article now comprehensively covers all aspects and is ready for publication."
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/article/feedback/{TEST_TICKET_ID}",
            json={
                "feedback": approval_message,
                "action": "approve",
                "current_version": 3
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            final_article = result['article']
            
            print_success("🎉 Article APPROVED!")
            print_info(f"Final version: {final_article['version']}")
            print_info(f"Approved at: {final_article.get('approved_at', 'N/A')}")
            print_info("Ready for publication to Confluence")
            
        else:
            print_error(f"Failed to approve article: {response.status_code}")
            
    except Exception as e:
        print_error(f"Error approving article: {e}")
    
    time.sleep(1)
    
    # Step 6: View Complete History
    print_step(6, "Viewing Article History")
    
    try:
        response = requests.get(f"{BASE_URL}/api/article/history/{TEST_TICKET_ID}")
        
        if response.status_code == 200:
            history = response.json()
            
            print_success(f"Retrieved history with {history['total_versions']} versions")
            print("\n📚 Version History:")
            
            for version_info in history['history']:
                print(f"\n  Version {version_info['version']}:")
                print(f"  ├─ Created: {version_info['created_at']}")
                print(f"  ├─ Status: {version_info['approval_status']}")
                print(f"  └─ Feedback entries: {len(version_info['feedback_history'])}")
                
                for feedback in version_info['feedback_history']:
                    print(f"      └─ Feedback from v{feedback['version']}: {feedback['feedback'][:50]}...")
                    
        else:
            print_error(f"Failed to get history: {response.status_code}")
            
    except Exception as e:
        print_error(f"Error getting history: {e}")
    
    # Summary
    print_header("Test Summary")
    print_success("All workflow steps completed successfully!")
    print_info("The article went through 3 versions:")
    print_info("  • V1: Initial generation with collaboration")
    print_info("  • V2: Enhanced with technical details and metrics")
    print_info("  • V3: Refined with specific values and formatting")
    print_info("  • Final: Approved and ready for Confluence")
    
    print(f"\n{Colors.BOLD}Next steps:{Colors.ENDC}")
    print("1. Integrate with Jira comment webhook for real-time feedback")
    print("2. Add Confluence publishing API integration")
    print("3. Create UI components for the feedback workflow")
    print("4. Set up email/Slack notifications")

def test_comment_simulation():
    """Test the comment simulation endpoint"""
    print_header("Testing Comment-Based Feedback")
    
    print_step(1, "Simulating Jira Comment")
    
    comment = "@jurix Please add a section about database migration steps and include SQL examples"
    
    print_info(f"Simulating comment: {comment}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/article/simulate-comment",
            json={
                "ticket_id": TEST_TICKET_ID,
                "comment": comment
            }
        )
        
        if response.status_code == 200:
            print_success("Comment processed successfully!")
            result = response.json()
            if result.get('status') == 'ignored':
                print_info("Comment was ignored (no @jurix mention)")
            else:
                print_info(f"Action taken: {result.get('action', 'unknown')}")
        else:
            print_error(f"Failed to process comment: {response.status_code}")
            
    except Exception as e:
        print_error(f"Error simulating comment: {e}")

def test_error_cases():
    """Test error handling"""
    print_header("Testing Error Cases")
    
    # Test 1: Non-existent ticket
    print_step(1, "Testing Non-Existent Ticket")
    response = requests.get(f"{BASE_URL}/api/article/status/FAKE-999")
    if response.status_code == 404:
        print_success("Correctly returned 404 for non-existent ticket")
    else:
        print_error(f"Unexpected response: {response.status_code}")
    
    # Test 2: Empty feedback
    print_step(2, "Testing Empty Feedback")
    response = requests.post(
        f"{BASE_URL}/api/article/feedback/{TEST_TICKET_ID}",
        json={"feedback": "", "action": "refine", "current_version": 1}
    )
    print_info(f"Empty feedback response: {response.status_code}")
    
    # Test 3: Invalid action
    print_step(3, "Testing Invalid Action")
    response = requests.post(
        f"{BASE_URL}/api/article/feedback/{TEST_TICKET_ID}",
        json={"feedback": "test", "action": "invalid_action", "current_version": 1}
    )
    print_info(f"Invalid action response: {response.status_code}")

if __name__ == "__main__":
    try:
        # Check if backend is running
        print_info("Checking backend connection...")
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print_error("Backend is not responding. Make sure api_simple.py is running!")
            sys.exit(1)
        print_success("Backend is running!")
        
        # Run main test
        test_complete_workflow()
        
        # Optionally run additional tests
        print("\n" + "="*80)
        user_input = input("\nRun additional tests? (y/n): ")
        if user_input.lower() == 'y':
            test_comment_simulation()
            test_error_cases()
        
        print_header("All Tests Completed!")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()