# test_aria_demo.py
# FIXED VERSION - Works with synchronous ARIA

from datetime import datetime
from orchestrator.core.aria_orchestrator import aria, ARIAContext, ARIAPersonality # type: ignore
import time

class ARIADemo:
    """Simple demo interface to test ARIA"""
    
    def __init__(self):
        self.current_context = ARIAContext(
            user_intent="",
            workspace="jira",
            current_project="PROJ123",
            current_ticket=None,
            user_history=[],
            active_insights=[],
            mood=ARIAPersonality.HELPFUL
        )
    
    def run_demo(self):
        """Run the interactive ARIA demo"""
        print("\n" + "="*60)
        print("🤖 ARIA DEMO - Your AI Team Member")
        print("="*60)
        
        # ARIA introduces herself
        intro = aria.introduce_myself({})
        print(f"\n{intro['message']}")
        print(f"\nCapabilities: {', '.join(intro['capabilities'])}")
        print("\n" + "-"*60)
        
        # Demo scenarios
        while True:
            print("\n📋 DEMO SCENARIOS:")
            print("1. 📊 Show Live Dashboard (Jira AI Insights Tab)")
            print("2. ✅ Simulate Ticket Resolution (Auto-documentation)")
            print("3. 💬 Chat with ARIA")
            print("4. 🔮 Get Predictions")
            print("5. 🎯 Full Demo Flow")
            print("6. 🚪 Exit")
            
            choice = input("\nChoose scenario (1-6): ").strip()
            
            if choice == "1":
                self.demo_live_dashboard()
            elif choice == "2":
                self.demo_ticket_resolution()
            elif choice == "3":
                self.demo_chat()
            elif choice == "4":
                self.demo_predictions()
            elif choice == "5":
                self.demo_full_flow()
            elif choice == "6":
                print("\n👋 ARIA: Goodbye! Happy to help anytime!")
                break
    
    def demo_live_dashboard(self):
        """Demo 1: Live Dashboard in Jira"""
        print("\n" + "="*60)
        print("📊 DEMO: Live Dashboard in Jira AI Insights Tab")
        print("="*60)
        
        print("\n[Simulating: User clicks on 'AI Insights' tab in Jira]")
        time.sleep(1)
        
        print("\n🤖 ARIA: Loading your real-time dashboard...")
        
        # Get dashboard data
        dashboard = aria._provide_live_dashboard(self.current_context)
        
        # Display dashboard
        print("\n" + "-"*40)
        print("🎯 REAL-TIME INTELLIGENCE DASHBOARD")
        print("-"*40)
        
        health = dashboard["data"]["sprint_health"]
        print(f"\n📈 Sprint Health: {'█' * (health['score'] // 10)}{'░' * (10 - health['score'] // 10)} {health['percentage']}")
        
        metrics = dashboard["data"]["live_metrics"]
        print(f"\n📊 Live Metrics (updates every 30s):")
        print(f"  ├─ Velocity: {metrics['velocity']} tickets/week")
        print(f"  ├─ Cycle Time: {metrics['cycle_time']:.1f} days")
        print(f"  └─ Bottlenecks: {len(metrics['bottlenecks'])} detected")
        
        print(f"\n🔮 AI Predictions:")
        for pred in dashboard["data"]["predictions"][:2]:
            print(f"  • {pred['message']}")
            print(f"    → {pred['suggestion']}")
        
        print(f"\n💬 ARIA: {dashboard['aria_message']}")
        
        time.sleep(2)
    
    def demo_ticket_resolution(self):
        """Demo 2: Automatic Documentation"""
        print("\n" + "="*60)
        print("✅ DEMO: Automatic Documentation on Ticket Resolution")
        print("="*60)
        
        ticket_id = input("\nEnter ticket ID (or press Enter for TICKET-001): ") or "TICKET-001"
        self.current_context.current_ticket = ticket_id
        
        print(f"\n[Simulating: User changes {ticket_id} status to 'Done']")
        time.sleep(1)
        
        print("\n🔔 Popup appears:")
        print("┌─────────────────────────────┐")
        print("│ 🤖 ARIA is processing...    │")
        print("│ Creating documentation...    │")
        print("└─────────────────────────────┘")
        
        # Process ticket resolution
        result = aria._handle_ticket_resolution(ticket_id, self.current_context)
        
        time.sleep(2)
        
        print("\n✅ Popup updates:")
        print("┌─────────────────────────────┐")
        print(f"│ {result['message']}      │")
        print("│                             │")
        print("│ [View Documentation]        │")
        print("│ [Ask ARIA about it]         │")
        print("└─────────────────────────────┘")
        
        print(f"\n📄 Article Preview:")
        print(f"{result['article_preview']}")
        
        if result['collaboration_applied']:
            print(f"\n🤝 Collaboration Intelligence Applied!")
    
    def demo_chat(self):
        """Demo 3: Chat with ARIA"""
        print("\n" + "="*60)
        print("💬 DEMO: Chat with ARIA")
        print("="*60)
        print("(Type 'back' to return to menu)")
        
        while True:
            user_input = input("\n💭 You: ").strip()
            
            if user_input.lower() == 'back':
                break
            
            print("\n🤖 ARIA: ", end="", flush=True)
            
            # Process chat
            response = aria.process_interaction(user_input, self.current_context)
            
            # Simulate typing
            message = response['message']
            for i, char in enumerate(message):
                print(char, end="", flush=True)
                if i % 50 == 0:  # Faster typing simulation
                    time.sleep(0.01)
            
            if response.get('suggested_actions'):
                print("\n\n📎 Suggested actions:", end="")
                for action in response['suggested_actions']:
                    print(f" [{action['label']}]", end="")
            
            # Update context
            self.current_context.user_history.append({
                "timestamp": datetime.now(),
                "input": user_input,
                "response": response['message']
            })
    
    def demo_predictions(self):
        """Demo 4: Predictive Insights"""
        print("\n" + "="*60)
        print("🔮 DEMO: ARIA's Predictive Insights")
        print("="*60)
        
        print("\n🤖 ARIA: Analyzing your project patterns...")
        time.sleep(1)
        
        predictions = aria._provide_predictions(self.current_context)
        
        print(f"\n{predictions['message']}")
        print("\n" + "-"*40)
        
        for pred in predictions['predictions']:
            print(f"\n🎯 {pred['type'].upper().replace('_', ' ')}")
            print(f"   Confidence: {'█' * int(pred['confidence'] * 10)}{'░' * (10 - int(pred['confidence'] * 10))} {pred['confidence']:.0%}")
            print(f"   {pred['message']}")
            print(f"   💡 {pred['suggestion']}")
        
        print(f"\n🤖 ARIA: {predictions['aria_advice']}")
    
    def demo_full_flow(self):
        """Demo 5: The Complete Flow"""
        print("\n" + "="*60)
        print("🎭 FULL DEMO: ARIA in Action")
        print("="*60)
        
        print('\n📍 "Let me show you JURIX - AI that lives where you work..."')
        time.sleep(2)
        
        # Step 1: Dashboard
        print('\n1️⃣ "Here\'s a normal Jira project, but watch this AI Insights tab..."')
        time.sleep(1)
        print("   → Click → Beautiful dashboard animates open")
        time.sleep(1)
        self.demo_live_dashboard()
        
        # Step 2: Ticket Resolution
        print('\n2️⃣ "Now watch what happens when I resolve a ticket..."')
        time.sleep(2)
        self.current_context.current_ticket = "TICKET-123"
        self.demo_ticket_resolution()
        
        # Step 3: Real-time Update
        print('\n3️⃣ "But here\'s the magic - watch the dashboard..."')
        time.sleep(1)
        print("   → Charts update live")
        print("   → 🔮 New Prediction appears: 'Sprint completion risk reduced!'")
        print("   → 📈 Velocity increased by 1 ticket")
        
        # Step 4: Ask ARIA
        print('\n4️⃣ "And if I need help..."')
        time.sleep(1)
        print('   💭 Type: "ARIA, why is the sprint at risk?"')
        time.sleep(1)
        
        response = aria.process_interaction(
            "Why is the sprint at risk?", 
            self.current_context
        )
        print(f"\n🤖 ARIA: {response['message']}")
        
        print("\n" + "="*60)
        print("✨ No new tools to learn. No tabs to switch.")
        print("   Just intelligence where you already work.")
        print("="*60)

def main():
    """Run the ARIA demo"""
    demo = ARIADemo()
    demo.run_demo()

if __name__ == "__main__":
    main()