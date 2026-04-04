"""
Phase 6: Manual ReAct Agent for GraphMed
Implements ReAct pattern (Reasoning + Acting) without external dependencies
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import PatientKnowledgeGraph
from src.memory import GlobalMemoryManager
from src.medical_kb import MedicalKnowledgeBase

import os
from dotenv import load_dotenv

load_dotenv()


class ReActAgent:
    """
    Manual ReAct Agent for clinical reasoning.
    Implements Thought -> Action -> Observation loop.
    """
    
    def __init__(self, persist_dir: str = "data", max_steps: int = 5):
        """
        Initialize the ReAct agent.
        
        Args:
            persist_dir: Base directory for persisted data
            max_steps: Maximum reasoning steps per query
        """
        self.persist_dir = Path(persist_dir)
        self.max_steps = max_steps
        
        # Initialize LLM
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=1000
        )
        
        # Initialize components
        self.graphs_dir = self.persist_dir / "graphs"
        self.memory_manager = GlobalMemoryManager(str(self.persist_dir / "chroma_db"))
        self.medical_kb = MedicalKnowledgeBase(str(self.persist_dir / "medical_kb"))
        
        # Cache for loaded graphs
        self._loaded_graphs = {}
        
        print("✅ ReAct Agent initialized (manual implementation)")
        print(f"   Max reasoning steps: {max_steps}")
    
    def _load_patient_graph(self, patient_id: str) -> PatientKnowledgeGraph:
        """Load a patient's knowledge graph."""
        if patient_id in self._loaded_graphs:
            return self._loaded_graphs[patient_id]
        
        graph_path = self.graphs_dir / f"{patient_id}_graph.json"
        if graph_path.exists():
            pkg = PatientKnowledgeGraph.load(str(graph_path))
            self._loaded_graphs[patient_id] = pkg
            return pkg
        else:
            return PatientKnowledgeGraph(patient_id)
    
    def _execute_action(self, action: str, params: Dict) -> str:
        """
        Execute a tool/action and return observation.
        
        Args:
            action: Name of the action/tool to execute
            params: Parameters for the action
        
        Returns:
            Observation result
        """
        print(f"   🔧 Executing: {action}({params})")
        
        try:
            if action == "query_conditions":
                pkg = self._load_patient_graph(params.get("patient_id"))
                entities = pkg.get_entities_by_type("CONDITION")
                if not entities:
                    return "No conditions found in patient record."
                return f"Conditions: {', '.join([data.get('name') for _, data in entities])}"
            
            elif action == "query_medications":
                pkg = self._load_patient_graph(params.get("patient_id"))
                entities = pkg.get_entities_by_type("MEDICATION")
                if not entities:
                    return "No medications found in patient record."
                return f"Medications: {', '.join([data.get('name') for _, data in entities])}"
            
            elif action == "query_symptoms":
                pkg = self._load_patient_graph(params.get("patient_id"))
                entities = pkg.get_entities_by_type("SYMPTOM")
                if not entities:
                    return "No symptoms documented."
                return f"Symptoms: {', '.join([data.get('name') for _, data in entities])}"
            
            elif action == "query_labs":
                pkg = self._load_patient_graph(params.get("patient_id"))
                entities = pkg.get_entities_by_type("LAB_VALUE")
                if not entities:
                    return "No lab values found."
                labs_info = [f"{data.get('name')}: {data.get('value', 'N/A')}" for _, data in entities]
                return f"Lab values: {', '.join(labs_info)}"
            
            elif action == "retrieve_memory":
                store = self.memory_manager.get_patient_store(params.get("patient_id"))
                results = store.retrieve_similar(params.get("query", ""), top_k=2)
                if not results:
                    return "No relevant past visits found."
                response = "Past visits:\n"
                for r in results:
                    response += f"- {r['metadata'].get('date', 'Unknown')}: {r['document'][:150]}...\n"
                return response
            
            elif action == "medical_kb":
                results = self.medical_kb.query(params.get("question", ""), top_k=2)
                if not results:
                    return "No medical knowledge found."
                response = "Medical knowledge:\n"
                for r in results:
                    response += f"- [{r['metadata'].get('type', 'info')}] {r['document'][:200]}...\n"
                return response
            
            elif action == "drug_interaction":
                result = self.medical_kb.check_drug_interaction(
                    params.get("drug1", ""), 
                    params.get("drug2", "")
                )
                if result:
                    return f"Drug interaction: {result['document']}"
                return f"No known interaction between {params.get('drug1')} and {params.get('drug2')}."
            
            elif action == "disease_info":
                results = self.medical_kb.get_disease_info(params.get("disease", ""))
                if results:
                    return f"Disease info: {results[0]['document']}"
                return f"No information found for {params.get('disease')}."
            
            elif action == "medication_info":
                results = self.medical_kb.get_medication_info(params.get("medication", ""))
                if results:
                    return f"Medication info: {results[0]['document']}"
                return f"No information found for {params.get('medication')}."
            
            elif action == "lab_reference":
                query = f"Lab reference for {params.get('lab_name', '')}"
                results = self.medical_kb.query(query, top_k=1)
                for r in results:
                    if r['metadata'].get('type') == 'lab_interpretation':
                        return f"Lab reference: {r['document']}"
                return f"No lab reference found for {params.get('lab_name')}."
            
            else:
                return f"Unknown action: {action}"
                
        except Exception as e:
            return f"Error executing {action}: {e}"
    
    def _parse_action(self, text: str) -> Optional[Tuple[str, Dict]]:
        """
        Parse action from agent's response.
        Looks for patterns like: Action: query_conditions["patient_id": "P001"]
        
        Returns:
            Tuple of (action_name, parameters) or None
        """
        # Pattern 1: Action: action_name[param1: value1, param2: value2]
        pattern1 = r'Action:\s*(\w+)\[(.*?)\]'
        match = re.search(pattern1, text, re.IGNORECASE)
        
        if match:
            action = match.group(1)
            params_str = match.group(2)
            
            # Parse parameters
            params = {}
            param_pattern = r'(\w+)\s*:\s*"([^"]+)"'
            for p_match in re.finditer(param_pattern, params_str):
                params[p_match.group(1)] = p_match.group(2)
            
            # Also look for simple key: value without quotes
            param_pattern2 = r'(\w+)\s*:\s*([^,\]]+)'
            for p_match in re.finditer(param_pattern2, params_str):
                key = p_match.group(1)
                value = p_match.group(2).strip().strip('"').strip("'")
                if key not in params:
                    params[key] = value
            
            return (action, params)
        
        # Pattern 2: Action: action_name(param1=value1, param2=value2)
        pattern2 = r'Action:\s*(\w+)\((.*?)\)'
        match = re.search(pattern2, text, re.IGNORECASE)
        
        if match:
            action = match.group(1)
            params_str = match.group(2)
            
            params = {}
            param_pattern = r'(\w+)\s*=\s*"([^"]+)"'
            for p_match in re.finditer(param_pattern, params_str):
                params[p_match.group(1)] = p_match.group(2)
            
            return (action, params)
        
        # Pattern 3: Simple Action: action_name with patient_id and query in text
        actions_list = ["query_conditions", "query_medications", "query_symptoms", "query_labs", 
                       "retrieve_memory", "medical_kb", "drug_interaction", "disease_info", 
                       "medication_info", "lab_reference"]
        
        for action in actions_list:
            if action in text.lower():
                params = {"patient_id": "P001"}  # Default
                
                # Try to extract patient_id
                pid_match = re.search(r'patient[_\s]id[:\s]+([A-Z0-9]+)', text, re.IGNORECASE)
                if pid_match:
                    params["patient_id"] = pid_match.group(1)
                
                # Try to extract query/question
                q_match = re.search(r'query[:\s]+["\']?([^"\'\n]+)["\']?', text, re.IGNORECASE)
                if q_match:
                    params["query"] = q_match.group(1).strip()
                
                # Try to extract drug names
                d1_match = re.search(r'drug1[:\s]+([A-Za-z]+)', text, re.IGNORECASE)
                if d1_match:
                    params["drug1"] = d1_match.group(1)
                d2_match = re.search(r'drug2[:\s]+([A-Za-z]+)', text, re.IGNORECASE)
                if d2_match:
                    params["drug2"] = d2_match.group(1)
                
                # Try to extract disease/medication
                disease_match = re.search(r'disease[:\s]+([A-Za-z\s]+?)(?:\n|$)', text, re.IGNORECASE)
                if disease_match:
                    params["disease"] = disease_match.group(1).strip()
                
                med_match = re.search(r'medication[:\s]+([A-Za-z\s]+?)(?:\n|$)', text, re.IGNORECASE)
                if med_match:
                    params["medication"] = med_match.group(1).strip()
                
                return (action, params)
        
        return None
    
    def _get_system_prompt(self, patient_id: str) -> str:
        """Get the system prompt for the agent."""
        return f"""You are GraphMed, a clinical reasoning AI assistant for Patient {patient_id}.

You have these actions/tools available:

1. query_conditions - Get patient's medical conditions
   Format: Action: query_conditions[patient_id: "{patient_id}"]

2. query_medications - Get patient's medications  
   Format: Action: query_medications[patient_id: "{patient_id}"]

3. query_symptoms - Get patient's symptoms
   Format: Action: query_symptoms[patient_id: "{patient_id}"]

4. query_labs - Get patient's lab values
   Format: Action: query_labs[patient_id: "{patient_id}"]

5. retrieve_memory - Search past visit summaries
   Format: Action: retrieve_memory[patient_id: "{patient_id}", query: "your search topic"]

6. medical_kb - Query medical knowledge base
   Format: Action: medical_kb[question: "your medical question"]

7. drug_interaction - Check drug interactions
   Format: Action: drug_interaction[drug1: "drug name", drug2: "drug name"]

8. disease_info - Get disease information
   Format: Action: disease_info[disease: "disease name"]

9. medication_info - Get medication information
   Format: Action: medication_info[medication: "medication name"]

10. lab_reference - Get lab reference ranges
    Format: Action: lab_reference[lab_name: "lab name"]

IMPORTANT RULES:
- Always use actions to get information - don't make up medical facts
- Use patient-specific actions first (query_conditions, query_medications)
- Format your actions exactly as shown above
- After each action, you'll receive an observation
- When you have enough information, provide a final answer starting with "Final Answer:"
- Include disclaimers in your final answer

Now, think step by step and use actions to answer the user's question."""
    
    def invoke(self, patient_id: str, query: str) -> Dict[str, Any]:
        """
        Invoke the agent with a query.
        
        Args:
            patient_id: Patient identifier
            query: Clinical question
        
        Returns:
            Agent response with reasoning trace
        """
        print(f"\n{'='*60}")
        print(f"💬 Patient: {patient_id}")
        print(f"❓ Query: {query}")
        print(f"{'='*60}\n")
        
        messages = [
            SystemMessage(content=self._get_system_prompt(patient_id)),
            HumanMessage(content=f"Question: {query}\n\nThink step by step. Use actions to gather information. When ready, provide Final Answer.")
        ]
        
        reasoning_steps = []
        actions_taken = []
        
        for step in range(self.max_steps):
            print(f"\n🔹 Step {step + 1}")
            
            # Get agent response
            response = self.llm.invoke(messages)
            response_text = response.content
            print(f"   💭 Thought: {response_text[:200]}...")
            
            # Check if this is a final answer
            if "Final Answer:" in response_text:
                final_answer = response_text.split("Final Answer:")[-1].strip()
                print(f"\n{'='*60}")
                print(f"🤖 GraphMed Response:")
                print(f"{'='*60}")
                print(final_answer)
                print(f"\n{'='*60}")
                print(f"📊 Reasoning Summary:")
                print(f"   Steps taken: {step + 1}")
                print(f"   Actions used: {actions_taken}")
                print(f"{'='*60}")
                
                return {
                    "answer": final_answer,
                    "reasoning_steps": reasoning_steps,
                    "actions_taken": actions_taken,
                    "patient_id": patient_id,
                    "query": query
                }
            
            # Parse and execute action
            action_info = self._parse_action(response_text)
            
            if action_info:
                action_name, params = action_info
                # Add patient_id if not present and needed
                if "patient_id" not in params and action_name in ["query_conditions", "query_medications", "query_symptoms", "query_labs", "retrieve_memory"]:
                    params["patient_id"] = patient_id
                
                observation = self._execute_action(action_name, params)
                actions_taken.append(action_name)
                reasoning_steps.append(f"Step {step + 1}: Used {action_name}")
                
                print(f"   👁️ Observation: {observation[:150]}...")
                
                # Add to conversation
                messages.append(AIMessage(content=response_text))
                messages.append(HumanMessage(content=f"Observation: {observation}\n\nContinue. If you have enough information, provide Final Answer."))
            else:
                # No action found, ask to use actions
                print(f"   ⚠️ No valid action found. Asking agent to use tools.")
                messages.append(AIMessage(content=response_text))
                messages.append(HumanMessage(content="Please use one of the available actions to gather information. Format: Action: action_name[parameters]"))
        
        # If we reach max steps, force a final answer
        print(f"\n⚠️ Reached maximum steps ({self.max_steps}). Forcing final answer...")
        
        messages.append(HumanMessage(content="Please provide your final answer now based on the information you have."))
        final_response = self.llm.invoke(messages)
        final_answer = final_response.content
        
        print(f"\n{'='*60}")
        print(f"🤖 GraphMed Response:")
        print(f"{'='*60}")
        print(final_answer)
        
        return {
            "answer": final_answer,
            "reasoning_steps": reasoning_steps,
            "actions_taken": actions_taken,
            "patient_id": patient_id,
            "query": query
        }


def test_agent():
    """Test the ReAct agent."""
    
    print("\n" + "="*60)
    print("🧪 TESTING REACT AGENT")
    print("="*60)
    
    agent = ReActAgent()
    
    # Simple test
    result = agent.invoke("P001", "What medications is this patient taking?")
    
    print(f"\n✅ Test complete!")


if __name__ == "__main__":
    test_agent()