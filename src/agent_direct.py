"""
Phase 6: Direct ReAct Agent for GraphMed
Uses Groq API directly without LangChain dependencies
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import PatientKnowledgeGraph
from src.memory import GlobalMemoryManager
from src.medical_kb import MedicalKnowledgeBase

load_dotenv()


class DirectReActAgent:
    """
    Direct ReAct Agent using Groq API (no LangChain).
    Implements Thought -> Action -> Observation loop.
    """
    
    def __init__(self, persist_dir: str = "data", max_steps: int = 5, llm_provider: Optional[str] = None):
        """
        Initialize the ReAct agent.
        
        Args:
            persist_dir: Base directory for persisted data
            max_steps: Maximum reasoning steps per query
        """
        self.persist_dir = Path(persist_dir)
        self.max_steps = max_steps
        # LLM provider selection:
        # 1) explicit constructor arg
        # 2) GRAPHMED_LLM_PROVIDER
        # 3) BASELINE_LLM_PROVIDER fallback
        # 4) default to groq
        self.llm_provider = (
            llm_provider
            or os.getenv("GRAPHMED_LLM_PROVIDER")
            or os.getenv("BASELINE_LLM_PROVIDER")
            or "groq"
        ).strip().lower()

        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_api_url = "https://api.groq.com/openai/v1/chat/completions"

        self.openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        self.openrouter_api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.openrouter_model = os.getenv("GRAPHMED_OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_model = os.getenv("GRAPHMED_GOOGLE_MODEL", "gemini-2.5-flash")
        
        # Initialize components
        self.graphs_dir = self.persist_dir / "graphs"
        self.memory_manager = GlobalMemoryManager(str(self.persist_dir / "chroma_db"))
        self.medical_kb = MedicalKnowledgeBase(str(self.persist_dir / "medical_kb"))
        
        # Cache for loaded graphs
        self._loaded_graphs = {}
        
        print("✅ Direct ReAct Agent initialized (no LangChain)")
        print(f"   Max reasoning steps: {max_steps}")
        print(f"   LLM provider: {self.llm_provider}")


    def _evolve_graph(self, patient_id: str, new_visit: Dict) -> str:
        """Evolve patient graph with new visit data."""
        try:
            from src.graph_evolution import evolve_patient_graph
            result = evolve_patient_graph(patient_id, new_visit, str(self.graphs_dir))
            
            summary = f"""
    Graph Evolution Results:
    - Added/Updated: {len([r for r in result['evolution_results'] if r['action'] in ['ADDED', 'UPDATED']])} entities
    - Conflicts detected: {len([r for r in result['evolution_results'] if r['action'] == 'CONFLICT_DETECTED'])}
    - Total nodes: {result['summary']['total_nodes']}
    - Total conflicts in graph: {result['summary']['conflicts']}
    """
            return summary
        except Exception as e:
            return f"Error evolving graph: {e}"
    
    def _call_groq(self, messages: List[Dict]) -> str:
        """Call Groq API directly."""
        if not self.groq_api_key:
            return "Error calling Groq API: GROQ_API_KEY is not configured"

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.groq_api_url, headers=headers, json=data, timeout=45)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"   ❌ API Error: {e}")
            return f"Error calling Groq API: {e}"

    def _call_google(self, messages: List[Dict]) -> str:
        """Call Google Generative Language API (Gemini) using chat transcript as prompt."""
        if not self.google_api_key:
            return "Error calling Google API: GOOGLE_API_KEY is not configured"

        transcript = []
        for msg in messages:
            role = str(msg.get("role", "user")).upper()
            content = str(msg.get("content", ""))
            transcript.append(f"{role}: {content}")
        prompt = "\n\n".join(transcript)

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1000,
            },
        }

        try:
            for model in [self.google_model, self._resolve_google_model()]:
                if not model:
                    continue
                url = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model}:generateContent?key={self.google_api_key}"
                )
                response = requests.post(url, json=payload, timeout=45)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return "Error calling Google API: empty response"
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(str(p.get("text", "")) for p in parts).strip()
                if not text:
                    return "Error calling Google API: empty text"
                self.google_model = model
                return text

            return (
                "Error calling Google API: no supported model found for generateContent "
                "(all candidate models returned 404)"
            )
        except Exception as e:
            print(f"   ❌ API Error: {e}")
            return f"Error calling Google API: {e}"

    def _call_openrouter(self, messages: List[Dict]) -> str:
        """Call OpenRouter using OpenAI-compatible chat-completions API."""
        if not self.openrouter_api_key:
            return "Error calling OpenRouter API: OPEN_ROUTER_API_KEY is not configured"

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.openrouter_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1000,
        }
        try:
            response = requests.post(self.openrouter_api_url, headers=headers, json=data, timeout=45)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"   ❌ API Error: {e}")
            return f"Error calling OpenRouter API: {e}"

    def _resolve_google_model(self) -> Optional[str]:
        """Resolve a supported Gemini model name for generateContent."""
        if not self.google_api_key:
            return None
        try:
            list_url = "https://generativelanguage.googleapis.com/v1beta/models"
            resp = requests.get(list_url, params={"key": self.google_api_key}, timeout=30)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            supported = []
            for m in models:
                methods = m.get("supportedGenerationMethods", []) or []
                if "generateContent" in methods:
                    name = str(m.get("name", ""))
                    if name.startswith("models/"):
                        name = name.split("/", 1)[1]
                    if name:
                        supported.append(name)

            priority = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
            for p in priority:
                if p in supported:
                    return p
            return supported[0] if supported else None
        except Exception:
            return None

    def _call_llm(self, messages: List[Dict]) -> str:
        """Dispatch LLM call to selected provider with optional fallbacks."""
        provider = self.llm_provider
        if provider == "openrouter":
            openrouter_result = self._call_openrouter(messages)
            if not openrouter_result.lower().startswith("error calling openrouter api"):
                return openrouter_result
            fallback_to_groq = os.getenv("GRAPHMED_OPENROUTER_FALLBACK_TO_GROQ", "1").strip() == "1"
            if fallback_to_groq:
                return self._call_groq(messages)
            return openrouter_result

        if provider == "google":
            google_result = self._call_google(messages)
            if not google_result.lower().startswith("error calling google api"):
                return google_result
            # Strict mode by default: stay on Google if selected.
            fallback_enabled = os.getenv("GRAPHMED_GOOGLE_FALLBACK_TO_GROQ", "0").strip() == "1"
            if fallback_enabled:
                return self._call_groq(messages)
            return google_result

        # default groq
        groq_result = self._call_groq(messages)
        # Optional fallback to Google when Groq rate-limits (opt-in only).
        fallback_enabled = os.getenv("GRAPHMED_GROQ_FALLBACK_TO_GOOGLE", "0").strip() == "1"
        if fallback_enabled and (
            "429" in groq_result.lower() or "too many requests" in groq_result.lower()
        ) and self.google_api_key:
            return self._call_google(messages)
        return groq_result
    
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
        """
        print(f"   🔧 Executing: {action}")
        
        try:
            if action == "query_conditions":
                pkg = self._load_patient_graph(params.get("patient_id", "P001"))
                entities = pkg.get_entities_by_type("CONDITION")
                if not entities:
                    return "No conditions found in patient record."
                return f"Patient's conditions: {', '.join([data.get('name') for _, data in entities])}"
            
            elif action == "query_medications":
                pkg = self._load_patient_graph(params.get("patient_id", "P001"))
                entities = pkg.get_entities_by_type("MEDICATION")
                # Filter out dosage-only entries
                med_names = []
                for _, data in entities:
                    name = data.get('name', '')
                    # Skip if it's just a number (dosage)
                    if not name.replace('.', '').replace('mg', '').strip().isdigit():
                        med_names.append(name)
                if not med_names:
                    return "No medications found in patient record."
                return f"Patient's medications: {', '.join(med_names)}"
            
            elif action == "query_symptoms":
                pkg = self._load_patient_graph(params.get("patient_id", "P001"))
                entities = pkg.get_entities_by_type("SYMPTOM")
                if not entities:
                    return "No symptoms documented."
                return f"Patient's symptoms: {', '.join([data.get('name') for _, data in entities])}"
            
            elif action == "query_labs":
                pkg = self._load_patient_graph(params.get("patient_id", "P001"))
                entities = pkg.get_entities_by_type("LAB_VALUE")
                if not entities:
                    return "No lab values found."
                labs_info = [f"{data.get('name')}: {data.get('value', 'N/A')}" for _, data in entities]
                return f"Patient's lab values: {', '.join(labs_info)}"
            
            elif action == "retrieve_memory":
                query_text = params.get("query", "")
                patient_id = params.get("patient_id", "P001")
                store = self.memory_manager.get_patient_store(patient_id)
                results = store.retrieve_similar(query_text, top_k=2)
                if not results:
                    return "No relevant past visits found."
                response = "Relevant past visits:\n"
                for r in results:
                    date = r['metadata'].get('date', 'Unknown')
                    preview = r['document'][:150]
                    response += f"  - {date}: {preview}...\n"
                return response
            
            elif action == "medical_kb":
                question = params.get("question", "")
                results = self.medical_kb.query(question, top_k=2)
                if not results:
                    return "No medical knowledge found."
                response = "Medical knowledge:\n"
                for r in results:
                    doc_type = r['metadata'].get('type', 'info')
                    response += f"  - [{doc_type}] {r['document'][:200]}...\n"
                return response
            
            elif action == "drug_interaction":
                drug1 = params.get("drug1", "")
                drug2 = params.get("drug2", "")
                result = self.medical_kb.check_drug_interaction(drug1, drug2)
                if result:
                    return f"DRUG INTERACTION: {result['document']}"
                return f"No known interaction found between {drug1} and {drug2}."
            
            elif action == "disease_info":
                disease = params.get("disease", "")
                results = self.medical_kb.get_disease_info(disease)
                if results:
                    return f"DISEASE INFO: {results[0]['document']}"
                return f"No information found for {disease}."
            
            elif action == "medication_info":
                medication = params.get("medication", "")
                results = self.medical_kb.get_medication_info(medication)
                if results:
                    return f"MEDICATION INFO: {results[0]['document']}"
                return f"No information found for {medication}."
            
            elif action == "lab_reference":
                lab_name = params.get("lab_name", "")
                query = f"Lab reference for {lab_name}"
                results = self.medical_kb.query(query, top_k=1)
                for r in results:
                    if r['metadata'].get('type') == 'lab_interpretation':
                        return f"LAB REFERENCE: {r['document']}"
                return f"No lab reference found for {lab_name}."
            
            else:
                return f"Unknown action: {action}"
                
        except Exception as e:
            return f"Error executing {action}: {e}"
    
    def _parse_action(self, text: str) -> Optional[Tuple[str, Dict]]:
        """
        Parse action from agent's response.
        """
        # Pattern: Action: action_name(parameter1="value1", parameter2="value2")
        pattern = r'Action:\s*(\w+)\s*\(\s*([^)]*)\s*\)'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            action = match.group(1)
            params_str = match.group(2)
            
            params = {}
            # Parse key="value" pairs
            param_pattern = r'(\w+)\s*=\s*"([^"]*)"'
            for p_match in re.finditer(param_pattern, params_str):
                params[p_match.group(1)] = p_match.group(2)
            
            # Also handle key=value without quotes
            param_pattern2 = r'(\w+)\s*=\s*([^,\s)]+)'
            for p_match in re.finditer(param_pattern2, params_str):
                key = p_match.group(1)
                value = p_match.group(2)
                if key not in params:
                    params[key] = value
            
            if params:
                return (action, params)
        
        # Pattern: Action: action_name[parameter: "value"]
        pattern2 = r'Action:\s*(\w+)\[\s*([^\]]+)\s*\]'
        match = re.search(pattern2, text, re.IGNORECASE)
        
        if match:
            action = match.group(1)
            params_str = match.group(2)
            
            params = {}
            param_pattern = r'(\w+)\s*:\s*"([^"]*)"'
            for p_match in re.finditer(param_pattern, params_str):
                params[p_match.group(1)] = p_match.group(2)
            
            if params:
                return (action, params)
        
        return None
    
    def _get_system_prompt(self, patient_id: str) -> str:
        """Get the system prompt for the agent."""
        return f"""You are GraphMed, a clinical reasoning AI assistant for Patient {patient_id}.

You have these actions/tools available. You MUST use them to gather information before answering:

1. query_conditions - Get patient's medical conditions
   Format: Action: query_conditions(patient_id="{patient_id}")

2. query_medications - Get patient's medications
   Format: Action: query_medications(patient_id="{patient_id}")

3. query_symptoms - Get patient's symptoms
   Format: Action: query_symptoms(patient_id="{patient_id}")

4. query_labs - Get patient's lab values
   Format: Action: query_labs(patient_id="{patient_id}")

5. retrieve_memory - Search past visit summaries
   Format: Action: retrieve_memory(patient_id="{patient_id}", query="search topic")

6. medical_kb - Query medical knowledge base
   Format: Action: medical_kb(question="your medical question")

7. drug_interaction - Check drug interactions
   Format: Action: drug_interaction(drug1="drug name", drug2="drug name")

8. disease_info - Get disease information
   Format: Action: disease_info(disease="disease name")

9. medication_info - Get medication information
   Format: Action: medication_info(medication="medication name")

10. lab_reference - Get lab reference ranges
    Format: Action: lab_reference(lab_name="lab name")

RULES:
- ALWAYS use actions to get information - NEVER make up medical facts
- First, use patient-specific actions (query_conditions, query_medications)
- After each action, you'll receive an observation
- When you have enough information, provide your answer starting with "Final Answer:"
- Include medical disclaimers in your final answer

Now, answer the user's question by thinking step by step and using actions."""
    
    def invoke(self, patient_id: str, query: str) -> Dict[str, Any]:
        """
        Invoke the agent with a query.
        """
        print(f"\n{'='*60}")
        print(f"💬 Patient: {patient_id}")
        print(f"❓ Query: {query}")
        print(f"{'='*60}\n")
        
        messages = [
            {"role": "system", "content": self._get_system_prompt(patient_id)},
            {"role": "user", "content": f"Question: {query}\n\nThink step by step. Use actions to gather information. When ready, provide Final Answer."}
        ]
        
        reasoning_steps = []
        actions_taken = []
        
        for step in range(self.max_steps):
            print(f"\n🔹 Step {step + 1}")
            
            # Get agent response
            response_text = self._call_llm(messages)
            print(f"   💭 Agent: {response_text[:200]}...")
            
            # Check for final answer
            if "Final Answer:" in response_text:
                final_answer = response_text.split("Final Answer:")[-1].strip()
                print(f"\n{'='*60}")
                print(f"🤖 GraphMed Response:")
                print(f"{'='*60}")
                print(final_answer)
                print(f"\n{'='*60}")
                print(f"📊 Reasoning Summary:")
                print(f"   Steps: {step + 1}")
                print(f"   Actions: {actions_taken}")
                print(f"{'='*60}")
                
                return {
                    "answer": final_answer,
                    "reasoning_steps": reasoning_steps,
                    "actions_taken": actions_taken
                }
            
            # Parse and execute action
            action_info = self._parse_action(response_text)
            
            if action_info:
                action_name, params = action_info
                if not params:
                    params = {}
                if "patient_id" not in params:
                    params["patient_id"] = patient_id
                
                observation = self._execute_action(action_name, params)
                actions_taken.append(action_name)
                reasoning_steps.append(f"Step {step + 1}: Used {action_name}")
                
                print(f"   👁️ Observation: {observation[:150]}...")
                
                # Add to conversation
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"Observation: {observation}\n\nContinue. If you have enough information, provide Final Answer."})
            else:
                # No action found
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": "Please use one of the available actions. Format: Action: action_name(parameter=\"value\")"})
        
        # Max steps reached
        messages.append({"role": "user", "content": "Please provide your final answer now based on available information."})
        final_answer = self._call_llm(messages)
        
        print(f"\n{'='*60}")
        print(f"🤖 GraphMed Response (max steps):")
        print(f"{'='*60}")
        print(final_answer)
        
        return {
            "answer": final_answer,
            "reasoning_steps": reasoning_steps,
            "actions_taken": actions_taken
        }


def test_agent():
    """Test the agent."""
    
    print("\n" + "="*60)
    print("🧪 TESTING DIRECT REACT AGENT")
    print("="*60)
    
    agent = DirectReActAgent()
    result = agent.invoke("P001", "What medications is this patient taking?")
    
    print(f"\n✅ Test complete!")


if __name__ == "__main__":
    test_agent()