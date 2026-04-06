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

        eval_mode = os.getenv("GRAPHMED_EVAL_MODE", "").strip().lower()
        default_temp = "0.0" if eval_mode in {"e1", "phase9", "eval", "factual"} else "0.1"
        try:
            self.llm_temperature = float(os.getenv("GRAPHMED_LLM_TEMPERATURE", default_temp))
        except Exception:
            self.llm_temperature = 0.0 if default_temp == "0.0" else 0.1
        
        # Initialize components
        self.graphs_dir = self.persist_dir / "graphs"
        self.memory_manager = GlobalMemoryManager(str(self.persist_dir / "chroma_db"))
        self.medical_kb = MedicalKnowledgeBase(str(self.persist_dir / "medical_kb"))
        
        # Cache for loaded graphs
        self._loaded_graphs = {}
        
        print("✅ Direct ReAct Agent initialized (no LangChain)")
        print(f"   Max reasoning steps: {max_steps}")
        print(f"   LLM provider: {self.llm_provider}")
        print(f"   LLM temperature: {self.llm_temperature}")


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
            "temperature": self.llm_temperature,
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
                "temperature": self.llm_temperature,
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
            "temperature": self.llm_temperature,
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

        def _format_dual_channel_response(payload: Dict[str, Any], label: str) -> str:
            current = payload.get("current_likely_state", [])
            historical = payload.get("critical_historical_facts", [])
            conflicts = payload.get("conflict_candidates", [])

            lines = [
                f"{label} (confidence-aware):",
                "  Confidence semantics: freshness/trust signal, not truth probability.",
                "  Current likely state:",
            ]

            if current:
                for item in current[:5]:
                    lines.append(
                        "   - "
                        f"{item.get('name')} [{item.get('type')}] "
                        f"score={item.get('hybrid_score', 0.0):.3f} "
                        f"conf={item.get('confidence', 0.0):.2f} "
                        f"recency={item.get('recency', 0.0):.2f}"
                    )
            else:
                lines.append("   - none")

            lines.append("  Critical historical facts:")
            if historical:
                for item in historical[:5]:
                    lines.append(
                        "   - "
                        f"{item.get('name')} [{item.get('type')}] "
                        f"conf={item.get('confidence', 0.0):.2f}"
                    )
            else:
                lines.append("   - none")

            lines.append("  Conflict candidates:")
            if conflicts:
                for item in conflicts[:5]:
                    lines.append(
                        "   - "
                        f"{item.get('name')} [{item.get('type')}] "
                        f"conf={item.get('confidence', 0.0):.2f} "
                        f"note={item.get('status', 'CONFLICTED')}"
                    )
            else:
                lines.append("   - none")

            return "\n".join(lines)
        
        try:
            if action == "query_conditions":
                pkg = self._load_patient_graph(params.get("patient_id", "P001"))
                payload = pkg.retrieve_dual_channel_facts(
                    query="active conditions and diagnosis status",
                    current_top_k=6,
                    historical_top_k=6,
                )
                has_any = payload.get("current_likely_state") or payload.get("critical_historical_facts")
                if not has_any:
                    return "No conditions found in patient record."
                return _format_dual_channel_response(payload, "Patient conditions")
            
            elif action == "query_medications":
                pkg = self._load_patient_graph(params.get("patient_id", "P001"))
                payload = pkg.retrieve_dual_channel_facts(
                    query="current medications and treatment history",
                    current_top_k=6,
                    historical_top_k=5,
                )
                filtered_current = []
                for item in payload.get("current_likely_state", []):
                    if item.get("type") != "MEDICATION":
                        continue
                    name = str(item.get("name", ""))
                    if name.replace('.', '').replace('mg', '').strip().isdigit():
                        continue
                    filtered_current.append(item)
                payload["current_likely_state"] = filtered_current

                if not payload.get("current_likely_state") and not payload.get("critical_historical_facts"):
                    return "No medications found in patient record."
                return _format_dual_channel_response(payload, "Patient medications")
            
            elif action == "query_symptoms":
                pkg = self._load_patient_graph(params.get("patient_id", "P001"))
                payload = pkg.retrieve_dual_channel_facts(
                    query="current symptoms and symptom history",
                    current_top_k=6,
                    historical_top_k=4,
                )
                payload["current_likely_state"] = [
                    x for x in payload.get("current_likely_state", []) if x.get("type") == "SYMPTOM"
                ]
                if not payload.get("current_likely_state") and not payload.get("critical_historical_facts"):
                    return "No symptoms documented."
                return _format_dual_channel_response(payload, "Patient symptoms")
            
            elif action == "query_labs":
                pkg = self._load_patient_graph(params.get("patient_id", "P001"))
                payload = pkg.retrieve_dual_channel_facts(
                    query="latest lab values and meaningful lab history",
                    current_top_k=7,
                    historical_top_k=4,
                )
                payload["current_likely_state"] = [
                    x for x in payload.get("current_likely_state", []) if x.get("type") == "LAB_VALUE"
                ]
                if not payload.get("current_likely_state"):
                    return "No lab values found."
                return _format_dual_channel_response(payload, "Patient labs")
            
            elif action == "retrieve_memory":
                query_text = params.get("query", "")
                patient_id = params.get("patient_id", "P001")
                try:
                    top_k = int(params.get("top_k", 2))
                except Exception:
                    top_k = 2
                top_k = max(1, min(8, top_k))
                store = self.memory_manager.get_patient_store(patient_id)
                results = store.retrieve_similar(query_text, top_k=top_k)
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
    Format: Action: retrieve_memory(patient_id="{patient_id}", query="search topic", top_k=4)

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

    def _detect_query_type(self, query: str) -> str:
        q = str(query or "").lower()
        if "using all visits" in q or "first and most recent" in q or "changed between" in q:
            return "multi_visit_context"
        if "safe" in q or "interaction" in q or "combination" in q:
            return "drug_safety"
        if "lab" in q or "trend" in q or "hba1c" in q or "egfr" in q:
            return "lab_trend"
        return "history"

    def _extract_drugs_from_query(self, query: str) -> Tuple[str, str]:
        q = str(query or "")
        pattern = r"combination of\s+([a-zA-Z0-9\- ]+?)\s+and\s+([a-zA-Z0-9\- ]+?)\s+(?:safe|for|\?|$)"
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if not m:
            return "", ""
        d1 = m.group(1).strip().lower()
        d2 = m.group(2).strip().lower()
        return d1, d2

    def _build_forced_actions(self, patient_id: str, query: str, query_type: str) -> List[Tuple[str, Dict[str, Any]]]:
        actions: List[Tuple[str, Dict[str, Any]]] = []

        if query_type == "multi_visit_context":
            actions.append(("retrieve_memory", {"patient_id": patient_id, "query": "first visit baseline summary", "top_k": 4}))
            actions.append(("retrieve_memory", {"patient_id": patient_id, "query": "most recent visit changes and progression", "top_k": 4}))
            actions.append(("query_conditions", {"patient_id": patient_id}))
            actions.append(("query_labs", {"patient_id": patient_id}))
            return actions

        if query_type == "drug_safety":
            d1, d2 = self._extract_drugs_from_query(query)
            actions.append(("query_medications", {"patient_id": patient_id}))
            actions.append(("retrieve_memory", {"patient_id": patient_id, "query": "recent medication changes and adverse effects", "top_k": 4}))
            if d1 and d2:
                actions.append(("drug_interaction", {"drug1": d1, "drug2": d2}))
            return actions

        if query_type == "lab_trend":
            actions.append(("query_labs", {"patient_id": patient_id}))
            actions.append(("retrieve_memory", {"patient_id": patient_id, "query": "lab trends across visits", "top_k": 4}))
            return actions

        # history/default
        actions.append(("query_conditions", {"patient_id": patient_id}))
        actions.append(("query_medications", {"patient_id": patient_id}))
        actions.append(("query_symptoms", {"patient_id": patient_id}))
        return actions

    def _get_eval_answer_contract(self, patient_id: str, query_type: str) -> str:
        return (
            "Evaluation answer contract (strict):\n"
            "- Output factual bullets only from tool observations.\n"
            "- No disclaimer, no advice, no extra narrative.\n"
            "- Use this exact format:\n"
            f"Final Answer:\n"
            f"patient_id: {patient_id}\n"
            f"question_type: {query_type}\n"
            "facts: <semicolon-separated factual findings>\n"
            "evidence_first_visit: <facts or none>\n"
            "evidence_latest_visit: <facts or none>\n"
            "confidence: <low|medium|high>"
        )

    def _strip_disclaimer_sentences(self, text: str) -> str:
        segments = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
        markers = [
            "medical disclaimer",
            "consult",
            "healthcare professional",
            "not medical advice",
            "informational purposes",
            "should not be considered",
        ]
        kept = [s for s in segments if not any(m in s.lower() for m in markers)]
        return " ".join(kept).strip()
    
    def invoke(
        self,
        patient_id: str,
        query: str,
        query_type: Optional[str] = None,
        evaluation_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Invoke the agent with a query.
        """
        print(f"\n{'='*60}")
        print(f"💬 Patient: {patient_id}")
        print(f"❓ Query: {query}")
        print(f"{'='*60}\n")
        
        resolved_query_type = query_type or self._detect_query_type(query)
        eval_mode = evaluation_mode or os.getenv("GRAPHMED_EVAL_MODE", "").strip().lower() in {
            "e1",
            "phase9",
            "eval",
            "factual",
        }

        user_prompt = f"Question: {query}\n\nThink step by step. Use actions to gather information. When ready, provide Final Answer."
        if eval_mode:
            user_prompt += "\n\n" + self._get_eval_answer_contract(patient_id, resolved_query_type)

        messages = [
            {"role": "system", "content": self._get_system_prompt(patient_id)},
            {"role": "user", "content": user_prompt}
        ]
        
        reasoning_steps = []
        actions_taken = []

        forced_actions = self._build_forced_actions(patient_id, query, resolved_query_type)
        for action_name, params in forced_actions:
            observation = self._execute_action(action_name, params)
            actions_taken.append(action_name)
            reasoning_steps.append(f"Pre-step: Used {action_name}")
            messages.append({"role": "assistant", "content": f"Action: {action_name}({params})"})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        
        for step in range(self.max_steps):
            print(f"\n🔹 Step {step + 1}")
            
            # Get agent response
            response_text = self._call_llm(messages)
            print(f"   💭 Agent: {response_text[:200]}...")
            
            # Check for final answer
            if "Final Answer:" in response_text:
                final_answer = response_text.split("Final Answer:")[-1].strip()
                if eval_mode:
                    final_answer = self._strip_disclaimer_sentences(final_answer)
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
                if eval_mode:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Use at least one tool action and then provide a strict contract-compliant Final Answer. "
                                "Do not include disclaimers."
                            ),
                        }
                    )
                else:
                    messages.append({"role": "user", "content": "Please use one of the available actions. Format: Action: action_name(parameter=\"value\")"})
        
        # Max steps reached
        messages.append({"role": "user", "content": "Please provide your final answer now based on available information."})
        final_answer = self._call_llm(messages)
        if eval_mode:
            final_answer = self._strip_disclaimer_sentences(final_answer)
        
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