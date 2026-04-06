"""
Phase 6: ReAct Agent with LangGraph for GraphMed
Stateful multi-turn clinical reasoning with tool grounding.
"""

from __future__ import annotations

import operator
import os
import re
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import PatientKnowledgeGraph
from src.memory import GlobalMemoryManager
from src.medical_kb import MedicalKnowledgeBase

load_dotenv()


class AgentState(TypedDict):
    """State used by LangGraph for a single turn execution."""

    messages: Annotated[List[BaseMessage], operator.add]
    patient_id: str
    current_query: str
    reasoning_steps: List[str]
    tool_calls_made: List[str]


class GraphMedAgent:
    """ReAct-style clinical agent using LangGraph for stateful multi-turn flows."""

    def __init__(self, persist_dir: str = "data"):
        self.persist_dir = Path(persist_dir)

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=1100,
        )

        self.graphs_dir = self.persist_dir / "graphs"
        self.memory_manager = GlobalMemoryManager(str(self.persist_dir / "chroma_db"))
        self.medical_kb = MedicalKnowledgeBase(str(self.persist_dir / "medical_kb"))

        self._loaded_graphs: Dict[str, PatientKnowledgeGraph] = {}

        # Stateful multi-turn store keyed by session id.
        self.sessions: Dict[str, Dict[str, Any]] = {}

        self.tools = self._create_tools()
        self.tool_executor = ToolExecutor(self.tools)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

        print("[OK] GraphMed Agent initialized with LangGraph")
        print(f"   Tools available: {len(self.tools)}")

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------
    def _load_patient_graph(self, patient_id: str) -> PatientKnowledgeGraph:
        if patient_id in self._loaded_graphs:
            return self._loaded_graphs[patient_id]

        graph_path = self.graphs_dir / f"{patient_id}_graph.json"
        if graph_path.exists():
            pkg = PatientKnowledgeGraph.load(str(graph_path))
            self._loaded_graphs[patient_id] = pkg
            return pkg

        pkg = PatientKnowledgeGraph(patient_id)
        self._loaded_graphs[patient_id] = pkg
        return pkg

    def _query_patient_graph_impl(self, patient_id: str, query_type: str) -> str:
        try:
            pkg = self._load_patient_graph(patient_id)

            query_map = {
                "conditions": "active conditions and diagnosis status",
                "medications": "current medications and treatment history",
                "symptoms": "current symptoms and symptom history",
                "labs": "latest lab values and meaningful lab history",
                "all": "overall patient current status and major history",
            }

            dual = pkg.retrieve_dual_channel_facts(
                query=query_map.get(query_type, query_map["all"]),
                current_top_k=6,
                historical_top_k=6,
            )

            def _render(channel_payload: Dict[str, Any], label: str, type_filter: Optional[str] = None) -> str:
                current = channel_payload.get("current_likely_state", [])
                historical = channel_payload.get("critical_historical_facts", [])
                conflicts = channel_payload.get("conflict_candidates", [])

                if type_filter:
                    current = [x for x in current if x.get("type") == type_filter]
                    historical = [x for x in historical if x.get("type") == type_filter or x in conflicts]

                lines = [
                    f"{label} (confidence-aware)",
                    "Confidence semantics: freshness/trust signal, not truth probability.",
                    "Current likely state:",
                ]

                if current:
                    for item in current[:5]:
                        lines.append(
                            "- "
                            f"{item.get('name')} [{item.get('type')}] "
                            f"score={item.get('hybrid_score', 0.0):.3f} "
                            f"conf={item.get('confidence', 0.0):.2f}"
                        )
                else:
                    lines.append("- none")

                lines.append("Critical historical facts:")
                if historical:
                    for item in historical[:5]:
                        lines.append(
                            "- "
                            f"{item.get('name')} [{item.get('type')}] conf={item.get('confidence', 0.0):.2f}"
                        )
                else:
                    lines.append("- none")

                lines.append("Conflict candidates:")
                if conflicts:
                    for item in conflicts[:5]:
                        lines.append(
                            "- "
                            f"{item.get('name')} [{item.get('type')}] conf={item.get('confidence', 0.0):.2f}"
                        )
                else:
                    lines.append("- none")
                return "\n".join(lines)

            if query_type == "conditions":
                return _render(dual, "Conditions", type_filter="CONDITION")

            if query_type == "medications":
                return _render(dual, "Medications", type_filter="MEDICATION")

            if query_type == "symptoms":
                return _render(dual, "Symptoms", type_filter="SYMPTOM")

            if query_type == "labs":
                return _render(dual, "Lab values", type_filter="LAB_VALUE")

            summary = pkg.summary()
            return (
                f"Graph summary: nodes={summary['total_nodes']}, edges={summary['total_edges']}, "
                f"conditions={summary['node_types'].get('CONDITION', 0)}, "
                f"medications={summary['node_types'].get('MEDICATION', 0)}\n"
                + _render(dual, "Top facts")
            )
        except Exception as e:
            return f"Error querying patient graph: {e}"

    def _retrieve_patient_memory_impl(self, patient_id: str, query: str) -> str:
        try:
            store = self.memory_manager.get_patient_store(patient_id)
            results = store.retrieve_similar(query, top_k=3)

            if not results:
                return "No relevant past visits found."

            lines = ["Relevant visit memories:"]
            for result in results:
                lines.append(
                    f"- Date: {result['metadata'].get('date', 'Unknown')} | "
                    f"Visit: {result['metadata'].get('visit_id', 'Unknown')} | "
                    f"Text: {result['document'][:240]}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error retrieving patient memory: {e}"

    def _query_medical_knowledge_impl(self, question: str) -> str:
        try:
            results = self.medical_kb.query(question, top_k=4)
            if not results:
                return "No relevant medical knowledge found."

            lines = ["Medical KB results:"]
            for r in results:
                meta = r.get("metadata", {})
                lines.append(
                    f"- Type: {meta.get('type', 'unknown')} | Source: {meta.get('source', 'unknown')} | "
                    f"Score: {r.get('score', 0.0):.3f} | Evidence: {r.get('document', '')[:260]}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error querying medical knowledge: {e}"

    def _check_drug_interaction_impl(self, drug1: str, drug2: str) -> str:
        try:
            result = self.medical_kb.check_drug_interaction(drug1, drug2)
            if not result:
                return f"No known interaction found between {drug1} and {drug2}."

            meta = result.get("metadata", {})
            return (
                f"Drug interaction found ({meta.get('source', 'unknown')}): "
                f"{result.get('document', '')[:500]}"
            )
        except Exception as e:
            return f"Error checking drug interaction: {e}"

    def _get_disease_info_impl(self, disease: str) -> str:
        try:
            results = self.medical_kb.get_disease_info(disease)
            if not results:
                return f"No disease information found for {disease}."
            best = results[0]
            return f"Disease info ({best.get('metadata', {}).get('source', 'unknown')}): {best.get('document', '')[:600]}"
        except Exception as e:
            return f"Error getting disease info: {e}"

    def _get_medication_info_impl(self, medication: str) -> str:
        try:
            results = self.medical_kb.get_medication_info(medication)
            if not results:
                return f"No medication information found for {medication}."
            best = results[0]
            return f"Medication info ({best.get('metadata', {}).get('source', 'unknown')}): {best.get('document', '')[:600]}"
        except Exception as e:
            return f"Error getting medication info: {e}"

    def _get_lab_reference_impl(self, lab_name: str) -> str:
        try:
            results = self.medical_kb.query(f"Lab reference for {lab_name}", top_k=4)
            for r in results:
                if r.get("metadata", {}).get("type") == "lab_interpretation":
                    return f"Lab reference: {r.get('document', '')[:500]}"
            return f"No lab reference found for {lab_name}."
        except Exception as e:
            return f"Error getting lab reference: {e}"

    def _get_clinical_guideline_impl(self, condition: str) -> str:
        try:
            results = self.medical_kb.query(f"Clinical guideline for {condition}", top_k=4)
            for r in results:
                if r.get("metadata", {}).get("type") == "clinical_guideline":
                    return (
                        f"Clinical guideline ({r.get('metadata', {}).get('source', 'unknown')}): "
                        f"{r.get('document', '')[:600]}"
                    )
            return f"No clinical guideline found for {condition}."
        except Exception as e:
            return f"Error getting clinical guideline: {e}"

    def _create_tools(self):
        @tool
        def query_patient_graph(patient_id: str, query_type: str) -> str:
            """Query structured patient graph. query_type: conditions|medications|symptoms|labs|all"""
            return self._query_patient_graph_impl(patient_id, query_type)

        @tool
        def retrieve_patient_memory(patient_id: str, query: str) -> str:
            """Retrieve semantically relevant past visit summaries for this patient."""
            return self._retrieve_patient_memory_impl(patient_id, query)

        @tool
        def query_medical_knowledge(question: str) -> str:
            """Query external non-patient medical knowledge (guidelines, interactions, disease info)."""
            return self._query_medical_knowledge_impl(question)

        @tool
        def check_drug_interaction(drug1: str, drug2: str) -> str:
            """Check if two drugs have known interactions."""
            return self._check_drug_interaction_impl(drug1, drug2)

        @tool
        def get_disease_info(disease: str) -> str:
            """Get disease information from external KB."""
            return self._get_disease_info_impl(disease)

        @tool
        def get_medication_info(medication: str) -> str:
            """Get medication information from external KB."""
            return self._get_medication_info_impl(medication)

        @tool
        def get_lab_reference(lab_name: str) -> str:
            """Get lab reference ranges and interpretations."""
            return self._get_lab_reference_impl(lab_name)

        @tool
        def get_clinical_guideline(condition: str) -> str:
            """Get clinical guideline recommendations for a condition."""
            return self._get_clinical_guideline_impl(condition)

        return [
            query_patient_graph,
            retrieve_patient_memory,
            query_medical_knowledge,
            check_drug_interaction,
            get_disease_info,
            get_medication_info,
            get_lab_reference,
            get_clinical_guideline,
        ]

    # ------------------------------------------------------------------
    # LangGraph workflow
    # ------------------------------------------------------------------
    def _to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value)
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    def _build_patient_profile(self, patient_id: str) -> Dict[str, Any]:
        """Create a deterministic patient profile from graph + latest memory."""
        profile = {
            "patient_id": patient_id,
            "conditions": [],
            "medications": [],
            "symptoms": [],
            "labs": {},
            "latest_visit": "Unknown",
            "latest_date": "Unknown",
            "risk_flags": [],
        }

        try:
            pkg = self._load_patient_graph(patient_id)
            profile["conditions"] = self._dedupe_preserve_order(
                [data.get("name", "") for _, data in pkg.get_entities_by_type("CONDITION") if data.get("name")]
            )
            profile["medications"] = self._dedupe_preserve_order(
                [data.get("name", "") for _, data in pkg.get_entities_by_type("MEDICATION") if data.get("name")]
            )
            profile["symptoms"] = self._dedupe_preserve_order(
                [data.get("name", "") for _, data in pkg.get_entities_by_type("SYMPTOM") if data.get("name")]
            )

            labs: Dict[str, Any] = {}
            for _, data in pkg.get_entities_by_type("LAB_VALUE"):
                name = data.get("name")
                value = data.get("value")
                if name:
                    labs[str(name)] = value
            profile["labs"] = labs
        except Exception:
            pass

        latest = self._get_memory_results(
            patient_id,
            "latest recent follow up current status most recent visit",
            top_k=1,
        )
        if latest:
            meta = latest[0].get("metadata", {})
            profile["latest_visit"] = meta.get("visit_id", "Unknown")
            profile["latest_date"] = meta.get("date", "Unknown")

        risk_flags: List[str] = []
        hba1c = self._to_float(profile["labs"].get("HbA1c"))
        ldl = self._to_float(profile["labs"].get("LDL"))
        sbp = self._to_float(profile["labs"].get("BP_systolic"))
        dbp = self._to_float(profile["labs"].get("BP_diastolic"))
        egfr = self._to_float(profile["labs"].get("eGFR"))

        if hba1c is not None and hba1c >= 8.0:
            risk_flags.append("Poor glycemic control (HbA1c high)")
        if ldl is not None and ldl >= 130:
            risk_flags.append("Elevated LDL cardiovascular risk")
        if (sbp is not None and sbp >= 140) or (dbp is not None and dbp >= 90):
            risk_flags.append("Uncontrolled blood pressure")
        if egfr is not None and egfr < 60:
            risk_flags.append("Reduced kidney function")

        profile["risk_flags"] = risk_flags
        return profile

    def _profile_text(self, profile: Dict[str, Any]) -> str:
        return (
            f"Patient: {profile.get('patient_id')}\n"
            f"Latest visit: {profile.get('latest_visit')} ({profile.get('latest_date')})\n"
            f"Conditions: {profile.get('conditions', [])}\n"
            f"Medications: {profile.get('medications', [])}\n"
            f"Symptoms: {profile.get('symptoms', [])}\n"
            f"Labs: {profile.get('labs', {})}\n"
            f"Risk flags: {profile.get('risk_flags', [])}"
        )

    def _classify_query_domain(self, query: str) -> str:
        q = query.lower()
        if any(t in q for t in ["diet", "food", "nutrition", "meal", "eat", "avoid"]):
            return "nutrition"
        if any(t in q for t in ["routine", "morning", "night", "schedule", "habits"]):
            return "routine"
        if any(t in q for t in ["exercise", "workout", "walk", "cardio", "strength", "activity"]):
            return "exercise"
        if any(t in q for t in ["issues", "problem", "status", "right now", "current"]):
            return "status"
        return "general"

    def _dynamic_answer_contract(self, domain: str, profile: Dict[str, Any]) -> str:
        """Domain-specific answer contract to avoid generic responses."""
        base = (
            "Response must be patient-specific, not generic. "
            "Explicitly reference at least 2 concrete patient factors from profile/evidence "
            "(conditions, medications, labs, symptoms, trends)."
        )

        if domain == "nutrition":
            return (
                base + " For nutrition queries, include: foods to prioritize, foods to limit/avoid, "
                "and why each recommendation maps to this patient's conditions/labs."
            )
        if domain == "routine":
            return (
                base + " For routine queries, include a concrete morning/day routine with checkpoints "
                "that tie to this patient's risks and conditions."
            )
        if domain == "exercise":
            return (
                base + " For exercise queries, include safe intensity guidance and contraindication-aware "
                "advice using this patient's conditions and current status."
            )
        if domain == "status":
            return (
                base + " For status queries, summarize current active issues, worsening/improving signals, "
                "and immediate priorities."
            )
        return base

    def _is_personalized_answer(self, answer: str, profile: Dict[str, Any]) -> bool:
        low = answer.lower()
        terms: List[str] = []
        terms.extend([str(x).lower() for x in profile.get("conditions", [])])
        terms.extend([str(x).lower() for x in profile.get("medications", [])])
        terms.extend([str(x).lower() for x in profile.get("symptoms", [])])
        terms.extend([str(x).lower() for x in profile.get("labs", {}).keys()])

        hits = 0
        for t in self._dedupe_preserve_order([x for x in terms if x]):
            if t in low:
                hits += 1
        return hits >= 2
    def _dedupe_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _extract_list_field(self, document: str, field_name: str) -> List[str]:
        pattern = rf"-\s*{re.escape(field_name)}:\s*(.*)"
        match = re.search(pattern, document)
        if not match:
            return []
        raw = match.group(1).strip()
        if not raw:
            return []
        return [x.strip() for x in raw.split(",") if x.strip()]

    def _extract_labs(self, document: str) -> Dict[str, float]:
        labs: Dict[str, float] = {}
        pattern = r"-\s*Labs:\s*(.*)"
        match = re.search(pattern, document)
        if not match:
            return labs

        parts = [x.strip() for x in match.group(1).split(",") if x.strip()]
        for part in parts:
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            num_match = re.search(r"-?\d+(?:\.\d+)?", value)
            if not num_match:
                continue
            try:
                labs[key.strip()] = float(num_match.group(0))
            except Exception:
                continue
        return labs

    def _get_memory_results(self, patient_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        try:
            store = self.memory_manager.get_patient_store(patient_id)
            return store.retrieve_similar(query, top_k=top_k)
        except Exception:
            return []

    def _format_memory_evidence(self, results: List[Dict[str, Any]], chunk_prefix: str) -> Dict[str, Any]:
        lines: List[str] = []
        citations: List[str] = []

        for idx, result in enumerate(results, 1):
            meta = result.get("metadata", {})
            visit_id = meta.get("visit_id", "Unknown")
            date = meta.get("date", "Unknown")
            preview = result.get("document", "")[:220]
            chunk_id = f"{chunk_prefix}{idx}"
            lines.append(f"- {chunk_id} | Visit {visit_id} ({date}) | {preview}")
            citations.append(f"Visit {visit_id} ({date}), Memory chunk {chunk_id}")

        if not lines:
            return {
                "text": "No relevant past visits found.",
                "citations": [],
            }

        return {
            "text": "\n".join(lines),
            "citations": citations,
        }

    def _get_clinical_delta_summary(self, patient_id: str) -> Dict[str, Any]:
        """Deterministically summarize change between latest and prior visit memories."""
        latest_results = self._get_memory_results(
            patient_id,
            "latest recent follow up current status most recent visit",
            top_k=6,
        )

        if not latest_results:
            return {
                "summary": "Clinical delta unavailable: no visit memories found.",
                "graph_update_recommendation": "No - insufficient visit data for delta analysis.",
                "citations": [],
            }

        def sort_key(item: Dict[str, Any]) -> str:
            return str(item.get("metadata", {}).get("date", ""))

        ordered = sorted(latest_results, key=sort_key)
        latest = ordered[-1]
        previous = ordered[-2] if len(ordered) >= 2 else None

        latest_meta = latest.get("metadata", {})
        latest_doc = latest.get("document", "")
        latest_visit = latest_meta.get("visit_id", "Unknown")
        latest_date = latest_meta.get("date", "Unknown")

        citations = [f"Visit {latest_visit} ({latest_date}), Memory chunk D1"]

        if previous is None:
            summary = (
                f"Only one visit available ({latest_visit}, {latest_date}); "
                "cannot compute trend across visits yet."
            )
            return {
                "summary": summary,
                "graph_update_recommendation": "No - wait for at least one prior visit for temporal delta.",
                "citations": citations,
            }

        prev_meta = previous.get("metadata", {})
        prev_doc = previous.get("document", "")
        prev_visit = prev_meta.get("visit_id", "Unknown")
        prev_date = prev_meta.get("date", "Unknown")
        citations.append(f"Visit {prev_visit} ({prev_date}), Memory chunk D2")

        latest_conditions = set(self._extract_list_field(latest_doc, "Diagnoses/Conditions"))
        prev_conditions = set(self._extract_list_field(prev_doc, "Diagnoses/Conditions"))
        latest_meds = set(self._extract_list_field(latest_doc, "Medications"))
        prev_meds = set(self._extract_list_field(prev_doc, "Medications"))

        added_conditions = sorted(list(latest_conditions - prev_conditions))
        removed_conditions = sorted(list(prev_conditions - latest_conditions))
        added_meds = sorted(list(latest_meds - prev_meds))
        removed_meds = sorted(list(prev_meds - latest_meds))

        latest_labs = self._extract_labs(latest_doc)
        prev_labs = self._extract_labs(prev_doc)
        shared_labs = sorted(set(latest_labs.keys()) & set(prev_labs.keys()))

        deltas: List[str] = []
        worsening_flags: List[str] = []
        for lab in shared_labs:
            delta = latest_labs[lab] - prev_labs[lab]
            deltas.append(f"{lab}: {prev_labs[lab]:.2f} -> {latest_labs[lab]:.2f} (delta {delta:+.2f})")

            if lab == "HbA1c" and delta > 0.30:
                worsening_flags.append("HbA1c rising")
            elif lab == "LDL" and delta > 10:
                worsening_flags.append("LDL rising")
            elif lab == "BP_systolic" and delta > 5:
                worsening_flags.append("BP systolic rising")
            elif lab == "BP_diastolic" and delta > 3:
                worsening_flags.append("BP diastolic rising")
            elif lab == "eGFR" and delta < -3:
                worsening_flags.append("eGFR declining")

        summary_parts = [
            f"Compared latest visit {latest_visit} ({latest_date}) against prior visit {prev_visit} ({prev_date}).",
            f"Condition changes: +{added_conditions if added_conditions else 'none'} / -{removed_conditions if removed_conditions else 'none'}.",
            f"Medication changes: +{added_meds if added_meds else 'none'} / -{removed_meds if removed_meds else 'none'}.",
            f"Lab deltas: {deltas if deltas else 'no comparable numeric lab pairs found.'}",
        ]

        graph_update_recommendation = "No - no strong delta signal requiring graph update."
        if added_conditions or removed_conditions or added_meds or removed_meds or worsening_flags:
            reasons = []
            if added_conditions or removed_conditions:
                reasons.append("condition set changed")
            if added_meds or removed_meds:
                reasons.append("medication set changed")
            if worsening_flags:
                reasons.append("worsening trend: " + ", ".join(self._dedupe_preserve_order(worsening_flags)))
            graph_update_recommendation = "Yes - " + "; ".join(reasons)

        return {
            "summary": " ".join(summary_parts),
            "graph_update_recommendation": graph_update_recommendation,
            "citations": citations,
        }

    def _has_required_sections_and_citations(self, answer: str, require_kb: bool) -> bool:
        low = answer.lower()
        has_sections = (
            "clinical summary" in low
            and "evidence citations" in low
            and "graph update recommendation" in low
            and "uncertainty note" in low
            and "safety disclaimer" in low
        )
        has_visit = bool(re.search(r"visit\s+\w+", answer, re.IGNORECASE))
        has_memory_chunk = bool(re.search(r"memory\s+chunk", answer, re.IGNORECASE))
        has_kb = bool(re.search(r"kb\s+source", answer, re.IGNORECASE))

        if require_kb:
            return has_sections and has_visit and has_memory_chunk and has_kb
        return has_sections and has_visit and has_memory_chunk

    def _enforce_citation_format(
        self,
        answer: str,
        mandatory: Dict[str, Any],
        turn_messages: List[BaseMessage],
    ) -> str:
        def _normalize_headers(text: str) -> str:
            replacements = {
                r"(?im)^\s*\*{0,2}\s*clinical summary\s*:?\s*\*{0,2}\s*$": "Clinical Summary:",
                r"(?im)^\s*\*{0,2}\s*evidence citations\s*:?\s*\*{0,2}\s*$": "Evidence citations:",
                r"(?im)^\s*\*{0,2}\s*graph update recommendation\s*:?\s*\*{0,2}\s*$": "Graph update recommendation:",
                r"(?im)^\s*\*{0,2}\s*uncertainty note\s*:?\s*\*{0,2}\s*$": "Uncertainty Note:",
                r"(?im)^\s*\*{0,2}\s*safety disclaimer\s*:?\s*\*{0,2}\s*$": "Safety Disclaimer:",
            }
            out = text
            for pattern, repl in replacements.items():
                out = re.sub(pattern, repl, out)
            return out

        require_kb = mandatory.get("require_kb", False)
        profile = mandatory.get("patient_profile", {})
        answer = _normalize_headers(answer)
        if self._has_required_sections_and_citations(answer, require_kb=require_kb) and self._is_personalized_answer(answer, profile):
            return answer

        citation_fix_prompt = (
            "Your previous answer is rejected because citation/section format is incomplete. "
            "Rewrite the answer with EXACT required sections:\n"
            "1) Clinical Summary\n"
            "2) Evidence citations\n"
            "3) Graph update recommendation\n"
            "4) Uncertainty Note\n"
            "5) Safety Disclaimer\n\n"
            "Use ONLY this evidence and these citations:\n"
            f"{mandatory['evidence_block']}\n\n"
            "Citations to include verbatim (as applicable):\n"
            f"{mandatory['citation_block']}\n\n"
            "Do not omit Evidence citations section.\n"
            "Also ensure recommendations are patient-specific and explicitly tied to patient profile factors."
        )

        revised = self.llm.invoke(turn_messages + [HumanMessage(content=citation_fix_prompt)])
        revised_text = revised.content if hasattr(revised, "content") else ""

        revised_text = _normalize_headers(revised_text)
        if self._has_required_sections_and_citations(revised_text, require_kb=require_kb) and self._is_personalized_answer(revised_text, profile):
            return revised_text

        # Deterministic fallback if model still violates citation format.
        return (
            "Clinical Summary:\n"
            "Grounded summary generated from mandatory graph, memory, and optional KB evidence.\n\n"
            "Evidence citations:\n"
            f"{mandatory['citation_block']}\n\n"
            "Graph update recommendation:\n"
            f"{mandatory['graph_update_recommendation']}\n\n"
            "Uncertainty Note:\n"
            "Some recommendations may require clinician validation against full chart details.\n\n"
            "Safety Disclaimer:\n"
            "This response is not a substitute for a licensed clinician's evaluation and treatment plan."
        )

    def _extract_block(self, evidence_block: str, block_name: str, max_chars: int = 280) -> str:
        pattern = rf"\[{re.escape(block_name)}\]\n(.*?)(?:\n\n\[|$)"
        match = re.search(pattern, evidence_block, re.DOTALL)
        if not match:
            return ""
        text = " ".join(match.group(1).strip().split())
        return text[:max_chars]

    def _build_compact_trace(
        self,
        query: str,
        mandatory: Dict[str, Any],
        turn_tools: List[str],
        used_fallback: bool,
    ) -> Dict[str, Any]:
        evidence_block = mandatory.get("evidence_block", "")
        citation_lines = [
            line.strip().lstrip("- ").strip()
            for line in mandatory.get("citation_block", "").splitlines()
            if line.strip()
        ]

        return {
            "query": query,
            "used_fallback": used_fallback,
            "turn_tools": turn_tools,
            "kb_required": mandatory.get("require_kb", False),
            "graph_update_recommendation": mandatory.get("graph_update_recommendation", "Unknown"),
            "steps": [
                "1) Mandatory pre-retrieval completed.",
                "2) Evidence injected into prompt.",
                "3) Optional ReAct tool loop executed as needed.",
                "4) Citation format enforced before final answer.",
            ],
            "evidence_preview": {
                "graph": self._extract_block(evidence_block, "GRAPH"),
                "memory_query": self._extract_block(evidence_block, "MEMORY_QUERY"),
                "memory_change": self._extract_block(evidence_block, "MEMORY_CHANGE_CHECK"),
                "clinical_delta": self._extract_block(evidence_block, "CLINICAL_DELTA"),
                "medical_kb": self._extract_block(evidence_block, "MEDICAL_KB"),
            },
            "citation_preview": citation_lines[:5],
        }

    def _needs_medical_kb(self, query: str) -> bool:
        q = query.lower()
        kb_terms = [
            "interaction", "guideline", "contraindication", "dose", "dosage",
            "treatment", "recommend", "diet", "safety", "side effect", "risk",
            "management", "what should", "what can", "plan", "food", "nutrition",
            "meal", "sodium", "sugar", "carb", "cholesterol", "fat", "fiber",
            "long life", "longevity", "avoid", "routine", "morning routine",
            "daily routine", "lifestyle", "exercise", "workout", "activity",
            "sleep", "habits", "prevention", "prevent", "healthy living",
            "issues", "problem", "advice", "suggest", "suggested", "recommendation",
        ]
        pubmed_terms = [
            "evidence", "literature", "study", "studies", "research", "pubmed",
            "trial", "meta analysis", "systematic review",
        ]
        return any(t in q for t in kb_terms) or any(t in q for t in pubmed_terms)

    def _build_mandatory_evidence(self, patient_id: str, query: str) -> Dict[str, Any]:
        """
        Mandatory memory-aware pre-retrieval step before LLM reasoning:
        1) patient graph summary
        2) patient visit memory retrieval for current query
        3) memory retrieval for change/progression detection
        4) medical KB retrieval for relevant clinical queries
        """
        graph_obs = self._query_patient_graph_impl(patient_id, "all")
        patient_profile = self._build_patient_profile(patient_id)
        query_domain = self._classify_query_domain(query)

        memory_results_query = self._get_memory_results(patient_id, query, top_k=3)
        memory_query_formatted = self._format_memory_evidence(memory_results_query, "M")

        memory_results_change = self._get_memory_results(
            patient_id,
            "changes since last visits progression worsening improving new diagnosis medication adjustment lab trend",
            top_k=3,
        )
        memory_change_formatted = self._format_memory_evidence(memory_results_change, "C")

        delta = self._get_clinical_delta_summary(patient_id)

        kb_obs = "Skipped (query not classified as external-clinical-knowledge dependent)."
        kb_citations: List[str] = []
        tools = ["query_patient_graph", "retrieve_patient_memory", "retrieve_patient_memory(change_check)"]
        require_kb = self._needs_medical_kb(query)
        if require_kb:
            kb_results = self.medical_kb.query(query, top_k=3)
            if kb_results:
                kb_lines = []
                for idx, r in enumerate(kb_results, 1):
                    meta = r.get("metadata", {})
                    src = meta.get("source", "unknown")
                    doc_type = meta.get("type", "unknown")
                    kb_chunk = f"K{idx}"
                    kb_lines.append(
                        f"- {kb_chunk} | KB source {src} | type {doc_type} | "
                        f"score {r.get('score', 0.0):.3f} | {r.get('document', '')[:220]}"
                    )
                    kb_citations.append(f"KB source {src}, KB chunk {kb_chunk}, type {doc_type}")
                kb_obs = "\n".join(kb_lines)
            else:
                kb_obs = "No relevant medical KB entries found."
            tools.append("query_medical_knowledge")

        evidence_block = (
            "MANDATORY PRE-RETRIEVAL EVIDENCE\n"
            f"[GRAPH]\n{graph_obs}\n\n"
            f"[PATIENT_PROFILE]\n{self._profile_text(patient_profile)}\n\n"
            f"[MEMORY_QUERY]\n{memory_query_formatted['text']}\n\n"
            f"[MEMORY_CHANGE_CHECK]\n{memory_change_formatted['text']}\n\n"
            f"[CLINICAL_DELTA]\n{delta['summary']}\n\n"
            f"[MEDICAL_KB]\n{kb_obs}"
        )

        citation_lines = []
        citation_lines.extend(memory_query_formatted["citations"])
        citation_lines.extend(memory_change_formatted["citations"])
        citation_lines.extend(delta["citations"])
        citation_lines.extend(kb_citations)
        citation_lines = self._dedupe_preserve_order(citation_lines)
        if not citation_lines:
            citation_lines = ["Visit Unknown (Unknown), Memory chunk M0"]

        citation_block = "\n".join(f"- {line}" for line in citation_lines)

        reasoning = [
            "Mandatory pre-retrieval executed.",
            "Checked patient graph for structured baseline facts.",
            "Checked vector memory for visit narrative context.",
            "Checked memory for progression/change signals across visits.",
            "Computed deterministic clinical delta between recent visits.",
        ]
        if require_kb:
            reasoning.append("Checked external medical KB for grounding evidence.")
        reasoning.append(f"Graph update recommendation: {delta['graph_update_recommendation']}")
        reasoning.append(f"Query domain classified as: {query_domain}")

        return {
            "evidence_block": evidence_block,
            "tools": tools,
            "reasoning": reasoning,
            "citation_block": citation_block,
            "graph_update_recommendation": delta["graph_update_recommendation"],
            "require_kb": require_kb,
            "patient_profile": patient_profile,
            "query_domain": query_domain,
            "dynamic_contract": self._dynamic_answer_contract(query_domain, patient_profile),
        }

    def _system_prompt(self, patient_id: str) -> str:
        return f"""You are GraphMed, a ReAct-style clinical assistant for patient {patient_id}.

Operating principles:
1) Think step-by-step internally, then act by calling tools.
2) For patient-specific questions, use patient tools first:
   - query_patient_graph
   - retrieve_patient_memory
3) For medical facts/recommendations, ground with external tools:
   - query_medical_knowledge
   - check_drug_interaction
   - get_disease_info / get_medication_info / get_clinical_guideline
4) Do not invent facts. If evidence is weak, say so.
5) Final answer must include:
   - concise clinical summary
   - key evidence used and tool/source references
   - uncertainty note when applicable
   - safety disclaimer: not a substitute for a licensed clinician
6) Use a maximum of 4 tool calls per question. After gathering enough evidence, stop calling tools and provide the final answer.
7) Never produce a final answer without using the provided mandatory pre-retrieval evidence.
8) Final answer format is mandatory with these exact section headers:
    Clinical Summary:
    Evidence citations:
    Graph update recommendation:
    Uncertainty Note:
    Safety Disclaimer:
"""

    def _build_graph(self):
        def agent_node(state: AgentState):
            messages = state["messages"]
            has_system = any(isinstance(m, SystemMessage) for m in messages)
            if not has_system:
                messages = [SystemMessage(content=self._system_prompt(state["patient_id"]))] + messages

            response = self.llm_with_tools.invoke(messages)

            reasoning = state.get("reasoning_steps", [])
            reasoning.append(
                f"Agent step: {'tool_call' if getattr(response, 'tool_calls', None) else 'final_response'}"
            )

            return {
                "messages": [response],
                "reasoning_steps": reasoning,
            }

        def tools_node(state: AgentState):
            last_message = state["messages"][-1]
            tool_messages: List[ToolMessage] = []
            tool_names: List[str] = []
            reasoning = state.get("reasoning_steps", [])

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tc in last_message.tool_calls:
                    tool_name = tc.get("name", "unknown_tool")
                    tool_names.append(tool_name)
                    try:
                        result = self.tool_executor.invoke(tc)
                        content = f"[tool:{tool_name}] {result}"
                    except Exception as e:
                        content = f"[tool:{tool_name}] Tool error: {e}"

                    tool_messages.append(
                        ToolMessage(content=content, tool_call_id=tc["id"])
                    )

            if tool_names:
                reasoning.append(f"Executed tools: {tool_names}")

            return {
                "messages": tool_messages,
                "tool_calls_made": state.get("tool_calls_made", []) + tool_names,
                "reasoning_steps": reasoning,
            }

        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tools_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    # ------------------------------------------------------------------
    # Stateful multi-turn session methods
    # ------------------------------------------------------------------
    def start_session(self, patient_id: str, session_id: Optional[str] = None) -> str:
        sid = session_id or f"session_{patient_id}_{uuid.uuid4().hex[:8]}"
        self.sessions[sid] = {
            "patient_id": patient_id,
            "messages": [SystemMessage(content=self._system_prompt(patient_id))],
            "reasoning_steps": [],
            "tool_calls_made": [],
            "turn_count": 0,
        }
        return sid

    def ask(self, session_id: str, query: str, max_iterations: int = 6) -> Dict[str, Any]:
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session_id: {session_id}")

        session = self.sessions[session_id]
        patient_id = session["patient_id"]

        mandatory = self._build_mandatory_evidence(patient_id, query)

        turn_messages = session["messages"] + [
            HumanMessage(
                content=(
                    "Use the following mandatory pre-retrieval evidence first.\n\n"
                    f"{mandatory['evidence_block']}\n\n"
                    "Required citation inventory:\n"
                    f"{mandatory['citation_block']}\n\n"
                    f"Graph update recommendation seed: {mandatory['graph_update_recommendation']}"
                )
            ),
            HumanMessage(
                content=(
                    f"Patient ID: {patient_id}\nQuestion: {query}\n\n"
                    "ReAct mode: Think -> Act (optional additional tool calls) -> Observe -> Answer. "
                    "If evidence already sufficient, provide final grounded answer now. "
                    "Do not skip the mandatory final answer section format.\n\n"
                    f"Dynamic patient-specific answer contract: {mandatory['dynamic_contract']}"
                )
            )
        ]

        initial_state: AgentState = {
            "messages": turn_messages,
            "patient_id": patient_id,
            "current_query": query,
            "reasoning_steps": session.get("reasoning_steps", []) + mandatory["reasoning"],
            "tool_calls_made": session.get("tool_calls_made", []) + mandatory["tools"],
        }

        try:
            result = self.graph.invoke(initial_state, {"recursion_limit": max_iterations * 4})

            final_message = result["messages"][-1] if result.get("messages") else AIMessage(content="No response generated.")
            answer = final_message.content if hasattr(final_message, "content") else "No response generated."
            answer = self._enforce_citation_format(answer, mandatory, turn_messages)

            graph_tools = result.get("tool_calls_made", [])
            turn_tools = self._dedupe_preserve_order(mandatory["tools"] + graph_tools)
            combined_tools = self._dedupe_preserve_order(session.get("tool_calls_made", []) + turn_tools)
            combined_reasoning = mandatory["reasoning"] + result.get("reasoning_steps", [])
            compact_trace = self._build_compact_trace(query, mandatory, turn_tools, used_fallback=False)

            session["messages"] = result["messages"]
            session["reasoning_steps"] = combined_reasoning
            session["tool_calls_made"] = combined_tools
            session["turn_count"] = session.get("turn_count", 0) + 1

            return {
                "session_id": session_id,
                "patient_id": patient_id,
                "query": query,
                "answer": answer,
                "reasoning_steps": combined_reasoning,
                "tools_called": combined_tools,
                "turn_tools_called": turn_tools,
                "compact_trace": compact_trace,
                "turn_count": session["turn_count"],
                "full_trace": result,
            }
        except Exception as e:
            # Fallback synthesis when the tool loop does not naturally terminate.
            fallback_messages = turn_messages + [
                HumanMessage(
                    content=(
                        "Stop tool use now. Provide a final answer based on available context and observations. "
                        "Include uncertainty and a safety disclaimer."
                    )
                )
            ]
            fallback = self.llm.invoke(fallback_messages)
            answer = fallback.content if hasattr(fallback, "content") else f"Agent error: {e}"
            answer = self._enforce_citation_format(answer, mandatory, turn_messages)

            fallback_tools = self._dedupe_preserve_order(mandatory["tools"])
            fallback_reasoning = mandatory["reasoning"] + [f"Fallback synthesis used: {e}"]
            compact_trace = self._build_compact_trace(query, mandatory, fallback_tools, used_fallback=True)

            session["messages"] = turn_messages + [AIMessage(content=answer)]
            session["reasoning_steps"] = fallback_reasoning
            session["tool_calls_made"] = fallback_tools
            session["turn_count"] = session.get("turn_count", 0) + 1

            return {
                "session_id": session_id,
                "patient_id": patient_id,
                "query": query,
                "answer": answer,
                "reasoning_steps": fallback_reasoning,
                "tools_called": fallback_tools,
                "turn_tools_called": fallback_tools,
                "compact_trace": compact_trace,
                "turn_count": session["turn_count"],
                "full_trace": None,
            }

    def invoke(self, patient_id: str, query: str, max_iterations: int = 6) -> Dict[str, Any]:
        """Backward-compatible one-shot entrypoint."""
        sid = self.start_session(patient_id)
        return self.ask(sid, query, max_iterations=max_iterations)


def test_agent() -> None:
    agent = GraphMedAgent()
    sid = agent.start_session("P001")
    r1 = agent.ask(sid, "What conditions does this patient have?")
    print("\nAnswer 1:\n", r1["answer"])
    r2 = agent.ask(sid, "Any medication interaction risk I should watch for?")
    print("\nAnswer 2:\n", r2["answer"])


if __name__ == "__main__":
    test_agent()
