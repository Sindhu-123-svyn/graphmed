"""
Phase 11: Streamlit Demo App for GraphMed

Interactive demo with:
- Patient selection
- Knowledge graph visualization
- Chat interface with visible reasoning/tool traces
- Graph evolution across patient visits
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from src.agent import GraphMedAgent
from src.graph import PatientKnowledgeGraph
from src.graph_evolution import GraphEvolution, evolve_patient_graph
from src.memory import GlobalMemoryManager


st.set_page_config(
	page_title="GraphMed Demo",
	page_icon="+",
	layout="wide",
	initial_sidebar_state="expanded",
)


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
GRAPHS_DIR = DATA_DIR / "graphs"
PATIENTS_PROCESSED_DIR = DATA_DIR / "patients_processed"
PATIENTS_RAW_DIR = DATA_DIR / "patients"

NODE_COLORS = {
	"PATIENT": "#111827",
	"CONDITION": "#B91C1C",
	"MEDICATION": "#0369A1",
	"LAB_VALUE": "#047857",
	"SYMPTOM": "#7C3AED",
	"PROCEDURE": "#B45309",
}


def _safe_read_json(file_path: Path) -> Dict[str, Any]:
	with file_path.open("r", encoding="utf-8") as f:
		return json.load(f)


@st.cache_data(show_spinner=False)
def list_patient_ids() -> List[str]:
	ids = set()
	if GRAPHS_DIR.exists():
		for file in GRAPHS_DIR.glob("*_graph.json"):
			ids.add(file.stem.replace("_graph", ""))
	if PATIENTS_PROCESSED_DIR.exists():
		for file in PATIENTS_PROCESSED_DIR.glob("*.json"):
			ids.add(file.stem)
	return sorted(ids)


@st.cache_data(show_spinner=False)
def load_graph_dict(patient_id: str) -> Dict[str, Any] | None:
	path = GRAPHS_DIR / f"{patient_id}_graph.json"
	if not path.exists():
		return None
	return _safe_read_json(path)


@st.cache_data(show_spinner=False)
def load_processed_patient(patient_id: str) -> Dict[str, Any] | None:
	path = PATIENTS_PROCESSED_DIR / f"{patient_id}.json"
	if not path.exists():
		return None
	return _safe_read_json(path)


def _safe_write_json(file_path: Path, payload: Dict[str, Any]) -> None:
	file_path.parent.mkdir(parents=True, exist_ok=True)
	with file_path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def _build_pyvis_html_from_graph_dict(graph_data: Dict[str, Any], height: str = "700px") -> str:
	net = Network(height=height, width="100%", directed=True)
	net.barnes_hut()

	for node in graph_data.get("nodes", []):
		node_id = node.get("id")
		node_type = node.get("type", "UNKNOWN")
		label = node.get("name", node_id)
		confidence = node.get("confidence", 1.0)
		title = (
			f"Type: {node_type}<br>"
			f"Name: {label}<br>"
			f"Last confirmed: {node.get('last_confirmed', 'NA')}<br>"
			f"Confidence: {float(confidence):.2f}"
		)
		net.add_node(
			node_id,
			label=label,
			color=NODE_COLORS.get(node_type, "#6B7280"),
			title=title,
			size=30 if node_type == "PATIENT" else 16,
		)

	for edge in graph_data.get("edges", []):
		src = edge.get("source")
		dst = edge.get("target")
		relation = edge.get("relation", "related_to")
		if src and dst:
			net.add_edge(src, dst, label=relation, title=f"Relation: {relation}")

	with NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp:
		net.save_graph(tmp.name)
		html = Path(tmp.name).read_text(encoding="utf-8")
	return html


def _graph_summary_metrics(graph_data: Dict[str, Any]) -> Dict[str, Any]:
	node_types: Dict[str, int] = {}
	conflicts = 0
	for node in graph_data.get("nodes", []):
		node_type = node.get("type", "UNKNOWN")
		node_types[node_type] = node_types.get(node_type, 0) + 1
		if str(node.get("status", "")).upper() == "CONFLICTED":
			conflicts += 1

	return {
		"total_nodes": len(graph_data.get("nodes", [])),
		"total_edges": len(graph_data.get("edges", [])),
		"node_types": node_types,
		"conflicts": conflicts,
	}


def _normalize_visit_for_evolution(visit: Dict[str, Any]) -> Dict[str, Any]:
	extracted = visit.get("extracted", {})
	if not isinstance(extracted, dict):
		extracted = {}
	return {
		"visit_id": visit.get("visit_id", "Unknown"),
		"date": visit.get("date", "Unknown"),
		"extracted": {
			"conditions": extracted.get("conditions", []) or visit.get("diagnoses", []),
			"medications": extracted.get("medications", []) or visit.get("medications", []),
			"symptoms": extracted.get("symptoms", []) or visit.get("symptoms", []),
			"lab_values": extracted.get("lab_values", {}) or visit.get("labs", {}),
			"procedures": extracted.get("procedures", []),
			"relationships": extracted.get("relationships", []),
		},
	}


def _split_csv_field(value: str) -> List[str]:
	if not value.strip():
		return []
	parts = [p.strip() for p in value.split(",")]
	cleaned: List[str] = []
	seen = set()
	for part in parts:
		if not part:
			continue
		key = part.lower()
		if key in seen:
			continue
		seen.add(key)
		cleaned.append(part)
	return cleaned


def _coerce_lab_value(value: str) -> Any:
	text = value.strip()
	if not text:
		return text
	try:
		if "." in text:
			return float(text)
		return int(text)
	except Exception:
		return text


def _parse_labs_input(raw_labs: str) -> Dict[str, Any]:
	"""Parse lines in key:value format into lab dictionary."""
	labs: Dict[str, Any] = {}
	for line in raw_labs.splitlines():
		line = line.strip()
		if not line or ":" not in line:
			continue
		key, val = line.split(":", 1)
		k = key.strip()
		if not k:
			continue
		labs[k] = _coerce_lab_value(val)
	return labs


def _find_assistant_suggestions(answer: str) -> Dict[str, List[str]]:
	"""Extract diagnosis and medication suggestions from assistant response text."""
	text = answer or ""

	def parse_list(fragment: str) -> List[str]:
		fragment = re.sub(r"\s+", " ", fragment)
		fragment = fragment.replace(" and ", ", ")
		items = [x.strip(" .;:-") for x in fragment.split(",") if x.strip(" .;:-")]
		cleaned: List[str] = []
		seen = set()
		for item in items:
			low = item.lower()
			if low in seen:
				continue
			seen.add(low)
			cleaned.append(item)
		return cleaned

	diagnoses: List[str] = []
	medications: List[str] = []

	dx_patterns = [
		r"conditions?\s*,?\s*including\s+([^\.\n]+)",
		r"diagnoses?\s*include\s+([^\.\n]+)",
		r"diagnoses?/conditions?\s*:\s*([^\n]+)",
	]
	med_patterns = [
		r"medication list includes\s+([^\.\n]+)",
		r"medications?\s*include\s+([^\.\n]+)",
		r"currently on\s+([^\.\n]+)",
	]

	for pattern in dx_patterns:
		match = re.search(pattern, text, flags=re.IGNORECASE)
		if match:
			diagnoses = parse_list(match.group(1))
			break

	for pattern in med_patterns:
		match = re.search(pattern, text, flags=re.IGNORECASE)
		if match:
			medications = parse_list(match.group(1))
			break

	return {"diagnoses": diagnoses, "medications": medications}


def _next_visit_id(processed_data: Dict[str, Any] | None) -> str:
	visits = (processed_data or {}).get("visits", [])
	max_id = 0
	for visit in visits:
		visit_id = str(visit.get("visit_id", ""))
		match = re.match(r"^V(\d+)$", visit_id)
		if match:
			max_id = max(max_id, int(match.group(1)))
	return f"V{max_id + 1}"


def _latest_visit(processed_data: Dict[str, Any] | None) -> Dict[str, Any] | None:
	visits = list((processed_data or {}).get("visits", []) or [])
	if not visits:
		return None
	visits.sort(key=lambda v: str(v.get("date", "")))
	return visits[-1]


def _merge_unique_preserve_order(*lists: List[str]) -> List[str]:
	merged: List[str] = []
	seen = set()
	for values in lists:
		for item in values:
			value = str(item).strip()
			if not value:
				continue
			key = value.lower()
			if key in seen:
				continue
			seen.add(key)
			merged.append(value)
	return merged


def _latest_assistant_answer(patient_id: str) -> str:
	messages = st.session_state.chat_messages.get(patient_id, [])
	for message in reversed(messages):
		if message.get("role") == "assistant" and message.get("content"):
			return str(message["content"])
	return ""


def _append_visit_to_patient_json(patient_id: str, visit: Dict[str, Any]) -> None:
	for base in [PATIENTS_PROCESSED_DIR, PATIENTS_RAW_DIR]:
		path = base / f"{patient_id}.json"
		if not path.exists():
			continue
		payload = _safe_read_json(path)
		visits = list(payload.get("visits", []) or [])
		visits.append(visit)
		visits.sort(key=lambda x: str(x.get("date", "")))
		payload["visits"] = visits
		payload["patient_id"] = patient_id
		_safe_write_json(path, payload)


def _update_vector_memory(patient_id: str, visit: Dict[str, Any]) -> None:
	manager = GlobalMemoryManager(str(DATA_DIR / "chroma_db"))
	store = manager.get_patient_store(patient_id)
	store.store_visit(visit, visit_id=visit.get("visit_id"))


def _refresh_after_updates(patient_id: str) -> None:
	list_patient_ids.clear()
	load_graph_dict.clear()
	load_processed_patient.clear()
	build_evolution_history.clear()

	try:
		agent = get_agent()
		agent._loaded_graphs.pop(patient_id, None)
	except Exception:
		pass


@st.cache_data(show_spinner=False)
def build_evolution_history(patient_id: str, processed: Dict[str, Any]) -> List[Dict[str, Any]]:
	visits = sorted(processed.get("visits", []), key=lambda v: v.get("date", ""))
	if not visits:
		return []

	graph = PatientKnowledgeGraph(patient_id)
	evolution = GraphEvolution(graph)
	history: List[Dict[str, Any]] = []

	for index, raw_visit in enumerate(visits, start=1):
		visit = _normalize_visit_for_evolution(raw_visit)
		results = evolution.evolve_with_visit(visit)

		add_count = sum(1 for r in results if str(r.get("operation", "")).upper() == "ADD")
		update_count = sum(1 for r in results if str(r.get("operation", "")).upper() == "UPDATE")
		conflict_count = sum(1 for r in results if str(r.get("operation", "")).upper() == "CONFLICT")

		history.append(
			{
				"visit_number": index,
				"visit_id": visit.get("visit_id", f"V{index}"),
				"date": visit.get("date", "Unknown"),
				"ops": {
					"ADD": add_count,
					"UPDATE": update_count,
					"CONFLICT": conflict_count,
				},
				"summary": evolution.get_evolution_summary(),
				"results": deepcopy(results),
				"graph_dict": deepcopy(graph.to_dict()),
			}
		)

	return history


@st.cache_resource(show_spinner=False)
def get_agent() -> GraphMedAgent:
	return GraphMedAgent(persist_dir=str(DATA_DIR))


def _init_session_state() -> None:
	if "chat_messages" not in st.session_state:
		st.session_state.chat_messages = {}
	if "agent_sessions" not in st.session_state:
		st.session_state.agent_sessions = {}


def _get_or_create_agent_session(agent: GraphMedAgent, patient_id: str) -> str:
	if patient_id not in st.session_state.agent_sessions:
		st.session_state.agent_sessions[patient_id] = agent.start_session(patient_id)
	return st.session_state.agent_sessions[patient_id]


def render_sidebar(patient_ids: List[str]) -> str:
	st.sidebar.title("GraphMed Demo")
	st.sidebar.caption("Interactive phase for graph + reasoning + evolution")
	selected = st.sidebar.selectbox("Select patient", patient_ids, index=0 if patient_ids else None)
	st.sidebar.markdown("---")
	st.sidebar.markdown("### Capabilities")
	st.sidebar.markdown("- Patient selection")
	st.sidebar.markdown("- Knowledge graph visualization")
	st.sidebar.markdown("- Chat with reasoning traces")
	st.sidebar.markdown("- Visit-by-visit graph evolution")
	return selected


def render_graph_tab(patient_id: str, graph_data: Dict[str, Any] | None) -> None:
	st.subheader(f"Knowledge Graph: {patient_id}")

	if not graph_data:
		st.warning("No graph file found for this patient yet. Run graph build phases first.")
		return

	metrics = _graph_summary_metrics(graph_data)
	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Nodes", metrics["total_nodes"])
	c2.metric("Edges", metrics["total_edges"])
	c3.metric("Conflicts", metrics["conflicts"])
	c4.metric("Types", len(metrics["node_types"]))

	st.markdown("#### Node Type Distribution")
	dist_df = pd.DataFrame(
		[{"type": k, "count": v} for k, v in sorted(metrics["node_types"].items(), key=lambda x: x[0])]
	)
	if not dist_df.empty:
		st.dataframe(dist_df, use_container_width=True, hide_index=True)

	st.markdown("#### Interactive Graph")
	graph_html = _build_pyvis_html_from_graph_dict(graph_data)
	components.html(graph_html, height=760, scrolling=True)


def render_chat_tab(patient_id: str) -> None:
	st.subheader(f"Clinical Chat: {patient_id}")
	st.caption("Each assistant response includes reasoning and tool traces from the GraphMed agent.")

	try:
		agent = get_agent()
	except Exception as exc:
		st.error(f"Agent initialization failed: {exc}")
		st.info("Check API keys and model configuration before using chat.")
		return

	session_id = _get_or_create_agent_session(agent, patient_id)
	if patient_id not in st.session_state.chat_messages:
		st.session_state.chat_messages[patient_id] = []

	messages = st.session_state.chat_messages[patient_id]

	for msg in messages:
		with st.chat_message(msg["role"]):
			st.markdown(msg["content"])
			trace = msg.get("trace")
			if msg["role"] == "assistant" and trace:
				with st.expander("Reasoning Trace", expanded=False):
					if trace.get("tools"):
						st.write("Tools called this turn:", ", ".join(trace["tools"]))
					if trace.get("steps"):
						st.write("Reasoning steps:")
						for step in trace["steps"]:
							st.write(f"- {step}")
					compact = trace.get("compact", {})
					if compact:
						st.json(compact)

	prompt = st.chat_input("Ask a patient-specific question...")
	if not prompt:
		return

	st.session_state.chat_messages[patient_id].append({"role": "user", "content": prompt})
	with st.chat_message("user"):
		st.markdown(prompt)

	with st.chat_message("assistant"):
		with st.spinner("Reasoning with graph + memory + medical KB..."):
			try:
				response = agent.ask(session_id, prompt)
				answer = response.get("answer", "No answer returned.")
				st.markdown(answer)

				trace_payload = {
					"tools": response.get("turn_tools_called", []),
					"steps": response.get("reasoning_steps", [])[-10:],
					"compact": response.get("compact_trace", {}),
				}
				with st.expander("Reasoning Trace", expanded=False):
					if trace_payload["tools"]:
						st.write("Tools called this turn:", ", ".join(trace_payload["tools"]))
					if trace_payload["steps"]:
						st.write("Reasoning steps:")
						for step in trace_payload["steps"]:
							st.write(f"- {step}")
					if trace_payload["compact"]:
						st.json(trace_payload["compact"])

				st.session_state.chat_messages[patient_id].append(
					{
						"role": "assistant",
						"content": answer,
						"trace": trace_payload,
					}
				)
			except Exception as exc:
				err = f"Chat request failed: {exc}"
				st.error(err)
				st.session_state.chat_messages[patient_id].append(
					{
						"role": "assistant",
						"content": err,
						"trace": None,
					}
				)


def render_evolution_tab(patient_id: str, processed: Dict[str, Any] | None) -> None:
	st.subheader(f"Graph Evolution Across Visits: {patient_id}")

	if not processed:
		st.warning("No processed patient file found for this patient.")
		return

	history = build_evolution_history(patient_id, processed)
	if not history:
		st.info("No visits available to simulate evolution.")
		return

	ops_df = pd.DataFrame(
		[
			{
				"visit": h["visit_id"],
				"date": h["date"],
				"ADD": h["ops"]["ADD"],
				"UPDATE": h["ops"]["UPDATE"],
				"CONFLICT": h["ops"]["CONFLICT"],
				"total_nodes": h["summary"].get("total_nodes", 0),
				"total_edges": h["summary"].get("total_edges", 0),
			}
			for h in history
		]
	)

	st.markdown("#### Visit Timeline")
	st.dataframe(ops_df, use_container_width=True, hide_index=True)

	st.markdown("#### Evolution Trends")
	trend_df = ops_df.set_index("visit")[["ADD", "UPDATE", "CONFLICT"]]
	st.line_chart(trend_df)

	visit_labels = [f"{h['visit_id']} ({h['date']})" for h in history]
	selected_idx = st.select_slider(
		"Inspect graph state after visit",
		options=list(range(len(visit_labels))),
		format_func=lambda i: visit_labels[i],
		value=len(visit_labels) - 1,
	)

	selected_state = history[selected_idx]
	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Visit", selected_state["visit_id"])
	c2.metric("ADD", selected_state["ops"]["ADD"])
	c3.metric("UPDATE", selected_state["ops"]["UPDATE"])
	c4.metric("CONFLICT", selected_state["ops"]["CONFLICT"])

	st.markdown("#### Graph Snapshot")
	graph_html = _build_pyvis_html_from_graph_dict(selected_state["graph_dict"], height="640px")
	components.html(graph_html, height=680, scrolling=True)

	with st.expander("Detailed evolution events for selected visit", expanded=False):
		st.json(selected_state["results"])


def render_new_visit_tab(patient_id: str, processed_data: Dict[str, Any] | None) -> None:
	st.subheader(f"New Visit Intake: {patient_id}")
	st.caption(
		"Creates a new visit, appends patient JSON, updates graph evolution, updates vector memory, and refreshes all tabs in-session."
	)

	if not processed_data:
		st.warning("No processed patient file found. Create patient data first.")
		return

	next_visit = _next_visit_id(processed_data)
	last_visit = _latest_visit(processed_data)
	latest_answer = _latest_assistant_answer(patient_id)
	suggestions = _find_assistant_suggestions(latest_answer)

	current_diagnoses = _split_csv_field(
		", ".join((last_visit or {}).get("diagnoses", []) or [])
	)
	current_medications = _split_csv_field(
		", ".join((last_visit or {}).get("medications", []) or [])
	)
	current_symptoms = _split_csv_field(
		", ".join((last_visit or {}).get("symptoms", []) or [])
	)

	generated_diagnoses = _split_csv_field(", ".join(suggestions.get("diagnoses", [])))
	generated_medications = _split_csv_field(", ".join(suggestions.get("medications", [])))

	final_diagnoses = _merge_unique_preserve_order(current_diagnoses, generated_diagnoses)
	final_medications = _merge_unique_preserve_order(current_medications, generated_medications)

	if latest_answer:
		with st.expander("Auto-extracted from latest assistant answer", expanded=False):
			st.write("Suggested diagnoses:", suggestions.get("diagnoses", []))
			st.write("Suggested medications:", suggestions.get("medications", []))
			st.write("Current diagnoses (latest visit):", current_diagnoses)
			st.write("Current medications (latest visit):", current_medications)
			st.write("Final diagnoses for new visit (current + generated):", final_diagnoses)
			st.write("Final medications for new visit (current + generated):", final_medications)
			st.write("Current symptoms (latest visit):", current_symptoms)
	else:
		st.info("No assistant answer found yet. Chat first to auto-suggest diagnoses and medications.")

	with st.form(key=f"new_visit_form_{patient_id}", clear_on_submit=False):
		c1, c2 = st.columns(2)
		with c1:
			visit_id = st.text_input("Visit ID", value=next_visit, disabled=True)
		with c2:
			visit_date = st.date_input("Visit date", value=date.today())

		note = st.text_area("Clinical note", height=160, placeholder="Enter physician/nurse note for this new visit")

		st.text_area(
			"Diagnoses (auto from LLM + existing latest visit)",
			value=", ".join(final_diagnoses),
			disabled=True,
			help="This field is auto-derived and locked.",
		)
		st.text_area(
			"Medications (auto from LLM + existing latest visit)",
			value=", ".join(final_medications),
			disabled=True,
			help="This field is auto-derived and locked.",
		)
		symptoms_text = st.text_input(
			"Symptoms (comma-separated)",
			placeholder="Headache, Nausea",
			help="Entered symptoms will be appended to latest-visit symptoms (deduplicated).",
		)
		labs_text = st.text_area(
			"Labs (one per line, format key:value)",
			height=120,
			placeholder="BP_systolic: 150\nBP_diastolic: 95\nHbA1c: 8.1",
		)

		submit = st.form_submit_button("Create New Visit and Update All")

	if not submit:
		return

	diagnoses = final_diagnoses
	medications = final_medications
	new_symptoms = _split_csv_field(symptoms_text)
	symptoms = _merge_unique_preserve_order(current_symptoms, new_symptoms)
	labs = _parse_labs_input(labs_text)

	if not note.strip():
		st.error("Clinical note is required.")
		return
	if not diagnoses:
		st.error("No diagnosis detected from LLM answer and existing history. Ask chat again with a clearer clinical recommendation.")
		return
	if not medications:
		st.error("No medication detected from LLM answer and existing history. Ask chat again with a clearer medication recommendation.")
		return
	if not symptoms:
		st.error("No symptoms provided and no existing symptoms found to append. Enter at least one symptom.")
		return

	st.info(f"Final symptoms for {next_visit}: {', '.join(symptoms)}")

	new_visit = {
		"visit_id": next_visit,
		"date": visit_date.isoformat(),
		"note": note.strip(),
		"labs": labs,
		"medications": medications,
		"diagnoses": diagnoses,
		"symptoms": symptoms,
		"extracted": {
			"conditions": diagnoses,
			"medications": medications,
			"symptoms": symptoms,
			"procedures": [],
			"relationships": [],
			"lab_values": labs,
		},
		"source": "streamlit_new_visit_intake",
	}

	with st.spinner("Applying updates to patient JSON, KG, and vector memory..."):
		try:
			_append_visit_to_patient_json(patient_id, new_visit)
			evolve_patient_graph(patient_id, new_visit, graphs_dir=str(GRAPHS_DIR))
			_update_vector_memory(patient_id, new_visit)
			_refresh_after_updates(patient_id)
			st.success(
				f"New visit {next_visit} created and propagated: patient JSON appended, graph evolved, memory updated, and visuals refreshed."
			)
			st.rerun()
		except Exception as exc:
			st.error(f"Failed to apply new visit updates: {exc}")


def main() -> None:
	st.title("GraphMed Interactive Demo")
	st.caption("Phase 11 demo for graph-based patient reasoning and temporal evolution.")

	_init_session_state()
	patient_ids = list_patient_ids()
	if not patient_ids:
		st.error("No patients found in data/graphs or data/patients_processed.")
		return

	selected_patient = render_sidebar(patient_ids)
	if not selected_patient:
		st.warning("Select a patient from the sidebar to begin.")
		return

	graph_data = load_graph_dict(selected_patient)
	processed_data = load_processed_patient(selected_patient)

	graph_tab, chat_tab, evolution_tab, new_visit_tab = st.tabs(
		["Knowledge Graph", "Clinical Chat + Traces", "Graph Evolution", "New Visit Intake"]
	)

	with graph_tab:
		render_graph_tab(selected_patient, graph_data)

	with chat_tab:
		render_chat_tab(selected_patient)

	with evolution_tab:
		render_evolution_tab(selected_patient, processed_data)

	with new_visit_tab:
		render_new_visit_tab(selected_patient, processed_data)


if __name__ == "__main__":
	main()
