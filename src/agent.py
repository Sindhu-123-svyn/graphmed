"""
Phase 6: ReAct Agent with LangGraph for GraphMed
Integrates knowledge graphs, vector memory, and medical KB into a single agent
"""

import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from pathlib import Path
from datetime import datetime
import operator

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import PatientKnowledgeGraph
from src.memory import GlobalMemoryManager, PatientMemoryStore
from src.medical_kb import MedicalKnowledgeBase

import os
from dotenv import load_dotenv

load_dotenv()


# Define Agent State
class AgentState(TypedDict):
    """State for the ReAct agent."""
    messages: Annotated[List[BaseMessage], operator.add]
    patient_id: str
    current_query: str
    reasoning_steps: List[str]
    tool_calls_made: List[str]


class GraphMedAgent:
    """
    ReAct agent for clinical reasoning with GraphMed.
    Combines knowledge graphs, vector memory, and medical knowledge base.
    """
    
    def __init__(self, persist_dir: str = "data"):
        """
        Initialize the GraphMed agent.
        
        Args:
            persist_dir: Base directory for persisted data
        """
        self.persist_dir = Path(persist_dir)
        
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
        
        # Define tools
        self.tools = self._create_tools()
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.tool_executor = ToolExecutor(self.tools)
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
        
        print("✅ GraphMed Agent initialized with LangGraph")
        print(f"   Tools available: {len(self.tools)}")
        for tool in self.tools:
            print(f"     - {tool.name}")
    
    def _load_patient_graph(self, patient_id: str) -> PatientKnowledgeGraph:
        """Load a patient's knowledge graph from disk."""
        if patient_id in self._loaded_graphs:
            return self._loaded_graphs[patient_id]
        
        graph_path = self.graphs_dir / f"{patient_id}_graph.json"
        if graph_path.exists():
            pkg = PatientKnowledgeGraph.load(str(graph_path))
            self._loaded_graphs[patient_id] = pkg
            return pkg
        else:
            return PatientKnowledgeGraph(patient_id)
    
    def _create_tools(self):
        """Create the tools for the agent."""
        
        @tool
        def query_patient_graph(patient_id: str, query_type: str) -> str:
            """
            Query the patient's knowledge graph for structured medical facts.
            
            Args:
                patient_id: Patient identifier (e.g., "P001")
                query_type: Type of information - "conditions", "medications", "symptoms", "labs", or "all"
            
            Returns:
                Structured information from the patient's graph
            """
            try:
                # This will be bound to self when the method is called
                # We need to access self through the closure
                agent = globals().get('self')
                # Instead, we'll use the instance method
                return self._query_patient_graph_impl(patient_id, query_type)
            except Exception as e:
                return f"Error querying patient graph: {e}"
        
        @tool
        def retrieve_patient_memory(patient_id: str, query: str) -> str:
            """
            Retrieve relevant past visit summaries based on semantic similarity.
            
            Args:
                patient_id: Patient identifier (e.g., "P001")
                query: The clinical question or topic to search for
            
            Returns:
                Relevant past visit summaries
            """
            try:
                return self._retrieve_patient_memory_impl(patient_id, query)
            except Exception as e:
                return f"Error retrieving patient memory: {e}"
        
        @tool
        def query_medical_knowledge(question: str) -> str:
            """
            Query the external medical knowledge base for drug interactions, disease information, etc.
            
            Args:
                question: Clinical question about drugs, diseases, or medical guidelines
            
            Returns:
                Medical knowledge from verified sources
            """
            try:
                return self._query_medical_knowledge_impl(question)
            except Exception as e:
                return f"Error querying medical knowledge: {e}"
        
        @tool
        def check_drug_interaction(drug1: str, drug2: str) -> str:
            """
            Check if two drugs have a known interaction.
            
            Args:
                drug1: First medication name
                drug2: Second medication name
            
            Returns:
                Information about potential drug interaction
            """
            try:
                return self._check_drug_interaction_impl(drug1, drug2)
            except Exception as e:
                return f"Error checking drug interaction: {e}"
        
        @tool
        def get_disease_info(disease: str) -> str:
            """
            Get information about a disease including symptoms, diagnosis, and treatment.
            
            Args:
                disease: Name of the disease (e.g., "Type 2 Diabetes", "Hypertension")
            
            Returns:
                Disease information from medical knowledge base
            """
            try:
                return self._get_disease_info_impl(disease)
            except Exception as e:
                return f"Error getting disease info: {e}"
        
        @tool
        def get_medication_info(medication: str) -> str:
            """
            Get information about a medication including indications and side effects.
            
            Args:
                medication: Name of the medication
            
            Returns:
                Medication information from medical knowledge base
            """
            try:
                return self._get_medication_info_impl(medication)
            except Exception as e:
                return f"Error getting medication info: {e}"
        
        @tool
        def get_lab_reference(lab_name: str) -> str:
            """
            Get reference ranges and interpretation for a lab value.
            
            Args:
                lab_name: Name of the lab test (e.g., "HbA1c", "eGFR")
            
            Returns:
                Lab reference information
            """
            try:
                return self._get_lab_reference_impl(lab_name)
            except Exception as e:
                return f"Error getting lab reference: {e}"
        
        @tool
        def get_clinical_guideline(condition: str) -> str:
            """
            Get clinical guideline for a medical condition.
            
            Args:
                condition: Medical condition (e.g., "Type 2 Diabetes", "Hypertension")
            
            Returns:
                Clinical guideline information
            """
            try:
                return self._get_clinical_guideline_impl(condition)
            except Exception as e:
                return f"Error getting clinical guideline: {e}"
        
        return [
            query_patient_graph,
            retrieve_patient_memory,
            query_medical_knowledge,
            check_drug_interaction,
            get_disease_info,
            get_medication_info,
            get_lab_reference,
            get_clinical_guideline
        ]
    
    # Implementation methods for tools
    def _query_patient_graph_impl(self, patient_id: str, query_type: str) -> str:
        """Implementation of query_patient_graph."""
        try:
            pkg = self._load_patient_graph(patient_id)
            
            if query_type == "conditions":
                entities = pkg.get_entities_by_type("CONDITION")
                if not entities:
                    return "No conditions found in patient record."
                return f"Patient's conditions: {', '.join([data.get('name') for _, data in entities])}"
            
            elif query_type == "medications":
                entities = pkg.get_entities_by_type("MEDICATION")
                if not entities:
                    return "No medications found in patient record."
                return f"Patient's medications: {', '.join([data.get('name') for _, data in entities])}"
            
            elif query_type == "symptoms":
                entities = pkg.get_entities_by_type("SYMPTOM")
                if not entities:
                    return "No symptoms documented."
                return f"Patient's documented symptoms: {', '.join([data.get('name') for _, data in entities])}"
            
            elif query_type == "labs":
                entities = pkg.get_entities_by_type("LAB_VALUE")
                if not entities:
                    return "No lab values found."
                labs_info = []
                for _, data in entities:
                    name = data.get('name')
                    value = data.get('value', 'N/A')
                    labs_info.append(f"{name}: {value}")
                return f"Patient's lab values: {', '.join(labs_info)}"
            
            else:  # all
                summary = pkg.summary()
                return f"Patient graph summary: {summary['total_nodes']} nodes, {summary['total_edges']} edges. Conditions: {summary['node_types'].get('CONDITION', 0)}, Medications: {summary['node_types'].get('MEDICATION', 0)}"
                
        except Exception as e:
            return f"Error querying patient graph: {e}"
    
    def _retrieve_patient_memory_impl(self, patient_id: str, query: str) -> str:
        """Implementation of retrieve_patient_memory."""
        try:
            store = self.memory_manager.get_patient_store(patient_id)
            results = store.retrieve_similar(query, top_k=3)
            
            if not results:
                return "No relevant past visits found."
            
            response = "Relevant past visits:\n\n"
            for i, result in enumerate(results, 1):
                response += f"[Visit {i}]\n"
                response += f"Date: {result['metadata'].get('date', 'Unknown')}\n"
                response += f"Summary: {result['document'][:300]}...\n\n"
            
            return response
            
        except Exception as e:
            return f"Error retrieving patient memory: {e}"
    
    def _query_medical_knowledge_impl(self, question: str) -> str:
        """Implementation of query_medical_knowledge."""
        try:
            results = self.medical_kb.query(question, top_k=3)
            
            if not results:
                return "No relevant medical knowledge found."
            
            response = "Medical knowledge base results:\n\n"
            for i, result in enumerate(results, 1):
                doc_type = result['metadata'].get('type', 'unknown')
                response += f"[{i}] {doc_type.replace('_', ' ').title()}:\n"
                response += f"{result['document'][:400]}\n\n"
            
            return response
            
        except Exception as e:
            return f"Error querying medical knowledge: {e}"
    
    def _check_drug_interaction_impl(self, drug1: str, drug2: str) -> str:
        """Implementation of check_drug_interaction."""
        try:
            result = self.medical_kb.check_drug_interaction(drug1, drug2)
            
            if result:
                return f"Drug Interaction Found:\n{result['document']}"
            else:
                return f"No known interaction found between {drug1} and {drug2}. Always consult a healthcare provider."
                
        except Exception as e:
            return f"Error checking drug interaction: {e}"
    
    def _get_disease_info_impl(self, disease: str) -> str:
        """Implementation of get_disease_info."""
        try:
            results = self.medical_kb.get_disease_info(disease)
            
            if results:
                return f"Disease Information for {disease}:\n{results[0]['document']}"
            else:
                return f"No disease information found for {disease}."
                
        except Exception as e:
            return f"Error getting disease info: {e}"
    
    def _get_medication_info_impl(self, medication: str) -> str:
        """Implementation of get_medication_info."""
        try:
            results = self.medical_kb.get_medication_info(medication)
            
            if results:
                return f"Medication Information for {medication}:\n{results[0]['document']}"
            else:
                return f"No medication information found for {medication}."
                
        except Exception as e:
            return f"Error getting medication info: {e}"
    
    def _get_lab_reference_impl(self, lab_name: str) -> str:
        """Implementation of get_lab_reference."""
        try:
            query = f"Lab reference for {lab_name}"
            results = self.medical_kb.query(query, top_k=2)
            
            for result in results:
                if result['metadata'].get('type') == 'lab_interpretation':
                    return f"Lab Reference for {lab_name}:\n{result['document']}"
            
            return f"No lab reference found for {lab_name}."
            
        except Exception as e:
            return f"Error getting lab reference: {e}"
    
    def _get_clinical_guideline_impl(self, condition: str) -> str:
        """Implementation of get_clinical_guideline."""
        try:
            results = self.medical_kb.query(f"Guideline for {condition}", top_k=2)
            
            for result in results:
                if result['metadata'].get('type') == 'clinical_guideline':
                    return f"Clinical Guideline for {condition}:\n{result['document']}"
            
            return f"No clinical guideline found for {condition}."
            
        except Exception as e:
            return f"Error getting clinical guideline: {e}"
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent graph."""
        
        # Define the agent node
        def agent_node(state: AgentState):
            """Call the LLM to decide which tool to use."""
            messages = state["messages"]
            
            # Add system prompt as first message if not present
            system_prompt = SystemMessage(content="""You are GraphMed, an AI clinical reasoning assistant with access to:
1. query_patient_graph - Get patient's conditions, medications, symptoms, or labs
2. retrieve_patient_memory - Search past visit summaries semantically  
3. query_medical_knowledge - Get drug interactions, disease info, clinical guidelines
4. check_drug_interaction - Check if two drugs interact
5. get_disease_info - Get detailed disease information
6. get_medication_info - Get medication details and side effects
7. get_lab_reference - Get normal ranges for lab tests
8. get_clinical_guideline - Get treatment guidelines

IMPORTANT RULES:
- Always query the patient's graph before answering patient-specific questions
- Check drug interactions before recommending medications
- Ground all medical claims in the knowledge base
- Cite which tool provided each piece of information
- If unsure, say so and suggest what additional information would help
- Be clear that you're an AI assistant, not a replacement for medical professionals

When using tools, provide the required parameters. For patient queries, always include patient_id.
Format your responses clearly and include appropriate disclaimers.""")

            # Check if system message already exists
            has_system = any(isinstance(m, SystemMessage) for m in messages)
            if not has_system:
                messages = [system_prompt] + messages
            
            # Call LLM with tools
            response = self.llm_with_tools.invoke(messages)
            
            return {
                "messages": [response],
                "reasoning_steps": state.get("reasoning_steps", []) + [f"Agent thought: {response.content[:100] if response.content else 'No content'}..."]
            }
        
        # Define tool execution node
        def tool_node(state: AgentState):
            """Execute the tool requested by the agent."""
            messages = state["messages"]
            last_message = messages[-1]
            
            # Execute tools
            tool_results = []
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    try:
                        result = self.tool_executor.invoke(tool_call)
                        tool_results.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call["id"]
                            )
                        )
                    except Exception as e:
                        tool_results.append(
                            ToolMessage(
                                content=f"Tool error: {e}",
                                tool_call_id=tool_call["id"]
                            )
                        )
            
            return {
                "messages": tool_results,
                "tool_calls_made": state.get("tool_calls_made", []) + [tc["name"] for tc in last_message.tool_calls] if hasattr(last_message, "tool_calls") and last_message.tool_calls else [],
                "reasoning_steps": state.get("reasoning_steps", []) + [f"Executed tools: {[tc['name'] for tc in last_message.tool_calls]}" if hasattr(last_message, "tool_calls") and last_message.tool_calls else "No tools executed"]
            }
        
        # Define router
        def should_continue(state: AgentState):
            """Determine whether to continue or end."""
            messages = state["messages"]
            last_message = messages[-1]
            
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        
        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def invoke(self, patient_id: str, query: str, max_iterations: int = 5) -> Dict[str, Any]:
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
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=f"Patient ID: {patient_id}\n\nQuestion: {query}")],
            "patient_id": patient_id,
            "current_query": query,
            "reasoning_steps": [],
            "tool_calls_made": []
        }
        
        # Run the agent
        try:
            result = self.graph.invoke(initial_state, {"recursion_limit": max_iterations * 2})
            
            # Extract final answer
            final_message = result["messages"][-1] if result["messages"] else None
            answer = final_message.content if final_message and hasattr(final_message, "content") else "No response generated."
            
            print(f"\n{'='*60}")
            print(f"🤖 GraphMed Response:")
            print(f"{'='*60}")
            print(answer)
            print(f"\n{'='*60}")
            print(f"📊 Reasoning Summary:")
            print(f"   Tools called: {result.get('tool_calls_made', [])}")
            print(f"   Reasoning steps: {len(result.get('reasoning_steps', []))}")
            print(f"{'='*60}")
            
            return {
                "answer": answer,
                "reasoning_steps": result.get("reasoning_steps", []),
                "tools_called": result.get("tool_calls_made", []),
                "full_trace": result
            }
        except Exception as e:
            error_msg = f"Agent error: {e}"
            print(f"\n❌ {error_msg}")
            return {
                "answer": error_msg,
                "reasoning_steps": [],
                "tools_called": [],
                "full_trace": None
            }


def test_agent():
    """Test the GraphMed agent with sample queries."""
    
    print("\n" + "="*60)
    print("🧪 TESTING GRAPHMED AGENT WITH LANGGRAPH")
    print("="*60)
    
    agent = GraphMedAgent()
    
    test_queries = [
        ("P001", "What medical conditions does this patient have?"),
        ("P001", "What medications is the patient taking?"),
    ]
    
    for patient_id, query in test_queries:
        print(f"\n{'─'*50}")
        result = agent.invoke(patient_id, query)
        print(f"{'─'*50}")


if __name__ == "__main__":
    test_agent()