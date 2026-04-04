"""
Phase 5: Medical RAG Knowledge Base for GraphMed
Priority ingestion pipeline:
1) DrugBank Open Data
2) CDC Clinical Guidelines
3) PubMed (targeted)
4) MedlinePlus Drug Information (optional)
"""

import csv
import json
import logging
import os
import re
import time
import uuid
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def _disable_chroma_telemetry_noise() -> None:
    """Disable telemetry calls that can fail with incompatible PostHog versions."""
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["CHROMA_TELEMETRY"] = "False"

    try:
        import posthog  # type: ignore

        def _capture_noop(*args, **kwargs):
            return None

        posthog.capture = _capture_noop
    except Exception:
        pass


_disable_chroma_telemetry_noise()

if load_dotenv is not None:
    load_dotenv()

import chromadb
from chromadb.config import Settings

warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry.product").setLevel(logging.ERROR)


SOURCE_PRIORITY = {
    "drugbank": 1,
    "cdc": 2,
    "pubmed": 3,
    "medlineplus": 4,
}


PREFERRED_DOC_TYPES = {
    "interaction": ["drug_interaction", "drug_disease", "medication_info"],
    "guideline": ["clinical_guideline", "disease_info"],
    "drug": ["medication_info", "drug_interaction"],
    "disease": ["disease_info", "clinical_guideline"],
    "lab": ["lab_interpretation", "clinical_guideline"],
    "general": ["clinical_guideline", "disease_info", "medication_info", "drug_interaction"],
}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class KBChunk:
    """Canonical chunk object for all source ingestors."""

    doc_id: str
    text: str
    metadata: Dict[str, Any]


class MedicalKnowledgeBase:
    """External medical KB with source-priority ingestion and weighted retrieval."""

    def __init__(self, persist_directory: str = "data/medical_kb"):
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        try:
            self.collection = self.client.get_or_create_collection(
                name="medical_knowledge_base",
                metadata={
                    "description": "External medical knowledge for GraphMed grounding",
                    "schema_version": "phase5_v2",
                },
            )
        except Exception:
            self.collection = self.client.get_collection(name="medical_knowledge_base")

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("[OK] Medical Knowledge Base initialized")
        print(f"   Existing entries: {self.collection.count()}")

    # ---------------------------------------------------------------------
    # Chunking and metadata schema
    # ---------------------------------------------------------------------
    def _normalize_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        return cleaned

    def _split_into_chunks(self, text: str, chunk_chars: int = 1200, overlap_chars: int = 220) -> List[str]:
        """
        Character-based chunking to keep broad source compatibility.
        Uses overlap to preserve context continuity across chunk boundaries.
        """
        text = self._normalize_text(text)
        if not text:
            return []

        if len(text) <= chunk_chars:
            return [text]

        chunks: List[str] = []
        start = 0
        step = max(1, chunk_chars - overlap_chars)

        while start < len(text):
            end = min(len(text), start + chunk_chars)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start += step

        return chunks

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Chroma metadata supports scalar values only."""
        out: Dict[str, Any] = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            else:
                out[k] = str(v)
        return out

    def _build_chunk_metadata(
        self,
        base_meta: Dict[str, Any],
        source: str,
        chunk_index: int,
        chunk_count: int,
        chunk_text: str,
    ) -> Dict[str, Any]:
        meta = dict(base_meta)
        meta["source"] = source
        meta["source_priority"] = SOURCE_PRIORITY.get(source, 9)
        meta["chunk_index"] = chunk_index
        meta["chunk_count"] = chunk_count
        meta["char_count"] = len(chunk_text)
        meta["ingested_at"] = _utc_iso()
        meta["schema_version"] = "phase5_v2"
        return self._sanitize_metadata(meta)

    def _to_chunks(
        self,
        source: str,
        doc_type: str,
        title: str,
        text: str,
        base_meta: Optional[Dict[str, Any]] = None,
        chunk_chars: int = 1200,
        overlap_chars: int = 220,
    ) -> List[KBChunk]:
        base_meta = base_meta or {}
        base_meta.update({
            "type": doc_type,
            "category": doc_type,
            "title": title,
        })

        chunks = self._split_into_chunks(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        out: List[KBChunk] = []

        for idx, chunk in enumerate(chunks):
            doc_id = f"{source}_{doc_type}_{uuid.uuid4().hex[:12]}_{idx}"
            meta = self._build_chunk_metadata(base_meta, source, idx, len(chunks), chunk)
            out.append(KBChunk(doc_id=doc_id, text=chunk, metadata=meta))

        return out

    def _upsert_chunks(self, chunks: List[KBChunk]) -> int:
        if not chunks:
            return 0

        ids = [c.doc_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [c.metadata for c in chunks]
        embeddings = self.embedder.encode(docs).tolist()

        self.collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )
        return len(chunks)

    # ---------------------------------------------------------------------
    # Ingestion APIs (priority ordered)
    # ---------------------------------------------------------------------
    def ingest_drugbank_json(self, file_path: str) -> int:
        """
        Ingest DrugBank structured export converted to JSON.
        Expected top-level: list[drug], where each drug can include:
        - name, description, mechanism_of_action, indications, contraindications,
          adverse_effects, interactions(list)
        """
        path = Path(file_path)
        if not path.exists():
            return 0

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        drugs = payload if isinstance(payload, list) else payload.get("drugs", [])
        total = 0

        for drug in drugs:
            name = str(drug.get("name", "")).strip()
            if not name:
                continue

            med_text = "\n".join([
                f"Drug: {name}",
                f"Description: {drug.get('description', '')}",
                f"Mechanism: {drug.get('mechanism_of_action', '')}",
                f"Indications: {drug.get('indications', '')}",
                f"Contraindications: {drug.get('contraindications', '')}",
                f"Adverse Effects: {drug.get('adverse_effects', '')}",
            ]).strip()

            med_chunks = self._to_chunks(
                source="drugbank",
                doc_type="medication_info",
                title=f"DrugBank {name}",
                text=med_text,
                base_meta={
                    "medication": name.lower(),
                    "evidence_level": "reference",
                    "dataset": "drugbank_open",
                },
            )
            total += self._upsert_chunks(med_chunks)

            for interaction in drug.get("interactions", []) or []:
                target = str(interaction.get("target", "")).strip()
                if not target:
                    continue
                interaction_text = "\n".join([
                    f"Interaction: {name} with {target}",
                    f"Severity: {interaction.get('severity', 'unknown')}",
                    f"Description: {interaction.get('description', '')}",
                    f"Recommendation: {interaction.get('recommendation', '')}",
                ]).strip()

                int_chunks = self._to_chunks(
                    source="drugbank",
                    doc_type="drug_interaction",
                    title=f"DrugBank interaction {name} + {target}",
                    text=interaction_text,
                    base_meta={
                        "drug1": name.lower(),
                        "drug2": target.lower(),
                        "severity": str(interaction.get("severity", "unknown")).lower(),
                        "evidence_level": "reference",
                        "dataset": "drugbank_open",
                    },
                    chunk_chars=900,
                    overlap_chars=120,
                )
                total += self._upsert_chunks(int_chunks)

        return total

    def ingest_cdc_guidelines_dir(self, guidelines_dir: str) -> int:
        """
        Ingest CDC guideline text/PDF files from local directory.
        Supported: .txt, .md and .pdf (requires pdfplumber).
        """
        base = Path(guidelines_dir)
        if not base.exists():
            return 0

        total = 0
        files = sorted(base.glob("**/*"))
        for path in files:
            if not path.is_file():
                continue

            suffix = path.suffix.lower()
            text = ""

            if suffix in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
            elif suffix == ".pdf":
                try:
                    import pdfplumber  # type: ignore

                    pages: List[str] = []
                    with pdfplumber.open(str(path)) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text() or ""
                            if page_text.strip():
                                pages.append(page_text)
                    text = "\n".join(pages)
                except Exception:
                    # Skip PDF if parser unavailable or file malformed.
                    continue
            else:
                continue

            text = self._normalize_text(text)
            if not text:
                continue

            topic = path.stem.replace("_", " ")
            chunks = self._to_chunks(
                source="cdc",
                doc_type="clinical_guideline",
                title=f"CDC {topic}",
                text=text,
                base_meta={
                    "condition": topic.lower(),
                    "source_org": "CDC",
                    "dataset": "cdc_guidelines",
                    "evidence_level": "guideline",
                    "file_name": path.name,
                },
                chunk_chars=1400,
                overlap_chars=260,
            )
            total += self._upsert_chunks(chunks)

        return total

    def ingest_pubmed_topics(
        self,
        topics: List[str],
        email: str,
        max_per_topic: int = 25,
    ) -> int:
        """Targeted PubMed ingestion (title + abstract chunks) for specific topics."""
        if not topics:
            return 0

        total = 0
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

        for topic in topics:
            params = {
                "db": "pubmed",
                "retmode": "json",
                "retmax": max_per_topic,
                "sort": "relevance",
                "term": topic,
                "email": email,
            }
            try:
                search = requests.get(f"{base}/esearch.fcgi", params=params, timeout=30)
                search.raise_for_status()
                ids = search.json().get("esearchresult", {}).get("idlist", [])
            except Exception:
                continue

            if not ids:
                continue

            try:
                fetch = requests.get(
                    f"{base}/efetch.fcgi",
                    params={
                        "db": "pubmed",
                        "id": ",".join(ids),
                        "retmode": "xml",
                        "email": email,
                    },
                    timeout=45,
                )
                fetch.raise_for_status()
                root = ET.fromstring(fetch.text)
            except Exception:
                continue

            for article in root.findall(".//PubmedArticle"):
                pmid = (article.findtext(".//PMID") or "").strip()
                title = (article.findtext(".//ArticleTitle") or "").strip()

                abstract_parts = []
                for a in article.findall(".//Abstract/AbstractText"):
                    label = a.attrib.get("Label", "").strip()
                    text = "".join(a.itertext()).strip()
                    if not text:
                        continue
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)

                abstract = " ".join(abstract_parts).strip()
                if not (title or abstract):
                    continue

                year = ""
                year_node = article.find(".//PubDate/Year")
                if year_node is not None and year_node.text:
                    year = year_node.text.strip()

                journal = (article.findtext(".//Journal/Title") or "").strip()
                article_text = "\n".join([
                    f"Topic: {topic}",
                    f"PMID: {pmid}",
                    f"Title: {title}",
                    f"Journal: {journal}",
                    f"Year: {year}",
                    f"Abstract: {abstract}",
                ]).strip()

                chunks = self._to_chunks(
                    source="pubmed",
                    doc_type="disease_info",
                    title=f"PubMed {topic} {pmid}",
                    text=article_text,
                    base_meta={
                        "condition": topic.lower(),
                        "pmid": pmid,
                        "journal": journal,
                        "year": year,
                        "dataset": "pubmed_eutils",
                        "evidence_level": "literature",
                    },
                )
                total += self._upsert_chunks(chunks)

                # Respect public API limits.
                time.sleep(0.12)

        return total

    def ingest_medlineplus_csv(self, file_path: str) -> int:
        """
        Ingest MedlinePlus drug info exported as CSV.
        Required columns: drug_name, summary
        Optional: uses, warnings, side_effects, source_url
        """
        path = Path(file_path)
        if not path.exists():
            return 0

        total = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (
                    row.get("drug_name")
                    or row.get("Drug Name")
                    or row.get("name")
                    or row.get("Name")
                    or ""
                ).strip()
                summary = (
                    row.get("summary")
                    or row.get("Summary")
                    or row.get("description")
                    or row.get("Description")
                    or ""
                ).strip()
                if not (name and summary):
                    continue

                text = "\n".join([
                    f"Drug: {name}",
                    f"Summary: {summary}",
                    f"Uses: {row.get('uses', '')}",
                    f"Warnings: {row.get('warnings', '')}",
                    f"Side Effects: {row.get('side_effects', '')}",
                ]).strip()

                chunks = self._to_chunks(
                    source="medlineplus",
                    doc_type="medication_info",
                    title=f"MedlinePlus {name}",
                    text=text,
                    base_meta={
                        "medication": name.lower(),
                        "dataset": "medlineplus",
                        "source_url": row.get("source_url", ""),
                        "evidence_level": "reference",
                    },
                )
                total += self._upsert_chunks(chunks)

        return total

    def ingest_priority_pipeline(
        self,
        drugbank_json: Optional[str] = None,
        cdc_guidelines_dir: Optional[str] = None,
        pubmed_topics: Optional[List[str]] = None,
        pubmed_email: Optional[str] = None,
        pubmed_max_per_topic: int = 25,
        medlineplus_csv: Optional[str] = None,
    ) -> Dict[str, int]:
        """Run ingestion in required priority order."""
        stats = {
            "drugbank_chunks": 0,
            "cdc_chunks": 0,
            "pubmed_chunks": 0,
            "medlineplus_chunks": 0,
            "total_chunks": 0,
        }

        if drugbank_json:
            stats["drugbank_chunks"] = self.ingest_drugbank_json(drugbank_json)

        if cdc_guidelines_dir:
            stats["cdc_chunks"] = self.ingest_cdc_guidelines_dir(cdc_guidelines_dir)

        if pubmed_topics and pubmed_email:
            stats["pubmed_chunks"] = self.ingest_pubmed_topics(
                topics=pubmed_topics,
                email=pubmed_email,
                max_per_topic=pubmed_max_per_topic,
            )

        if medlineplus_csv:
            stats["medlineplus_chunks"] = self.ingest_medlineplus_csv(medlineplus_csv)

        stats["total_chunks"] = (
            stats["drugbank_chunks"]
            + stats["cdc_chunks"]
            + stats["pubmed_chunks"]
            + stats["medlineplus_chunks"]
        )
        return stats

    # ---------------------------------------------------------------------
    # Retrieval strategy (weighted rerank on top of vector similarity)
    # ---------------------------------------------------------------------
    def _detect_query_intent(self, question: str) -> str:
        q = question.lower()
        if any(k in q for k in ["interaction", "contraindicat", "safe to take", "co-administer"]):
            return "interaction"
        if any(k in q for k in ["guideline", "recommended", "first-line", "protocol", "cdc"]):
            return "guideline"
        if any(k in q for k in ["dose", "dosage", "drug", "medication", "side effect"]):
            return "drug"
        if any(k in q for k in ["disease", "symptom", "diagnos", "treat"]):
            return "disease"
        if any(k in q for k in ["lab", "hba1c", "egfr", "ldl", "blood pressure"]):
            return "lab"
        return "general"

    def _rank_score(self, distance: Optional[float], metadata: Dict[str, Any], intent: str) -> float:
        # Chroma distance: lower is better. Convert to similarity-like score.
        if distance is None:
            base = 0.0
        else:
            base = max(0.0, 1.0 - float(distance))

        doc_type = str(metadata.get("type", ""))
        source_priority = int(metadata.get("source_priority", 9))
        source_bonus = max(0.0, (5 - source_priority) * 0.04)

        preferred = PREFERRED_DOC_TYPES.get(intent, PREFERRED_DOC_TYPES["general"])
        type_bonus = 0.08 if doc_type in preferred else 0.0

        evidence_level = str(metadata.get("evidence_level", "")).lower()
        evidence_bonus = 0.04 if evidence_level in {"guideline", "reference"} else 0.0

        return base + source_bonus + type_bonus + evidence_bonus

    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Weighted retrieval strategy:
        1) broad semantic search from Chroma
        2) intent-aware reranking with source-priority weighting
        3) return top_k final chunks
        """
        query_embedding = self.embedder.encode(question).tolist()
        intent = self._detect_query_intent(question)

        try:
            raw = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max(12, top_k * 4),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        candidates: List[Dict[str, Any]] = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0] if raw.get("documents") else []
        metas = raw.get("metadatas", [[]])[0] if raw.get("metadatas") else []
        dists = raw.get("distances", [[]])[0] if raw.get("distances") else []

        for i, cid in enumerate(ids):
            doc = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else None
            score = self._rank_score(dist, meta, intent)
            candidates.append(
                {
                    "id": cid,
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                    "score": score,
                }
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    # ---------------------------------------------------------------------
    # Backward-compatible helper APIs used by agent
    # ---------------------------------------------------------------------
    def check_drug_interaction(self, drug1: str, drug2: str) -> Optional[Dict[str, Any]]:
        query = f"interaction between {drug1} and {drug2}"
        results = self.query(query, top_k=6)

        d1 = drug1.lower().strip()
        d2 = drug2.lower().strip()
        for r in results:
            meta = r.get("metadata", {})
            if meta.get("type") != "drug_interaction":
                continue
            md1 = str(meta.get("drug1", "")).lower()
            md2 = str(meta.get("drug2", "")).lower()
            if (d1 in md1 and d2 in md2) or (d1 in md2 and d2 in md1):
                return r
        return results[0] if results else None

    def get_disease_info(self, disease: str) -> List[Dict[str, Any]]:
        results = self.query(f"disease information and treatment for {disease}", top_k=6)
        return [r for r in results if r.get("metadata", {}).get("type") in {"disease_info", "clinical_guideline"}]

    def get_medication_info(self, medication: str) -> List[Dict[str, Any]]:
        results = self.query(f"medication information for {medication}", top_k=6)
        return [r for r in results if r.get("metadata", {}).get("type") in {"medication_info", "drug_interaction"}]

    def verify_fact(self, claim: str) -> Dict[str, Any]:
        results = self.query(claim, top_k=3)
        if not results:
            return {
                "verified": False,
                "confidence": 0.0,
                "evidence": None,
                "message": "No relevant medical knowledge found",
            }

        best = results[0]
        confidence = max(0.0, min(1.0, float(best.get("score", 0.0))))
        return {
            "verified": confidence >= 0.35,
            "confidence": confidence,
            "evidence": best.get("document"),
            "source_type": best.get("metadata", {}).get("type", "unknown"),
            "source": best.get("metadata", {}).get("source", "unknown"),
            "message": "Found supporting medical evidence",
        }

    def get_summary(self) -> Dict[str, Any]:
        try:
            all_docs = self.collection.get(include=["metadatas"])
            type_counts: Dict[str, int] = {}
            source_counts: Dict[str, int] = {}

            for meta in all_docs.get("metadatas", []) or []:
                doc_type = str(meta.get("type", "unknown"))
                source = str(meta.get("source", "unknown"))
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1

            return {
                "total_entries": self.collection.count(),
                "by_type": type_counts,
                "by_source": source_counts,
            }
        except Exception:
            return {"total_entries": 0, "by_type": {}, "by_source": {}}


def _default_pubmed_topics() -> List[str]:
    return [
        "type 2 diabetes guideline",
        "hypertension management",
        "chronic kidney disease treatment",
        "coronary artery disease medication",
        "heart failure guideline",
        "metformin contraindications",
        "lisinopril hyperkalemia",
        "warfarin nsaid bleeding interaction",
    ]


def create_medical_knowledge_base(
    persist_dir: str = "data/medical_kb",
    sources_dir: str = "data/medical_sources",
    rebuild: bool = False,
) -> MedicalKnowledgeBase:
    """
    Create and populate Phase 5 KB in strict source-priority order.

    Expected source files/directories:
    - data/medical_sources/drugbank/drugbank_open.json
    - data/medical_sources/cdc_guidelines/ (txt/md/pdf)
    - data/medical_sources/medlineplus/medlineplus_drugs.csv

    Optional environment variable:
    - PUBMED_EMAIL (required to fetch PubMed through E-utilities)
    """
    print("\n" + "=" * 64)
    print("GRAPHMED PHASE 5 - MEDICAL RAG KNOWLEDGE BASE")
    print("=" * 64)

    kb = MedicalKnowledgeBase(persist_dir)

    if rebuild:
        try:
            kb.client.delete_collection("medical_knowledge_base")
        except Exception:
            pass
        kb.collection = kb.client.get_or_create_collection(
            name="medical_knowledge_base",
            metadata={"description": "External medical knowledge for GraphMed grounding", "schema_version": "phase5_v2"},
        )

    source_root = Path(sources_dir)
    drugbank_json = source_root / "drugbank" / "drugbank_open.json"
    cdc_dir = source_root / "cdc_guidelines"
    medlineplus_csv = source_root / "medlineplus" / "medlineplus_drugs.csv"

    pubmed_email = os.getenv("PUBMED_EMAIL", "").strip()
    topics = _default_pubmed_topics()

    print("\nSource checks:")
    print(f"  DrugBank JSON present: {drugbank_json.exists()} ({drugbank_json})")
    print(f"  CDC directory present: {cdc_dir.exists()} ({cdc_dir})")
    print(f"  MedlinePlus CSV present: {medlineplus_csv.exists()} ({medlineplus_csv})")

    try:
        import pdfplumber  # type: ignore

        print("  pdfplumber available: True")
    except Exception:
        print("  pdfplumber available: False (CDC PDF ingestion will be skipped)")

    if medlineplus_csv.exists():
        try:
            with open(medlineplus_csv, "r", encoding="utf-8") as f:
                row_count = max(0, sum(1 for _ in f) - 1)
            print(f"  MedlinePlus data rows: {row_count}")
        except Exception:
            print("  MedlinePlus data rows: unknown (read error)")

    stats = kb.ingest_priority_pipeline(
        drugbank_json=str(drugbank_json) if drugbank_json.exists() else None,
        cdc_guidelines_dir=str(cdc_dir) if cdc_dir.exists() else None,
        pubmed_topics=topics,
        pubmed_email=pubmed_email if pubmed_email else None,
        pubmed_max_per_topic=20,
        medlineplus_csv=str(medlineplus_csv) if medlineplus_csv.exists() else None,
    )

    print("\nIngestion results (priority order):")
    print(f"  DrugBank chunks:     {stats['drugbank_chunks']}")
    print(f"  CDC chunks:          {stats['cdc_chunks']}")
    print(f"  PubMed chunks:       {stats['pubmed_chunks']}")
    print(f"  MedlinePlus chunks:  {stats['medlineplus_chunks']}")
    print(f"  Total chunks added:  {stats['total_chunks']}")

    summary = kb.get_summary()
    print("\nKnowledge base summary:")
    print(f"  Total entries: {summary['total_entries']}")
    print(f"  By source: {summary.get('by_source', {})}")
    print(f"  By type: {summary.get('by_type', {})}")

    if not pubmed_email:
        print("\n[INFO] PUBMED_EMAIL not set. PubMed ingestion skipped.")

    print("\n[OK] Phase 5 knowledge base ready.")
    return kb


if __name__ == "__main__":
    create_medical_knowledge_base()
