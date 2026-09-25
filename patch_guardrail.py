import pathlib
import re

p = pathlib.Path('app/guardrails/evidence_guardrail.py')
t = p.read_text()

new_validate = """    def _attempt_resolve(self, req_name: str, requirements: dict, session_id: str, task_description: str, research_agent: ResearchAgent, attempt: int):
        from app.guardrails.models import Requirement, RequirementClassification, EvidenceState, ContextInfo
        from app.agents.models import EvidenceLink
        
        query = self._get_query_for_attempt(req_name, task_description, attempt)
        result = research_agent.run(session_id=session_id, query=query, top_k=5)
        
        if result.sufficient_evidence and result.evidence:
            supporting_evidences = []
            for ev in result.evidence:
                if self._explicitly_supports(req_name, str(ev.get("text", ""))):
                    supporting_evidences.append(ev)
            
            if supporting_evidences:
                best_ev = None
                for ev in supporting_evidences:
                    text_lower = str(ev.get("text", "")).lower()
                    task_lower = task_description.lower()
                    
                    if "cifar-10" in task_lower and "imagenet" in text_lower and "cifar-10" not in text_lower:
                        continue
                    if "imagenet" in task_lower and "cifar-10" in text_lower and "imagenet" not in text_lower:
                        continue
                    
                    best_ev = ev
                    break
                
                if best_ev:
                    ev_link = EvidenceLink(
                        source=str(best_ev.get("source", best_ev.get("filename", ""))),
                        page=int(best_ev.get("page", 0)),
                        chunk_id=str(best_ev.get("chunk_id", "")),
                        evidence_text=str(best_ev.get("text", ""))
                    )
                    
                    val = self._normalize_value(req_name, str(best_ev.get("text", "")))
                    if val is None:
                        val = str(best_ev.get("text", ""))
                        
                    keywords = research_agent._keywords(query)
                    excerpt = research_agent._best_sentence(str(best_ev["text"]), keywords)
                    answer = f"{val} (Based on {best_ev['source']}, page {best_ev['page']}: {excerpt})"
                    
                    ctx = ContextInfo(
                        document_id=ev_link.source,
                        experiment=task_description
                    )
                    
                    requirements[req_name] = Requirement(
                        name=req_name,
                        value=answer,
                        classification=RequirementClassification.PAPER_SUPPORTED,
                        state=EvidenceState.SUPPORTED,
                        evidence=ev_link,
                        context=ctx
                    )
                else:
                    if len(supporting_evidences) > 1:
                        requirements[req_name] = Requirement(
                            name=req_name,
                            state=EvidenceState.CONTEXT_MISMATCH,
                            reason="Evidence found but did not match target context."
                        )
                    else:
                        requirements[req_name] = Requirement(
                            name=req_name,
                            state=EvidenceState.NOT_FOUND,
                            reason="Explicit evidence not found."
                        )

    def validate(
        self, 
        session_id: str, 
        task_description: str,
        research_agent: ResearchAgent
    ) -> GuardrailResult:
        from app.guardrails.models import Requirement, RequirementClassification, EvidenceState, GuardrailTerminalState, PaperRelevance, ImplementationSupport
        
        paradigm = self._detect_paradigm(session_id, research_agent)
        
        critical_fields = self._get_critical_fields(paradigm)
        optional_fields = self._get_optional_fields(paradigm)
        all_possible_fields = self._get_all_possible_fields()
        
        requirements = {}
        for req in all_possible_fields:
            if req in optional_fields:
                requirements[req] = Requirement(
                    name=req, 
                    state=EvidenceState.SUPPORTED,
                    classification=RequirementClassification.IMPLEMENTATION_CHOICE,
                    value="Assigned to default implementation choice.",
                    reason="Optional engineering parameter mapped to default."
                )
            elif req not in critical_fields:
                requirements[req] = Requirement(
                    name=req, 
                    state=EvidenceState.NOT_APPLICABLE_TO_PARADIGM,
                    reason=f"Not applicable to {paradigm.value}"
                )
            else:
                requirements[req] = Requirement(name=req, state=EvidenceState.NOT_FOUND)
        
        # Pass 1: Automatic Spec Review
        for req_name in critical_fields:
            if requirements[req_name].state != EvidenceState.SUPPORTED:
                self._attempt_resolve(req_name, requirements, session_id, task_description, research_agent, attempt=1)
                
        all_critical_supported = all(requirements[req].state == EvidenceState.SUPPORTED for req in critical_fields)
        if all_critical_supported:
            return GuardrailResult(
                terminal_state=GuardrailTerminalState.PASS,
                attempt_count=1,
                requirements=requirements,
                session_id=session_id,
                code_generation_allowed=True,
                task_description=task_description,
                paper_relevance=PaperRelevance.SUPPORTED,
                implementation_support=ImplementationSupport.SUPPORTED,
                detected_paradigm=paradigm
            )
            
        # Pass 2: Targeted Doubt Queries
        attempt = 2
        while attempt <= self.max_attempts:
            missing_core_fields = [f for f in critical_fields if requirements[f].state != EvidenceState.SUPPORTED]
            for req_name in missing_core_fields:
                self._attempt_resolve(req_name, requirements, session_id, task_description, research_agent, attempt=attempt)
                
            all_critical_supported = all(requirements[req].state == EvidenceState.SUPPORTED for req in critical_fields)
            if all_critical_supported:
                return GuardrailResult(
                    terminal_state=GuardrailTerminalState.PASS,
                    attempt_count=attempt,
                    requirements=requirements,
                    session_id=session_id,
                    code_generation_allowed=True,
                    task_description=task_description,
                    paper_relevance=PaperRelevance.SUPPORTED,
                    implementation_support=ImplementationSupport.SUPPORTED,
                    detected_paradigm=paradigm
                )
            attempt += 1

        return GuardrailResult(
            terminal_state=GuardrailTerminalState.UNRESOLVED,
            attempt_count=self.max_attempts,
            requirements=requirements,
            reason="Could not retrieve sufficient evidence for all critical required fields after maximum attempts.",
            session_id=session_id,
            code_generation_allowed=False,
            task_description=task_description,
            paper_relevance=PaperRelevance.SUPPORTED,
            implementation_support=ImplementationSupport.APPLICABLE,
            detected_paradigm=paradigm
        )"""

# Find the start and end of validate
import re
match = re.search(r'    def validate\(.*?return GuardrailResult\([^)]*\)\n?', t, flags=re.DOTALL)
if match:
    t = t[:match.start()] + new_validate + "\n" + t[match.end():]
    p.write_text(t)
    print("Patched successfully")
else:
    print("Could not find validate method")
