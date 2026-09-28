import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from app.rag.session_manager import SessionRAGManager
from app.agents.research_agent import ResearchAgent

rag = SessionRAGManager()
agent = ResearchAgent(rag)

pdf = r"D:\SENTRA AI\data\sessions\489eb6fb-194c-4812-8578-8d23ecc654c5\uploads\886d664f-a3f5-4dd0-a196-373e188fd21a_DETECTION OF IRREGULAR, SUB-MM OPAQUE STRUCTURES IN THE ORION MOLECULAR CLOUDS.pdf"
s = rag.create_session()
with open(pdf, 'rb') as f:
    rag.add_documents(s, [(os.path.basename(pdf), f.read())])

stage1_q = "Is this paper's main contribution a computational method, algorithm, model, or system that could be implemented as code? Answer only YES or NO, with a one-sentence reason."
res = agent.run(session_id=s, query=stage1_q)
print("ORION STAGE 1 ANSWER:", res.answer)
print("Sufficient evidence:", res.sufficient_evidence)
