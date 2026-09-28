import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from app.rag.session_manager import SessionRAGManager
from app.agents.research_agent import ResearchAgent

rag = SessionRAGManager()
agent = ResearchAgent(rag)

pdfs = ['Attention.pdf', 'Belief.pdf', 'Orion.pdf']
texts = {
    'Attention.pdf': 'We propose a new computational model called Transformer. It is an algorithm that can be implemented as code. We train it on a dataset.',
    'Belief.pdf': 'We describe a computational method and reasoning system for belief maintenance. The algorithm can be implemented as code.',
    'Orion.pdf': 'Detection of Irregular, Sub-mm Opaque Structures in the Orion Molecular Clouds. We observe physical phenomena and astrophysics observations. We have some equations for radiative transfer and datasets of the molecular clouds. No computational method is proposed.'
}

from reportlab.pdfgen import canvas
for pdf, text in texts.items():
    c = canvas.Canvas(pdf)
    c.drawString(100, 750, text)
    c.save()

q = "Is this paper's main contribution a computational method, algorithm, model, or system that could be implemented as code? Answer only YES or NO, with a one-sentence reason."

for pdf in pdfs:
    s = rag.create_session()
    with open(pdf, 'rb') as f:
        rag.add_documents(s, [(pdf, f.read())])
    res = agent.run(session_id=s, query=q)
    print(f'\n--- {pdf} ---')
    print(res.answer)
