import fitz

def create_pdf(filename, text):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text, fontsize=12)
    doc.save(filename)
    doc.close()

text_attention = "We propose the Transformer, a novel neural network architecture based solely on attention mechanisms. The training optimization procedure minimizes cross-entropy loss. We trained on the WMT 2014 English-to-German dataset. The model has a dimension of 512, 8 heads, and 6 encoder and decoder layers. The core equation is softmax."

text_belief = "We introduce a Waltz-style propagation algorithm for belief maintenance in Bayesian Networks. This procedure uses a constraint satisfaction logic. The core probability equations involve the interval-extended chain rule. We demonstrate with a case study and test case. The threshold parameter beta is used."

text_orion = "We present observations of irregular, sub-mm opaque structures in the Orion Molecular Clouds. We used the ALMA telescope to measure emission spectra. The datasets include astronomical coordinates. The core physical models simulate stellar formation. No machine learning or neural networks were used."

create_pdf("Attention.pdf", text_attention)
create_pdf("Belief.pdf", text_belief)
create_pdf("Orion.pdf", text_orion)
print("PDFs generated.")
