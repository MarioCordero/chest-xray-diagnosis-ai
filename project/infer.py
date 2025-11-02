from fastai.vision.all import *
import sys
learn = load_learner("outputs/export.pkl")
img = Path(sys.argv[1])
pred, idx, probs = learn.predict(img)
print(f"Predicción: {pred} | P(NORMAL)={probs[learn.dls.vocab.o2i['NORMAL']]:.3f} | P(PNEUMONIA)={probs[learn.dls.vocab.o2i['PNEUMONIA']]:.3f}")