This project implements a minimal Retrieval-Augmented Generation prototype as well as
a small evaluation script to measure the prototype regarding retriver and end-to-end answer.
# 1) Requirements:
    -  Python: 3.10+
    -  You should run locally an OpenAI-compatible server
    -  store your documents (PDF etc.) inside ./data/
    -  csv file for evaluation script
  
# 2) Install dependencies
   -  pip install -r requirements.txt
# 3) Run Prototype
  - python rag_min.py
# 4) Run Evaluation (Recall@k and Faithfulness)
  - python eval_rag.py --dataset gold.csv --k 5 --faith-threshold 0.70
