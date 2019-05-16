echo RQ6: Comparing the automated predictions to the unified non-expert controlled set 
echo \(Using only gold set functions for comparison\)
echo -----------------------------------------------------------------
echo 
echo ----------------- Baseline 12 token Approach --------------------
echo Text similarity as measured by ROUGE Scores \(comparing hypotheses to all non-empty references\):
echo
python3 ../metrics/rouge_metric.py ../datasets/pkl/unified_controlled_gold.pkl ../datasets/pkl/baselines_gold.pkl 
echo
echo Text similarity as measured by ROUGE Scores \(comparing only non-empty hypotheses to non-empty references\):
echo
python3 ../metrics/rouge_metric.py ../datasets/pkl/unified_controlled_gold.pkl ../datasets/pkl/baselines_gold.pkl  -ne
echo
echo \"Is-empty?\" binary classification similarity:
echo
python3 ../metrics/empty_classification_metric.py ../datasets/pkl/unified_controlled_gold.pkl ../datasets/pkl/baselines_gold.pkl 
echo
echo Unitization similarity as measured by Krippendorff\'s unitized alpha
echo
python3 ../metrics/krippendorff_metric.py ../datasets/pkl/unified_controlled_gold.pkl ../datasets/pkl/baselines_gold.pkl 

echo
echo ------------------ Standard BiLSTM Approach --------------------
echo Text similarity as measured by ROUGE Scores \(comparing hypotheses to all non-empty references\):
echo
python3 ../metrics/rouge_metric.py ../datasets/pkl/unified_controlled_gold.pkl ../datasets/pkl/bilstm_final.pkl 
echo
echo Text similarity as measured by ROUGE Scores \(comparing only non-empty hypotheses to non-empty references\):
echo
python3 ../metrics/rouge_metric.py ../datasets/pkl/unified_controlled_gold.pkl ../datasets/pkl/bilstm_final.pkl  -ne
echo
echo \"Is-empty?\" binary classification similarity:
echo
python3 ../metrics/empty_classification_metric.py ../datasets/pkl/unified_controlled_gold.pkl ../datasets/pkl/bilstm_final.pkl 
echo
echo Unitization similarity as measured by Krippendorff\'s unitized alpha
echo
python3 ../metrics/krippendorff_metric.py ../datasets/pkl/unified_controlled_gold.pkl ../datasets/pkl/bilstm_final.pkl 

echo
echo ----------- BiLSTM+F \(w/ Source Code\) Approach -----------------
echo Text similarity as measured by ROUGE Scores \(comparing hypotheses to all non-empty references\):
echo
python3 ../metrics/rouge_metric.py ../datasets/pkl/unified_controlled_gold.pkl ../datasets/pkl/bilstm_f_final.pkl 
echo
echo Text similarity as measured by ROUGE Scores \(comparing only non-empty hypotheses to non-empty references\):
echo
python3 ../metrics/rouge_metric.py ../datasets/pkl/unified_controlled_gold.pkl ../datasets/pkl/bilstm_f_final.pkl  -ne
echo
echo \"Is-empty?\" binary classification similarity:
echo
python3 ../metrics/empty_classification_metric.py ../datasets/pkl/unified_controlled_gold.pkl ../datasets/pkl/bilstm_f_final.pkl 
echo
echo Unitization similarity as measured by Krippendorff\'s unitized alpha
echo
python3 ../metrics/krippendorff_metric.py ../datasets/pkl/unified_controlled_gold.pkl ../datasets/pkl/bilstm_f_final.pkl 
echo