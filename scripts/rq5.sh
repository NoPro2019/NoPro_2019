echo RQ5: Comparing the non-expert expanded set to the unified expert gold set 
echo \(Using only gold set functions for comparison\)
echo -----------------------------------------------------------------
echo Text similarity as measured by ROUGE Scores \(comparing hypotheses to all non-empty references\):
echo
python3 ../metrics/rouge_metric.py ../datasets/pkl/unified_expert_gold.pkl ../datasets/pkl/expanded_gold.pkl 
echo
echo Text similarity as measured by ROUGE Scores \(comparing only non-empty hypotheses to non-empty references\):
echo
python3 ../metrics/rouge_metric.py ../datasets/pkl/unified_expert_gold.pkl ../datasets/pkl/expanded_gold.pkl  -ne
echo
echo \"Is-empty?\" binary classification similarity:
echo
python3 ../metrics/empty_classification_metric.py ../datasets/pkl/unified_expert_gold.pkl ../datasets/pkl/expanded_gold.pkl 
echo
echo Unitization similarity as measured by Krippendorff\'s unitized alpha
echo
python3 ../metrics/krippendorff_metric.py ../datasets/pkl/unified_expert_gold.pkl ../datasets/pkl/expanded_gold.pkl 
echo