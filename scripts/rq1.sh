echo RQ1: Comparing individual experts to the Unified Expert Gold Set.
echo -----------------------------------------------------------------
echo Text similarity as measured by ROUGE Scores \(comparing hypotheses to all non-empty references\):
echo
python3 ../metrics/rouge_metric.py ../datasets/pkl/unified_expert_gold.pkl ../datasets/pkl/expert_gold.pkl -i
echo
echo Text similarity as measured by ROUGE Scores \(comparing only non-empty hypotheses to non-empty references\):
echo
python3 ../metrics/rouge_metric.py ../datasets/pkl/unified_expert_gold.pkl ../datasets/pkl/expert_gold.pkl -i -ne
echo
echo \"Is-empty?\" binary classification similarity:
echo
python3 ../metrics/empty_classification_metric.py ../datasets/pkl/unified_expert_gold.pkl ../datasets/pkl/expert_gold.pkl -i
echo
echo Unitization similarity as measured by Krippendorff\'s unitized alpha
echo
python3 ../metrics/krippendorff_metric.py ../datasets/pkl/unified_expert_gold.pkl ../datasets/pkl/expert_gold.pkl -i
echo