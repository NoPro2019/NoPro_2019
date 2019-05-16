
s# README - Metrics

## Description 
Each of these scripts runs one of the metrics described in the paper and prints the results to the standard output. They each take two datasets as inputs and compare all common FIDs between the two functions. They are designed to work with pickle data formatted like the files in datasets/pkl/ - in other words, they expect a pickled dict with three top-level dicts (coms, labels, indices), each of which are organized by FID. They automatically check whether there is only one response per FID, or multiple. If multiple, scores by default will be calculated by averaging the results of the multiple hypotheses for each FID, and averaging those average scores. There is an option to instead analyze the results on an annotator-by-annotator level.

## Usage

ROUGE:
python3 rouge_metric.py [-h] [-i] [-ne] reference hypothesis

Empty classification:
python3 empty_classification_metric.py [-h] [-i] reference hypothesis

Krippendorff's alpha:
python3 krippendorff_metric.py [-h] [-i] reference hypothesis

* reference - pickle file containing "reference" labels 
* hypothesis - pickle file containing "hypothesis" labels
* -h - Display help message
* -i - Compare individual annotators
* -ne - Compare only non-empty hypotheses and references (ROUGE metrics only)