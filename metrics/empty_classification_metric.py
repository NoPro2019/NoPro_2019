import pickle
import argparse
from statistics import mean, median, stdev

'''
	This file contains the code used to calculate classification metrics,
    as reported in the paper. Compare two sets from the terminal
    by running:
    python3 empty_classification_metric.py ref_file hyp_file
    If you have multiple hypotheses and would like to compare individual users, use the -i flag:
    python3 empty_classification_metric.py ref_file hyp_file -i
'''

def prepare_results(p, r, f, metric):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def average_scores(scores):
    if len(scores)==0: return None
    output = {}
    for rouge_type in scores[0]:
        output[rouge_type] = {}
        for metric in scores[0][rouge_type]:
            output[rouge_type][metric] = mean([score[rouge_type][metric] for score in scores if score])
    return output

def empty_score(hyps, refs):
    tp = tn = fp = fn = 0
    for i in range(len(refs)):
        if refs[i] == "":
            if hyps[i] == "": tp += 1
            else: fn += 1
        else:
            if hyps[i] == "": fp += 1
            else: tn += 1
    scores = {"is-empty": {"tp": tp, "fp": fp, "tn":tn, "fn":fn}}
    return scores

def get_empty_score(hyps, refs):
    mult_hyps = not isinstance(hyps[0], str)
    if mult_hyps:
        scores = []
        for i in range(len(refs)):
            ref_scores = [empty_score([hyps[i][user]], [refs[i]]) for user in hyps[i]]
            scores.append(average_scores([score for score in ref_scores if score != None]))
        scores = average_scores([score for score in scores if score != None])
    else:
        scores = empty_score(hyps, refs)

    tp = scores["is-empty"]["tp"]
    fp = scores["is-empty"]["fp"]
    fn = scores["is-empty"]["fn"]
    if tp!=0:
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f = 2*p*r/(p+r)
    else: p=r=f=0
    scores = {"is-empty":{"p":p, "r": r, "f":f}}

    return scores

def get_indiv_empty_score(hyps, refs):
    users = set([user for hyp in hyps for user in hyp])
    hyps = {user: [labels[user] if user in labels else None for labels in hyps] for user in users}
    all_scores = {}
    for user in hyps:
        user_refs = [refs[i] for i in range(len(refs)) if hyps[user][i] != None]
        user_hyps = [hyps[user][i] for i in range(len(hyps[user])) if hyps[user][i] != None]
        if '' not in user_refs: continue
        user_score = get_empty_score(user_hyps, user_refs)
        if user_score: all_scores[user] = user_score
    empty_scores = [score for score in all_scores.values() if score]
    empty_max = {emptytype: {metric: max([empty_scores[i][emptytype][metric] for i in range(len(empty_scores))])for metric in empty_scores[0][emptytype]} for emptytype in empty_scores[0]}
    empty_min = {emptytype: {metric: min([empty_scores[i][emptytype][metric] for i in range(len(empty_scores))])for metric in empty_scores[0][emptytype]} for emptytype in empty_scores[0]}
    empty_mean = {emptytype: {metric: mean([empty_scores[i][emptytype][metric] for i in range(len(empty_scores))])for metric in empty_scores[0][emptytype]} for emptytype in empty_scores[0]}
    empty_median = {emptytype: {metric: median([empty_scores[i][emptytype][metric] for i in range(len(empty_scores))])for metric in empty_scores[0][emptytype]} for emptytype in empty_scores[0]}
    empty_stddev = {emptytype: {metric: stdev([empty_scores[i][emptytype][metric] for i in range(len(empty_scores))])for metric in empty_scores[0][emptytype]} for emptytype in empty_scores[0]}
    
    return(empty_max, empty_min, empty_mean, empty_median, empty_stddev)

def print_scores(scores):
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        print(prepare_results(results['p'], results['r'], results['f'], metric))

def print_indiv_scores(scores):
    print("Max:")
    print_scores(scores[0])
    print("Min:")
    print_scores(scores[1])
    print("Mean:")
    print_scores(scores[2])
    print("Median:")
    print_scores(scores[3])
    print("Stdev:")
    print_scores(scores[4])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('reference', type=str)
    parser.add_argument('hypothesis', type=str)
    parser.add_argument('-i', action='store_true', default=False)

    args = parser.parse_args()
    refs = pickle.load(open(args.reference, "rb"))
    hyps = pickle.load(open(args.hypothesis, "rb"))
    indiv = args.i

    fid_list = list(set(refs['coms'].keys()) & set(hyps['coms'].keys()))

    # print(refs['labels'])
    # print(fid_list)
    refs = [refs['labels'][fid] for fid in fid_list]
    hyps = [hyps['labels'][fid] for fid in fid_list]

    if indiv: 
        print_indiv_scores(get_indiv_empty_score(hyps, refs))
    else:
        print_scores(get_empty_score(hyps, refs))
