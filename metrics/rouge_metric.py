import rouge
import pickle
import argparse
from statistics import mean, median, stdev

'''
    This file contains the code used to calculate classification metrics,
    as reported in the paper. Compare two sets from the terminal
    by running:
    python3 rouge_metric.py ref_file hyp_file
    If you have multiple hypotheses and would like to compare individual users, use the -i flag:
    python3 rouge_metric.py ref_file hyp_file -i
    If you would like to consider only cases where the hypothesis and reference are not empty, use the -ne flag:
     python3 rouge_metric.py ref_file hyp_file -ne
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

def rouge_score(hyps, refs, ne):
    hyps = [hyps[i] for i in range(len(hyps)) if len(refs[i])>0]
    refs = [refs[i] for i in range(len(refs)) if len(refs[i])>0]
    if ne:
        refs = [refs[i] for i in range(len(refs)) if len(hyps[i])>0]
        hyps = [hyps[i] for i in range(len(hyps)) if len(hyps[i])>0]

    if len(hyps) == 0: return None
    
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                   max_n=4,
                   limit_length=True,
                   length_limit=100,
                   length_limit_type='words',
                   alpha=.5, # Default F1_score
                   weight_factor=1.2)

    scores = evaluator.get_scores(hyps, refs)
    return scores

def get_rouge_score(hyps, refs, ne):
    mult_hyps = not isinstance(hyps[0], str)
    if mult_hyps:
        scores = []
        for i in range(len(refs)):
            ref_scores = [rouge_score([hyps[i][user]], [refs[i]], ne) for user in hyps[i]]
            scores.append(average_scores([score for score in ref_scores if score != None]))
        scores = average_scores([score for score in scores if score != None])
    else:
        scores = rouge_score(hyps, refs, ne)
    return scores

def get_indiv_rouge_score(hyps, refs, ne):
    users = set([user for hyp in hyps for user in hyp])
    hyps = {user: [labels[user] if user in labels else None for labels in hyps] for user in users}
    all_scores = {}
    for user in hyps:
        user_refs = [refs[i] for i in range(len(refs)) if hyps[user][i] != None]
        user_hyps = [hyps[user][i] for i in range(len(hyps[user])) if hyps[user][i] != None]
        user_score = rouge_score(user_hyps, user_refs, ne)
        if user_score: all_scores[user] = user_score
    rouge_scores = [score for score in all_scores.values() if score]
    rouge_max = {rougetype: {metric: max([rouge_scores[i][rougetype][metric] for i in range(len(rouge_scores))])for metric in rouge_scores[0][rougetype]} for rougetype in rouge_scores[0]}
    rouge_min = {rougetype: {metric: min([rouge_scores[i][rougetype][metric] for i in range(len(rouge_scores))])for metric in rouge_scores[0][rougetype]} for rougetype in rouge_scores[0]}
    rouge_mean = {rougetype: {metric: mean([rouge_scores[i][rougetype][metric] for i in range(len(rouge_scores))])for metric in rouge_scores[0][rougetype]} for rougetype in rouge_scores[0]}
    rouge_median = {rougetype: {metric: median([rouge_scores[i][rougetype][metric] for i in range(len(rouge_scores))])for metric in rouge_scores[0][rougetype]} for rougetype in rouge_scores[0]}
    rouge_stddev = {rougetype: {metric: stdev([rouge_scores[i][rougetype][metric] for i in range(len(rouge_scores))])for metric in rouge_scores[0][rougetype]} for rougetype in rouge_scores[0]}
    return(rouge_max, rouge_min, rouge_mean, rouge_median, rouge_stddev)

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
    parser.add_argument('-ne', action='store_true', default=False)

    args = parser.parse_args()
    refs = pickle.load(open(args.reference, "rb"))
    hyps = pickle.load(open(args.hypothesis, "rb"))
    indiv = args.i
    ne = args.ne

    fid_list = list(set(refs['coms'].keys()) & set(hyps['coms'].keys()))

    # print(refs['labels'])
    # print(fid_list)
    refs = [refs['labels'][fid] for fid in fid_list]
    hyps = [hyps['labels'][fid] for fid in fid_list]

    if indiv: 
        print_indiv_scores(get_indiv_rouge_score(hyps, refs, ne))
    else:
        print_scores(get_rouge_score(hyps, refs, ne))

      
