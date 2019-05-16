import pickle
import argparse
from statistics import mean, median, stdev


'''
	This file contains the code used to calculate classification metrics,
	as reported in the paper. Compare two sets from the terminal
	by running:
    python3 krippendorff_metric.py ref_file hyp_file
    If you have multiple hypotheses and would like to compare individual users, use the -i flag:
    python3 krippendorff_metric.py ref_file hyp_file -i
'''

def find_sub_list(sl,l):
    #If no label, return [-1,-1]
    sl = sl.split()
    l = l.split()
    if sl == []: return [-1, -1]
    sll=len(sl)
    #Finds the index in the comment of the word that matches the first word in the label
    #If a span of the label's length starting at that index has the right final word, it chooses those as the label indices
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1
    return [-2, -2]

def alpha_score(hyps, refs, coms):
    sample_agreements = []
    for i in range(len(refs)):
        labels = [hyps[i], refs[i]]
        agreement = sample_agreement(labels, coms[i])
        if agreement != None: sample_agreements.append(agreement)
    alpha = mean(sample_agreements)
    return alpha

def get_alpha_score(hyps, refs, coms):
    mult_hyps = not isinstance(hyps[0], str)
    if mult_hyps:
        scores = []
        for i in range(len(refs)):
            ref_scores = [alpha_score([hyps[i][user]], [refs[i]], [coms[i]]) for user in hyps[i]]
            scores.append(mean([score for score in ref_scores if score != None]))
        scores = mean([score for score in scores if score != None])
    else:
        scores = alpha_score(hyps, refs, coms)
    return scores

def get_indiv_alpha_score(hyps, refs, coms):
    users = set([user for hyp in hyps for user in hyp])
    hyps = {user: [labels[user] if user in labels else None for labels in hyps] for user in users}
    all_scores = {}
    for user in hyps:
        user_coms = [coms[i] for i in range(len(coms)) if hyps[user][i] != None]
        user_refs = [refs[i] for i in range(len(refs)) if hyps[user][i] != None]
        user_hyps = [hyps[user][i] for i in range(len(hyps[user])) if hyps[user][i] != None]
        user_score = get_alpha_score(user_hyps, user_refs, user_coms)
        if user_score: all_scores[user] = user_score
    alpha_scores = [score for score in all_scores.values() if score]
    alpha_max = max(alpha_scores)
    alpha_min = min(alpha_scores)
    alpha_mean = mean(alpha_scores)
    alpha_median = median(alpha_scores)
    alpha_stddev = stdev(alpha_scores)
    
    return(alpha_max, alpha_min, alpha_mean, alpha_median, alpha_stddev)


def sample_agreement(labels, com):
    com_len = len(com.split())
    tags = [find_sub_list(label, com) for label in labels]
    if -2 in tags: return None
    spans = []
    for i,tag in enumerate(tags):
        if -1 not in tag:
            spans += [[0,tag[0], 0, i], [tag[0], tag[1]+1, 1, i], [tag[1]+1, com_len, 0, i]]
        else:
            spans += [[0, com_len, 0, i]]
    observed = observed_disagreement(com_len, spans)
    expected = expected_disagreement(com_len, spans)
    if observed == 0: return 1
    else: return 1-observed/expected

def observed_disagreement(com_len, spans):
    m = len(set([span[3] for span in spans]))
    numerator = sum([delta_stat(span1, span2) for span1 in spans for span2 in spans if span1[3] != span2[3]])
    denomenator = m*(m-1)*(com_len**2)
    return numerator/denomenator

def delta_stat(t1, t2):
    l1 = t1[1] - t1[0]
    l2 = t2[1] - t2[0]
    r = 0
    if t1[2] == 1 and t2[2] == 1 and -l1 < t1[0]-t2[0] and t1[0]-t2[0] < l2:
        r= (t1[0]-t2[0])**2 + (t1[0] + l1 - t2[0] -l2)**2 
    elif t1[2] ==0 and t2[2] ==1 and l1 - l2 >= t2[0]-t1[0] and t2[0]-t1[0] >= 0 : r= l2**2
    elif t2[2] ==0 and t1[2] ==1 and l1 - l2 <= t2[0]-t1[0] and t2[0]-t1[0] <= 0 : r= l1**2
    else: r= 0
    return r

def expected_disagreement(com_len, spans):
    m = len(set([span[3] for span in spans]))
    L = com_len
    Nc = sum([1 for span in spans if span[2]==1])
    numerator = 0
    for span in spans:
        if span[2]!=1: continue
        l = span[1]-span[0]
        a = ((Nc-1)/3) * ((2*l**3) - (3*l**2) + l)
        b = 0
        for span2 in spans:
            if span2[2]==0:
                l2 = span2[1]-span2[0]
                if l2>=l:
                    b += (l2-l+1)
        b *= l**2
        numerator += a+b
    numerator *= (2/L)
    denomenator = m*L*(m*L-1) - sum([(span[1]-span[0])*(span[1]-span[0]-1) for span in spans if span[2]==1])
    return numerator/denomenator

def print_scores(alpha):
    print("\tAlpha: {}".format(alpha))

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

    coms = [refs['coms'][fid] for fid in fid_list]
    refs = [refs['labels'][fid] for fid in fid_list]
    hyps = [hyps['labels'][fid] for fid in fid_list]

    if indiv: 
        print_indiv_scores(get_indiv_alpha_score(hyps, refs, coms))
    else:
        print_scores(get_alpha_score(hyps, refs, coms))