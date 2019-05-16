# README - Datasets

## Organization

The .pkl files contained in this folder are organized as follows:

* Each pickle file contains a dict with the following keys: 'coms', 'labels', and 'indices'. Each of these keys points to a separate dict file, whose keys are a set of function ids (FIDs). Each of the three dicts in a fild contain the same set of FID keys.

	* Entries in 'coms' refer to the processed function associated with the fid (e.g. "this function adds the two inputs").
	
	* Entries in 'labels' refer to the set of labels produced by the specified population for the corresponding comment (e.g. "adds the two inputs"). If an annotator provided an empty label, the label will be an empty string.
	
	* Entries in 'indices' refer to the start and end indices of a corresponding label of a corresponding comment. These are stored as a tuple. Note that the end index refers to the index of the last word in the label, rather than the boundary of the label (e.g. (2, 5)). If an annotator provided an empty label, the indices will be (-1, -1).

* Files with filenames ending in "gold" include only results for the functions in the gold set (produced by the population specified in the filename).

* The files with filenames beginning in "unified", as well as the files containing bilstm model results ("bilstm_f_final.pkl" and "bilstm_final.pkl"), contain only one label and set of indices for each fid. For them, the FID keys in the 'labels' and 'indices' each point to a single string/tuple entry (respectively).

* The remainder of the files contain annotations produced by multiple annotators. Each fid in the 'labels' and 'indices' dicts therefore points to another dict, in which the keys are user ids and the values are the results corresponding to that user.

* The function_data.pkl file provides data associated with each FID, including the raw source code, the raw comment text, the name of the file the function was found in, and the unique id of the project the function was found in

## Overview: 

For **bilstm_f_final.pkl**, **bilstm_final.pkl**, **unified_controlled_gold.pkl**, **unified_controlled.pkl**, and **unified_expert_gold.pkl**:

data = { "coms": coms, "labels":labels, "indices":indices}
        coms = {fid: "Function comment"}
        labels = {fid: "portion of comment tagged by method} 
        indices = {fid: [start index, end index]}

For **expert_gold.pkl**, **controlled.pkl**, **controlled_gold.pkl**, **expanded.pkl**, and **expanded_gold.pkl**:

data = { "coms": coms, "labels":labels, "indices":indices}
        coms = {fid: "Function comment"}
        labels = {fid: {user: "portion of comment tagged by user"}} 
        indices = {fid: {user: [start index, end index]}}  

For **function_data.pkl** (see below): 
data = {fid: {'filename': "filename.c", 'raw_com': "/*Raw comment text*/", 'raw_src': "Raw source code", 'pid': uniquePprojectId}}

## Function data

We include a trimmed-down version of the dataset provided by LeClair et al. (2018) in a file called **"function_data.pkl"**, which consists of relevant data for all functions from the aforementioned dataset which had non-empty comments (~.7m). The various sets annotated in our investigations were drawn from from this set. This file is too large to host on GitHub; you may function_data.pkl [here](https://drive.google.com/file/d/1PjoLmbEaBGd-dNMkYxj3Ggv8db-hurkS/view?usp=sharing), or contact the authors of this paper.
