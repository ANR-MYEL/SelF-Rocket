import numpy as np

def highest_median_voting(ballots, candidates):
    scores = {candidate: [] for candidate in candidates}
    for ballot in ballots:
        for candidate, score in ballot.items():
            if candidate in candidates:
                scores[candidate].append(score)

    medians = {}

    for candidate, candidate_scores in scores.items():
        if candidate_scores:
            medians[candidate] = np.median(candidate_scores)
        else:
            medians[candidate] = 0 

    winner = max(medians, key=medians.get)
    
    return winner