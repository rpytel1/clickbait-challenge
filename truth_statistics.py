import json

fake_clickbait = []
fake_non_clickbait = []

truth_file = 'data/clickbait-training/truth.jsonl'
with open(truth_file, encoding="utf-8") as f:
    for ind, line in enumerate(f):
        post = json.loads(line)
        if post["truthClass"] == 'clickbait':
            if post["truthMean"] < 0.5:
                fake_clickbait += [post["truthMean"]]
        else:
            if post["truthMean"] >= 0.5:
                fake_non_clickbait += [post["truthMean"]]

print('Fake clickbaits: ', fake_clickbait)


print('Fake non-clickbaits: ', fake_non_clickbait)