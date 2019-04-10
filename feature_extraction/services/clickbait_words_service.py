import re
clickbait_words = [
    "A Single",
    "Absolutely",
    "Amazing",
    "Awesome",
    "Best",
    "Breathtaking",
    "But what happened next",
    "Can change your life",
    "Can't Even Handle",
    "Can't Handle",
    "Cannot Even Handle",
    "Doesn't want you to see",
    "Epic",
    "Everything You Need To Know",
    "Gasp-Worthy",
    "Go Viral",
    "Greatest",
    "Incredible",
    "Infuriate",
    "Literally",
    "Mind Blowing",
    "Mind-Blowing",
    "Mind BLOWN",
    "Mind Blown",
    "Need To Visit Before You Die",
    "Nothing Could Prepare Me For",
    "Of All Time",
    "Of All Time",
    "Of All-Time",
    "OMG",
    "One Weird Trick",
    "Perfection",
    "Priceless",
    "Prove",
    "Right Now",
    "Scientific Reasons",
    "Shocked",
    "Shocking",
    "Simple Lessons",
    "Stop What You're Doing",
    "Stop What You&#8217;re Doing",
    "TERRIFYING",
    "Terrifying",
    "That Will Make You Rethink",
    "The World's Best",
    "This Is What Happens",
    "Totally blew my mind",
    "Unbelievable",
    "Unimaginable",
    "WHAT?",
    "Whoa",
    "WHOA",
    "Whoah",
    "Will Blow Your Mind",
    "Will Change Your Life Forever",
    "Won the Internet",
    "Wonderful",
    "Worst",
    "Wow",
    "WOW",
    "You Didn't Know Exist",
    "You Didn't Know Existed",
    "You Didn’t Know Exist",
    "You Didn’t Know Existed",
    "You Didn&#8217;t Know Exist",
    "You Didn&#8217;t Know Existed",
    "You Won't Believe",
    "You Won’t Believe",
    "You Won&#8217;t Believe",
    "You Wont Believe",
    "Have To See To Believe"
]


clickbait_regexp = [
    r"\b(?:Top )?(?:(?:\d+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty|Thirty|Forty|Fourty|Fifty|Sixty|Seventy|Eighty|Ninety|Hundred)(?: |-)?)+ Things",
	r"\b[Rr]estored [Mm]y [Ff]aith [Ii]n [Hh]umanity\b",
	r"\b[Rr]estored [Oo]ur [Ff]aith [Ii]n [Hh]umanity\b",
	r"\b(?:Top )?(?:(?:\d+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty|Thirty|Forty|Fourty|Fifty|Sixty|Seventy|Eighty|Ninety|Hundred)(?: |-)?)+ Weird",
	r"\b^(?:Is|Can|Do|Will) (?:.*)\?\B",
    r"\b^(?:[Rr]easons\s|[Ww]hy\s|[Hh]ow\s|[Ww]hat\s[Yy]ou\s[Ss]hould\s[Kk]now\s[Aa]bout\s)(?:.*)\b$",
    r"\bThe Best[\s\w+]+\sEver\b"
]

# function calculating no of occurances of clickbait specific words
def get_clickbait_words_features(entry):
    num_clickbaits_in_post = 0
    num_clickbait_patterns_in_post = 0
    for w in clickbait_words:
        num_clickbaits_in_post += entry['postText'][0].count(w)*len(w)
    for reg_exp in clickbait_regexp:
        num_clickbait_patterns_in_post += sum(list(map(lambda x: len(x), re.findall(reg_exp, entry['postText'][0]))))
    return 0 if not entry['postText'][0] else num_clickbaits_in_post/len(entry['postText'][0]), \
           0 if not entry['postText'][0] else num_clickbait_patterns_in_post/len(entry['postText'][0])


def get_feat_names():
    return "num_of_clickbait_words", "num_clickbait_patterns_in_post"

