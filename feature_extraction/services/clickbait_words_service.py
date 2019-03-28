from nltk import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
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


def get_clickbait_words_features(entry):
    num_clickbaits_in_post = 0
    words = [x for x in tokenizer.tokenize(entry["postText"][0])]
    for word in words:
        if word in clickbait_words:
            num_clickbaits_in_post += 1
    return num_clickbaits_in_post


def get_feat_names():
    return "num_of_clickbait words"

