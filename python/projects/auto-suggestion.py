dataset = """
It is paramount to note that the availability and application of due process of law are equally important for both criminal justice system and the civil justice system. The reasoning behind this statement is that one of the main principles of this legal requirement is that every individual needs to be treated fairly.
It is understandable that many may think that it is more important for criminal justice system because punishments are much more severe in most cases, and it is essential to avoid any unfairness. However, it needs to be said that it is necessary to make sure that any case is reviewed equally, and it is an essence of the Law of the United States.
In my opinion, a defendant in a civil lawsuit should be afforded the same measure of due process as a defendant in a criminal trial because a particular disparity in treatment is currently present. The central issue that is worth mentioning is that the rights of those who are suspected of a crime are rather limited during the pre-trial stages, and this aspect should not be disregarded.
"""
ignoreset = [".", ",", "!", "'", '"', "?"]

def chunkdata(data):
    words = []
    currentword = ""
    for letter in data:
        if letter == " ":
            words.append(currentword)
            currentword = ""
        elif letter not in ignoreset:
            currentword += letter

    if currentword != "":
        words.append(currentword)
    return words

def pairset(words):
    pairs = []
    currentPair = []
    currentIndex = 0

    for word in words:
        if currentIndex == 2:
            pairs.append(currentPair)
            currentPair = []
            currentIndex = 0
        currentPair.append(word)
        currentIndex += 1
    if len(currentPair) != 0:
        pairs.append(currentPair)
    
    return pairs

def suggest(word, pairs):
    indexPos = 0
    for pair in pairs:
        if word == pair[0]:
            return pair[1]
        elif word == pair[1]:
            return pairs[indexPos+1][0]
        indexPos += 1

words = chunkdata(dataset)
pairs = pairset(words)

userinput = ""

while userinput.lower() != "end":
    userinput = input("Enter a sentence or word (enter end to stop): ")
    userWords = chunkdata(userinput)
    lastWord = userWords[len(userWords) - 1]
    suggestion = suggest(lastWord, pairs)
    if suggestion:
        print(f"{userinput} {suggestion}?")
    else:
        print("No suggestion found")
