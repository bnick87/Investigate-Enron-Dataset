#!/usr/bin/env python
# coding: utf-8

# In[18]:


#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        translator=str.maketrans('','',string.punctuation)
        text_string = content[1].translate(translator)

        ### project part 2: comment out the line below
        ##words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english")
        words = ' '.join([stemmer.stem(word) for word in text_string.split()])

    return words

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print (text)



if __name__ == '__main__':
    main()
    ##https://kite.com/python/answers/how-to-convert-each-line-in-a-text-file-into-a-list-in-python
    ##https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate/34294398#:~:text=You%20have%20to%20create%20a,punctuation%20you%20want%20to%20None%20.&text=instead%2C%20or%20create%20a%20table%20as%20shown%20in%20the%20other%20answer.

