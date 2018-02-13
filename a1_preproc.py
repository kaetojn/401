import spacy
import sys
import argparse
import os
import json
import html.parser
import re
import string


indir = '/u/cs401/A1/data/';

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = ''
    nlp = spacy.load('en', disable=['parser', 'ner'])
    if 1 in steps:
        #Remove all newline characters
        comment = comment.replace('\n', ' ').replace('\r', '')
    if 2 in steps:
        #Replace HTML character codes
        html_parser = html.parser.HTMLParser()
        comment = html_parser.unescape(comment)
    if 3 in steps:
        #Remove all URLs
        comment = re.sub(r'^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$', '', comment, flags=re.MULTILINE)
        
    if 4 in steps:
        #Add whitespace before punctuations
        y = string.punctuation
        x = 0
        i = 0

        while(i+1 <= len(comment)):
            if(comment[i] in y):
                if(i+1 < len(comment)):
                    #Not End of String (Multiple Punctuation)
                    if(comment[i+1] in y):
                        x = i
                        try:
                            while(comment[i] in y):
                                i += 1
                            comment = comment[:x] + " " + comment[x:i] + " " + comment[i:]
                            i+=2
                        #End of String (Multiple Punctuation)
                        except IndexError:
                            comment = comment[:x] + " " + comment[x:] + " "
                            i+=1
                            pass


                    #Not End of String (Single Punctuation)
                    else:
                        #Exceptions to split (Abbreviation, Hypen, Apostrophe)
                        if((comment[i] == '.') & (comment[i+1] != ' ')):
                            i += 1
                        elif((comment[i] == '-') & (comment[i+1] != ' ')):
                            i += 1
                        elif(((comment[i] == '\'') & (comment[i+1] != ' ')) or ((comment[i] == '\'') & (comment[i-1] in string.ascii_letters))):
                            i += 1
                        else:
                            comment = comment[:i] + ' ' + comment[i] + ' ' + comment[i+1:]
                            i += 2

                #End of String (Single Punctuation)
                else:
                    #Only case you dont tokenize up punctuation at the end of sentence
                    if(comment[i] == '\''):
                        i += 1
                    #Split any other punctuation
                    else:
                        comment = comment[:i] + " " + comment[i:] + " "
                        i += 2
            else:
                i += 1
    if 5 in steps:
        #Add whitespace to clitics
        y = string.ascii_letters
        i = 0
        while(i+1 <= len(comment)):
            if(comment[i] == '\''):
                #Not End of String
                if(i+1 < len(comment)):
                    # possesive singular 's (it's)
                    if((comment[i+1] == 's')):
                        comment = comment[:i] + ' ' + comment[i:]+  ' '
                        i+=2
                    # possesive plural s' (dogs')
                    elif((comment[i-1] == 's')):
                        comment = comment[:i] + ' ' + comment[i ]+  ' '
                        i+=2
                    #Clitic (e.g., don't)
                    elif((comment[i+1] in y)):
                        comment = comment[:i-1] + ' ' + comment[i-1:] + ' '
                        i+=2
                    else:
                        i+=1
                #End of String                
                else:
                     if(comment[i] == '\''):
                        comment = comment[:i] + ' ' + comment[i]
                        i+=2   
            else:   
                i += 1
    lists = []
    if 6 in steps:
        #Tagging Token
        doc = spacy.tokens.Doc(nlp.vocab, words=comment.split())
        doc = nlp.tagger(doc)

        
        for token in doc:
            x = token.text + '/' + token.tag_
            lists.append(x)
        comment = ' '.join(lists)
    if 7 in steps:
        #Removing StopWords
        x = []
        z = comment.split()
        
        for i in range(len(z)):
            if z[i] in open('/u/cs401/Wordlists/StopWords').read().splitlines():
                x.append(i)
        for index in sorted(x, reverse=True):
            del z[index] 
        
        comment = ' '.join(z)
    if 8 in steps:
        #Lematization
        z = comment.split()
        y = []
        for i in z:

            delimiter = i.index('/')
            word = i[:delimiter]
            tag = i[delimiter:]
            doc = nlp(word)

            for token in doc:
                lemma =  token.lemma_
                if lemma[0] == '-':
                    i = word + tag
                else:
                    i = lemma + tag
                y.append(i)
        comment = ' '.join(y)
    if 9 in steps:
        #Add a newline between each sentence
        x = []
        z = comment.split()
        for i in range(len(z)):
            delimiter = z[i].index('/')
            word = z[i][:delimiter]
            tag = z[i][delimiter:]
            
            if((word == '.')):
                y = z[i] + "\n"
                x.append(y)
            elif (((tag == '.')) and (word not in open('/u/cs401/Wordlists/abbrev.english').read().splitlines())):

                y = z[i] + "\n"
                x.append(y)
            else:
                x.append(z[i])
        comment = ' '.join(x)
    if 10 in steps:
        #Lowercase everything
        comment = comment.lower()
    modComm = comment
    return modComm

def main( args ):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))
            counter = 0
            
            for i in range(len(data)):
                counter+=1

                #Subsampling Circular List
                z = (args.ID[0]+i) % len(data)
                
                #encoding unicode lines to string
                data[z] = data[z].encode('utf8')

                #reading lines
                j = json.loads(data[z])

                #adding a field to each selected line called 'cat' with the value of 'file'
                j['cat'] = file

                #process the body field with preproc1
                modComm = preproc1(j['body'], range(1,11))

                #replace the 'body' field with the processed text
                j['body'] = modComm

                #choose to retain fields from those lines that are relevant to you
                wanted_keys = ['id', 'body', 'cat'] # The keys you want
                mydict = dict((k, j[k]) for k in wanted_keys if k in j)

                # append the result to 'allOutput'
                allOutput.append(mydict)
                
                #args.max
                if counter == int(args.max):
                    break


            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1, help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (int(args.max) > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
        
    main(args)
