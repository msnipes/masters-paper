# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy
import numpy as np
import gensim
import operator
import pandas as pd
import re 
import xml.etree.ElementTree as ET
import nltk
import sys


from matplotlib import pyplot as plt
from gensim import corpora
from gensim.corpora import Dictionary
from gensim import models
from gensim.models import Phrases
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.preprocessing import remove_stopwords
from gensim.test.utils import common_corpus
from nltk.stem import WordNetLemmatizer


"""import XML"""

tree=ET.parse('LISS2012to2017.xml')
root=tree.getroot()

paperdata={}

i=0
for child in root: i+=1

"""XML to dictionary"""

for j in range(i):
    
    rec=root[j]
    
    """establish XML location"""
    header=rec[0]
    controlInfo=header[0]
    pubinfo=controlInfo[2]
    dt=pubinfo[0]
    artinfo=controlInfo[3]
    tig=artinfo[3]
    atl=tig[0]
    aug=artinfo[4]
    au=aug[0]
    sug=artinfo[5]
    ab=artinfo[6]
    displayinfo=header[1]
    plink=displayinfo[0]       
    url=plink[0]
    
    """create subjects list"""
    subjects=[]
    for child in sug: 
        subjects.append(child.text)
    
    """create dictionary entry"""
    paperdata[j]=[atl.text, au.text, dt.text, ab.text, subjects]


"""Dictionary to DataFrame""" 

rawdataflipped=pd.DataFrame(paperdata)
rawdata = rawdataflipped.transpose()
rawdata.columns=['title', 'author', 'year', 'abstract', 'subjects']
                        

 
"""Dictionary of Subjects"""

subjectdict={}

for i in range(536):
    tempsubjlist=rawdata["subjects"].loc[i]
    for j in range(len(tempsubjlist)):
        if tempsubjlist[j] in subjectdict:
            subjectdict[tempsubjlist[j]] +=1
        else:
            subjectdict[tempsubjlist[j]]=1

"""Subject List Reverse Frequency"""

subjectlist=[]
subjectkeylist=list(subjectdict.keys())

for i in range(len(subjectkeylist)):
    subjectlist.append((subjectkeylist[i], subjectdict[subjectkeylist[i]]))
 
subjectlist.sort(key=operator.itemgetter(1), reverse=True)   


"""Number of Subjects Distribution"""    

numsub=[]

for j in range(536): 
      num=len(paperdata[j][4])
      numsub.append(num)
     
dictnumbs={}     
for i in range(max(numsub)+1):
    dictnumbs[i]=numsub.count(i)
    
x=[]
y=[]

for i in dictnumbs:
    x.append(i)
    y.append(dictnumbs[i])
    
plt.title("Distribution of Subject Headings over Papers")
plt.xlabel("Number of Subject Headings")
plt.xlim(0,16)
plt.ylabel("Number of Papers")
plt.ylim(0,120)
plt.scatter(x,y)   



threetosix = (dictnumbs[3] + dictnumbs[4] + dictnumbs[5] + dictnumbs[6])/536
tenplus=(dictnumbs[10] + dictnumbs[11] + dictnumbs[12] + dictnumbs[13] + dictnumbs[14] + dictnumbs[15])/536



threeplus=[]

for i in range(len(subjectlist)):
    if subjectlist[i][1] >=3: 
        threeplus.append(subjectlist[i])

sixplus=[]
for i in range(len(subjectlist)):
    if subjectlist[i][1] >=6: 
        sixplus.append(subjectlist[i])
        


"""Top 20 Subject Headings"""        

subjectdf=pd.DataFrame(subjectlist, columns=['Subject Heading','Number of Papers'])

N=subjectdf.loc[19][1]
for i in range(len(subjectdf)):
    if subjectdf.loc[i][1] < N:
        end=i
        break
    
top20subjects=subjectdf[:end]
top20subjects.to_csv('Top_20_Subject_Headings.csv', index=False)



"""Functions for Processing Abstracts"""

additionalstopwords=["paper", "results", "literature", "conducted", "determine", "examines", "findings", "study", "north", "carolina", "chapel", "hill", "studies", "reflected"]
libraryterms=['library','librarianship','librarian','libraries','librarians']
wnl=WordNetLemmatizer()


def RemoveWords(a,b):
    c=[]
    for word in a:
        if word not in b:
            c.append(word)
    return (c)

def Lemmatize(a):
    for i in range(len(a)):
        a[i]=wnl.lemmatize(a[i])
    return(a)

def Tokenize(a):
   atemp=a.lower()
   atemp=re.sub(r'[0-9]','',atemp)
   atemp=re.sub(r'[^\w\s]','',atemp)
   atemp=remove_stopwords(atemp)
   atemp=atemp.split()
   atemp=RemoveWords(atemp,additionalstopwords)
   atemp=Lemmatize(atemp)
   return (atemp)  



"""Create Corpora"""


all_abstracts=[] 
library_abstracts=[]
non_library_abstracts=[]

for i in range(len(paperdata)):
    new_data=Tokenize(paperdata[i][3])
    all_abstracts.append(new_data)
    
    for i in range(len(libraryterms)):
        if libraryterms[i] in new_data:
            library_abstracts.append(new_data)
            break
        
    else:
        non_library_abstracts.append(new_data)
            
        

"""LDA Model - All Abstracts, 20 Topics"""

overalldict=corpora.Dictionary(all_abstracts)
overalldict.filter_extremes(no_above=.4)
overalldict.filter_extremes(no_below=10)
overallcorpus=[overalldict.doc2bow(text) for text in all_abstracts]


lda_overall=models.LdaModel(overallcorpus, num_topics=20, id2word=overalldict)


original_stdout=sys.stdout
all_abstracts_file=open('LDA_All_Abstracts.txt', 'w+')
sys.stdout=all_abstracts_file

for i, topic in lda_overall.show_topics(formatted=True, num_topics=20):
    topic=re.sub(r'[0-9]','',topic)
    topic=re.sub(r'[^\w\s]','',topic)
    print("Topic " + str(i+1)+": "+ topic)
    print()
all_abstracts_file.close() 
sys.stdout=original_stdout


"""LDA Model - Library Abstracts, 10 Topics"""

librarydict=corpora.Dictionary(library_abstracts)
librarydict.filter_extremes(no_above=.5)
librarydict.filter_extremes(no_below=5)
libcorpus=[librarydict.doc2bow(text) for text in library_abstracts]


lda_library=models.LdaModel(libcorpus, num_topics=10, id2word=librarydict)

library_abstracts_file=open('LDA_Library_Abstracts.txt','w+')
sys.stdout=library_abstracts_file

for i, topic in lda_library.show_topics(formatted=True, num_topics=10):
    topic=re.sub(r'[0-9]','',topic)
    topic=re.sub(r'[^\w\s]','',topic)
    print("Topic " + str(i+1)+": "+ topic)
    print()
library_abstracts_file.close()
sys.stdout=original_stdout

"""LDA Model - Non-Library Abstracts, 10 Topics"""

nonlibrarydict=corpora.Dictionary(non_library_abstracts)
nonlibrarydict.filter_extremes(no_above=.5)
nonlibrarydict.filter_extremes(no_below=5)
nonlibcorpus=[nonlibrarydict.doc2bow(text) for text in non_library_abstracts]


lda_nonlibrary=models.LdaModel(nonlibcorpus, num_topics=10, id2word=nonlibrarydict)

non_library_abstracts_file=open('LDA_Non_Library_Abstracts.txt','w+')
sys.stdout=non_library_abstracts_file

for i, topic in lda_nonlibrary.show_topics(formatted=True, num_topics=10):
    topic=re.sub(r'[0-9]','',topic)
    topic=re.sub(r'[^\w\s]','',topic)
    print("Topic " + str(i+1)+": "+ topic)
    print()
non_library_abstracts_file.close()
sys.stdout=original_stdout


"""Analyze Topics"""

def DocTitles(a):
    doctitles=[]
    for i in range(len(all_abstracts)):
        if a in all_abstracts[i]:
           doctitles.append(paperdata[i][0])
    return(doctitles)
   
def NumDocs(a):
    counter=0
    for i in range(len(all_abstracts)):
        if a in all_abstracts[i]:
            counter +=1
    return(counter)







