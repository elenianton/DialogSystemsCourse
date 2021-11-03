import os
import os.path
import pandas as pd
import re
import statistics
from nltk.tokenize import word_tokenize

#define ".txt" file as 'corpus_path'
corpus_path = os.path.abspath("dialog-babi-task5-full-dialogs-trn.txt")

#%%

new_str=[]
with open(corpus_path) as f:
    lines = f.readlines()    
    df= pd.DataFrame(lines) #create a dataframe 
    rows= df[0]     
    #print(rows)
    for row in rows:
       if '\t' in row:  #remove tabs
           #print(row)
           new_str.append(row) #append rows with tab to a new list
       else:
           #print('No tab')
           row.rstrip() #remove rows without tabs

#%% Define columns of dataframe, define utterances  
           
user=[]
chatbot=[]
list_num=[]
user_final=[]
new_str_1= ""
for str in new_str:
    new_str_1 += str
    a,b= str.split('\t')
    user.append(a) #user utterances
    chatbot.append(b) #machine utterances
    num=re.findall('\d+',a)
    list_num.append(num[0]) #stands for dialog turns
    
    user_final_a= a.replace(num[0]+(' '), '')
    user_final.append(user_final_a)

#%% create the final dataframe named 'final', drop "<SILENCE>" from utterances
final = pd.DataFrame(
    {'Number': list_num,
     'User': user_final,
     'Chatbot': chatbot
    })  


index = final.index
condition = final["User"] == "<SILENCE>"
silence_indeces = index[condition]


for i in silence_indeces:
    final.loc[i-1][2]= final.loc[i-1][2] + " " + final.loc[i][2]


df = final.drop([i for i in silence_indeces]) #drop 'silence' indeces from final dataframe

#%% calculate number of turns per dialogue

dialogues=0
turns=0
total_turns_list=[]
total_dialogues_list=[]
for index, row in df.iterrows():
    if row['Number'] == "1":
        total_turns_list.append([turns])
        turns= 1
        dialogues +=1   
        total_dialogues_list.append(dialogues)
    else:
        turns+=1
             
total_turns_list.append([turns])
total_turns_list= total_turns_list[1:]


#%% Define sentences in a list in order to compute statistics
sentences_list=[]

for index, row in df.iterrows():
    sentences_list.append(row['User'])
    sentences_list.append(row['Chatbot'])

tokenized_sents = [word_tokenize(i) for i in sentences_list] #tokenize sentences
words=0
total_words_list=[]
for sent in tokenized_sents:
    for word in sent:
        word = re.sub(r'[^\w\s]', '', word) #remove punctuation
        words+=1
        #print(word)
        

total_words_list.append(words) 
print('Total number of words: ', words)
print('Total number of dialogues: ', dialogues)
print("--------------------------------------------------------------------")

#print(total_turns_list)        

#%% Mean and standar deviation     
flat_list_turns = [item for sublist in total_turns_list for item in sublist]

print('Mean for number of turns:', statistics.mean(flat_list_turns))
print("Standard Deviation of turns:", (statistics.stdev(flat_list_turns)))
print("--------------------------------------------------------------------")

words_per_sent_list=[]
for sent in tokenized_sents:
     number_of_words = len(sent)
     words_per_sent_list.append(number_of_words)

#per sentence
print('Mean for number of words per turn:', statistics.mean(words_per_sent_list))
print("Standard Deviation of words per turn:", (statistics.stdev(words_per_sent_list)))   
print("--------------------------------------------------------------------")

#%% Number of unique words (vocabulary)
vocabulary = []
i = 0
for words in tokenized_sents:
    for word in words:
        if word not in vocabulary:
            vocabulary.append(word)
            i+=1
       
print('Vocabulary size: ', i)    
