#not finished

import PyPDF2
import os

#open pdf

path=os.path.dirname(os.path.abspath(__file__))
file_handle = open(path + '/' + "Sense-and-Sensibility-by-Jane-Austen.pdf", 'rb')
pdfReader = PyPDF2.PdfReader(file_handle)


frequency_table = {}


#function to clean and split the text into the words
def split_text(text):
    text=text.lower()
    words=text.split()
    return words

for page_number in range(len(pdfReader.pages)):
    page=pdfReader.pages[page_number]
    page_text = page.extract_text()


    words=split_text(page_text)

    for word in words:
        if word in frequency_table:
            frequency_table[word] += 1
        else:
            frequency_table[word] = 1


print(frequency_table)

file_handle.close()
