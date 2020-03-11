import requests
from bs4 import BeautifulSoup
from gensim.summarization import summarize
from pprint import pprint
url_list = ["https://www.latimes.com/california/story/2020-03-09/coronavirus-cases-in-california-rise-to-133-here-is-what-you-need-to-know","https://nypost.com/2020/03/09/new-jersey-coronavirus-patient-thinks-he-caught-it-at-times-square-hotel/","https://www.kvue.com/article/news/health/coronavirus/coronavirus-man-traveling-from-austin-to-india-tests-positive/269-77511074-d840-456e-b846-419adf96a861","https://newyork.cbslocal.com/2020/03/09/coronavirus-update-new-jersey-patient-speaks-out/"]
test_url = url_list[3]

page = requests.get(test_url).text
soup = BeautifulSoup(page, "html5lib")

headline = soup.find('h1').get_text()

p_tags = soup.find_all('p')
# Get the text from each of the “p” tags and strip surrounding whitespace.
p_tags_text = [tag.get_text().strip() for tag in p_tags]

sentence_list = [sentence for sentence in p_tags_text if not '\n' in sentence]
sentence_list = [sentence for sentence in sentence_list if '.' in sentence]
# Combine list items into string.
article = ' '.join(sentence_list)

summary = summarize(article, ratio=0.3)

with open('/Users/marcdelacruz/PycharmProjects/INFO3700HonorsProject/output1.txt','w') as f:
    f.writelines(summary)
