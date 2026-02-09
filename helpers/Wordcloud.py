
#Script used to visualize our dataset as a word cloud to get an overview over content.

from datasets import load_dataset
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load MeetingBank
dataset = load_dataset("huuuyeah/meetingbank")
# summaries = [item["transcript"] for item in dataset["train"]]
summaries = [item["summary"] for item in dataset["train"]]


# Combine all summaries into one text
all_text = " ".join(summaries)

# Generate the word cloud
wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color="white",
    max_words=200,
    colormap='autumn'
).generate(all_text)

# Display
plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
