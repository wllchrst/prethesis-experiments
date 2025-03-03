Models that we have trained on english dataset has tremendous results, to find out why this is we tested the dataset where we drop data from 2 labels that has the least data to find out the relations.

Previous week augmentation result from the function is up to 8, now for this week experiment the amount of sentence that can be augmented from a single sentence is limited to 3.

Text Augmentation for Indonesian Dataset we are going to use augmentation technique Synonym Replacement, Back Translation, and Random Deletion.  The method is based on a paper from this link: https://jtiik.ub.ac.id/index.php/jtiik/article/view/7325, author have given where the synonym data came from but the website is down so we opted to use synonyms data from this GitHub repository: https://github.com/adnanzulkarnain/Tesaurus-Bahasa.

Ideas For Next Experiment:
- Indonesia dataset has other attributes not just text and label, like Product price, overall rating, total review that I think plays a big role in the sentimental value of the data, maybe we can use this other attributes for training the dataset.

Results:
- English dataset (twitter and hugging face dataset) shows really high accuracy when dropping 2 labels.
- Indonesian dataset shows really high overfitting (even though we got better accuracy).

Research on finding models that is trending, and is being use really often is stiill in theme of BERT where i find a  list of models like:
- FinBert
- XLM-RoBERTa
- Distli-RoBERTa
- and many more BERT things

Interesting Models:
- Indonesia DistilBERT https://huggingface.co/fathurfrs/indonesia-distilledbert-sentiment-classification
- 
