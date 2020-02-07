import pandas as pd
from config import THRESHOLD

def submit(test_img_paths, test_predictions):
    submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})
    submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > THRESHOLD else 'cleaned')
    submission_df['id'] = submission_df['id'].str.replace('test/unknown/', '')
    submission_df['id'] = submission_df['id'].str.replace('.jpg', '')
    submission_df.set_index('id', inplace=True)
    return submission_df